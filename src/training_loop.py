import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import numpy as np
from rouge_score import rouge_scorer
import os
import json
import logging
import bitsandbytes as bnb # For PagedAdamW8bit

from configs import finetune_config as cfg
from src.utils import get_main_device # Import get_main_device

logger = logging.getLogger(__name__)

class TrainingMetrics:
    def __init__(self):
        self.history = []
        self.best_metric_value = -float('inf') # Assuming higher is better (e.g., ROUGE-L)
        self.best_metric_name = 'eval_rougeL' # Metric to track for best model

    def update(self, log_entry):
        self.history.append(log_entry)
        current_metric = log_entry.get(self.best_metric_name, -float('inf'))
        if current_metric > self.best_metric_value:
            self.best_metric_value = current_metric
            return True # New best
        return False

    def save_history(self, filepath):
        cfg.ensure_dir_exists(filepath)
        with open(filepath, "w") as f:
            json.dump(self.history, f, indent=4)
        logger.info(f"Metrics history saved to {filepath}")


def calculate_rouge_scores(predictions_ids, label_ids, tokenizer):
    """Calculates ROUGE scores for generated predictions against labels."""
    # Replace -100 (ignore index) with pad_token_id for decoding labels
    label_ids = np.where(label_ids == -100, tokenizer.pad_token_id, label_ids)

    # Decode predictions and labels
    # Ensure all token IDs are non-negative before decoding
    predictions_ids = np.maximum(predictions_ids, 0) # Clip any negative values to 0 (e.g. UNK or pad)
    
    decoded_preds = tokenizer.batch_decode(predictions_ids, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_results = {'rouge1': [], 'rouge2': [], 'rougeL': []}

    for pred, label in zip(decoded_preds, decoded_labels):
        if not label:  # Skip if the ground truth label is empty after decoding
            continue
        if not pred:   # If prediction is empty, use a space to avoid errors in scorer
            pred = " "
        scores = scorer.score(target=label, prediction=pred)
        rouge_results['rouge1'].append(scores['rouge1'].fmeasure)
        rouge_results['rouge2'].append(scores['rouge2'].fmeasure)
        rouge_results['rougeL'].append(scores['rougeL'].fmeasure)

    avg_scores = {}
    for metric_name, values in rouge_results.items():
        if values:
            avg_scores[f'eval_{metric_name}'] = np.mean(values)
        else: # No valid scores, report 0
            avg_scores[f'eval_{metric_name}'] = 0.0
            logger.warning(f"No valid samples to calculate ROUGE for {metric_name}. Reporting 0.")
            
    return avg_scores


def evaluate_model(model, eval_dataloader, tokenizer, device, main_data_device):
    """Performs evaluation on the model and returns metrics."""
    logger.info("--- Starting Evaluation ---")
    model.eval()
    total_eval_loss = 0
    all_preds_ids = []
    all_label_ids = []

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating", leave=False):
            batch = {k: v.to(main_data_device) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(**batch)
            
            total_eval_loss += outputs.loss.item()
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            
            all_preds_ids.append(predictions.cpu().numpy())
            all_label_ids.append(batch["labels"].cpu().numpy())

    avg_eval_loss = total_eval_loss / len(eval_dataloader)
    
    all_preds_ids_np = np.concatenate(all_preds_ids, axis=0)
    all_label_ids_np = np.concatenate(all_label_ids, axis=0)

    rouge_metrics = calculate_rouge_scores(all_preds_ids_np, all_label_ids_np, tokenizer)
    
    metrics = {'eval_loss': avg_eval_loss, **rouge_metrics}
    logger.info(f"Evaluation Results: Loss = {avg_eval_loss:.4f}, "
                f"ROUGE-1 = {rouge_metrics.get('eval_rouge1', 0):.4f}, "
                f"ROUGE-L = {rouge_metrics.get('eval_rougeL', 0):.4f}")
    return metrics


def run_training_loop(model, tokenizer, tokenized_datasets):
    """Main custom training loop."""
    
    main_data_device = get_main_device(model) # Determine where to send data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # For scaler
    logger.info(f"Main data processing device: {main_data_device}")
    logger.info(f"Scaler/general compute device: {device}")

    # DataLoaders
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        shuffle=True,
        batch_size=cfg.PER_DEVICE_TRAIN_BATCH_SIZE
    )
    eval_dataloader = None
    if "validation" in tokenized_datasets and len(tokenized_datasets["validation"]) > 0:
        eval_dataloader = DataLoader(
            tokenized_datasets["validation"],
            batch_size=cfg.PER_DEVICE_EVAL_BATCH_SIZE
        )
    else:
        logger.warning("No validation set found or it's empty. Skipping evaluation during training.")


    # Optimizer and Scheduler
    try:
        optimizer = bnb.optim.PagedAdamW8bit(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.LEARNING_RATE
        )
        logger.info("Using PagedAdamW8bit optimizer.")
    except ImportError:
        logger.warning("PagedAdamW8bit not available. Falling back to torch.optim.AdamW.")
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.LEARNING_RATE
        )

    num_training_steps_per_epoch = len(train_dataloader) // cfg.GRADIENT_ACCUMULATION_STEPS
    if len(train_dataloader) % cfg.GRADIENT_ACCUMULATION_STEPS != 0:
        num_training_steps_per_epoch +=1
    
    total_training_steps = cfg.NUM_TRAIN_EPOCHS * num_training_steps_per_epoch
    num_warmup_steps = int(cfg.WARMUP_RATIO * total_training_steps)

    lr_scheduler = get_scheduler(
        name="cosine", # "linear" or "cosine" are common
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_training_steps,
    )

    # Mixed Precision Scaler
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    logger.info(f"Total optimizer steps: {total_training_steps}")
    logger.info(f"Warmup steps: {num_warmup_steps}")

    # Evaluation frequency
    eval_every_n_steps = 0
    if eval_dataloader and cfg.EVAL_EVERY_N_OPTIMIZER_STEPS_FACTOR > 0:
        steps_per_epoch_approx = len(train_dataloader) // cfg.GRADIENT_ACCUMULATION_STEPS
        eval_every_n_steps = max(1, steps_per_epoch_approx // cfg.EVAL_EVERY_N_OPTIMIZER_STEPS_FACTOR)
        logger.info(f"Will evaluate every {eval_every_n_steps} optimizer steps.")
    elif eval_dataloader: # Evaluate at end of epoch if factor is 0 or less
        eval_every_n_steps = total_training_steps # Effectively end of training, or handle per epoch
        logger.info("Will evaluate at the end of each epoch.")


    training_metrics = TrainingMetrics()
    global_step = 0
    actual_model_steps = 0 # Forward/backward passes before optimizer step

    logger.info("--- Starting Custom Training Loop ---")
    for epoch in range(cfg.NUM_TRAIN_EPOCHS):
        model.train()
        epoch_train_loss = 0
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch+1}/{cfg.NUM_TRAIN_EPOCHS} Training",
            disable=not logger.isEnabledFor(logging.INFO) # Disable tqdm if logger level is higher than INFO
        )
        
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(progress_bar):
            batch = {k: v.to(main_data_device) for k, v in batch.items()}
            
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(**batch)
                loss = outputs.loss
            
            if cfg.GRADIENT_ACCUMULATION_STEPS > 1:
                loss = loss / cfg.GRADIENT_ACCUMULATION_STEPS
            
            scaler.scale(loss).backward()
            epoch_train_loss += loss.item() * cfg.GRADIENT_ACCUMULATION_STEPS # Unscale for logging

            actual_model_steps += 1
            
            if actual_model_steps % cfg.GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()), 
                    cfg.MAX_GRAD_NORM
                )
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                current_lr = lr_scheduler.get_last_lr()[0] if lr_scheduler else cfg.LEARNING_RATE
                progress_bar.set_postfix({
                    'loss': loss.item() * cfg.GRADIENT_ACCUMULATION_STEPS,
                    'lr': f"{current_lr:.2e}",
                    'opt_step': global_step
                })

                if eval_dataloader and eval_every_n_steps > 0 and \
                   (global_step % eval_every_n_steps == 0 or global_step == total_training_steps) :
                    
                    eval_metrics = evaluate_model(model, eval_dataloader, tokenizer, device, main_data_device)
                    log_entry = {
                        'epoch': epoch + 1,
                        'optimizer_step': global_step,
                        'train_loss_batch': loss.item() * cfg.GRADIENT_ACCUMULATION_STEPS,
                        **eval_metrics
                    }
                    if training_metrics.update(log_entry):
                        logger.info(f"New best model found! ROUGE-L: {training_metrics.best_metric_value:.4f}. Saving adapter...")
                        save_path = os.path.join(cfg.OUTPUT_DIR, "best_model_adapter")
                        cfg.ensure_dir_exists(save_path)
                        model.save_pretrained(save_path)
                        tokenizer.save_pretrained(save_path)
                        logger.info(f"Best model adapter saved to {save_path}")
                    
                    training_metrics.save_history(os.path.join(cfg.OUTPUT_DIR, "metrics_history.json"))
                    model.train() # Set back to training mode

        avg_epoch_train_loss = epoch_train_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch+1} finished. Average Training Loss: {avg_epoch_train_loss:.4f}")
        
        # End of epoch evaluation if not done by step-based evaluation
        if eval_dataloader and (eval_every_n_steps <= 0 or (cfg.EVAL_EVERY_N_OPTIMIZER_STEPS_FACTOR <= 0 and global_step < total_training_steps)):
            eval_metrics = evaluate_model(model, eval_dataloader, tokenizer, device, main_data_device)
            log_entry = {
                'epoch': epoch + 1, 'optimizer_step': global_step,
                'train_loss_epoch_avg': avg_epoch_train_loss, **eval_metrics
            }
            if training_metrics.update(log_entry):
                logger.info(f"New best model (end of epoch)! ROUGE-L: {training_metrics.best_metric_value:.4f}. Saving adapter...")
                save_path = os.path.join(cfg.OUTPUT_DIR, "best_model_adapter")
                cfg.ensure_dir_exists(save_path)
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                logger.info(f"Best model adapter saved to {save_path}")
            training_metrics.save_history(os.path.join(cfg.OUTPUT_DIR, "metrics_history.json"))
            model.train()

    logger.info("--- Training Finished ---")
    final_save_path = os.path.join(cfg.OUTPUT_DIR, cfg.NEW_ADAPTER_NAME) # Use NEW_ADAPTER_NAME
    cfg.ensure_dir_exists(final_save_path)
    model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    logger.info(f"Final model adapter saved to {final_save_path}")
    training_metrics.save_history(os.path.join(cfg.OUTPUT_DIR, "metrics_history.json"))
    
    return training_metrics.history