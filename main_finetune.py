import logging
import os

from configs import finetune_config as cfg
from src.utils import setup_logging, set_seed,  ensure_dir_exists
from src.data_preparation import prepare_dataset_subset, inspect_subset_dataset
from src.dataset_utils import load_and_split_dataset, get_tokenizer, tokenize_datasets
from src.model_utils import load_model_for_finetuning
from src.training_loop import run_training_loop
from src.plotting import plot_training_metrics

def main():
    setup_logging() # Basic logging setup
    logger = logging.getLogger(__name__)

    logger.info("--- Starting Fine-tuning Process ---")

    # Setup
    set_seed(cfg.SEED)

    ensure_dir_exists(cfg.OUTPUT_DIR + "/") # Ensure output dir exists (add / for os.path.dirname)
    ensure_dir_exists(cfg.get_processed_dataset_path()) # Ensure dataset dir exists

    # 1. Prepare Dataset (downloads, cleans, subsets if not already done)
    logger.info("Step 1: Preparing dataset subset...")
    prepare_dataset_subset()
    inspect_subset_dataset() # Optional: inspect a few samples

    # 2. Load and Split Processed Dataset
    logger.info("Step 2: Loading and splitting processed dataset...")
    raw_datasets = load_and_split_dataset(cfg.PROCESSED_DATASET_PATH, cfg.SEED)

    # 3. Load Tokenizer
    logger.info("Step 3: Loading tokenizer...")
    tokenizer = get_tokenizer(cfg.MODEL_ID)

    # 4. Tokenize Datasets
    logger.info("Step 4: Tokenizing datasets...")
    tokenized_datasets = tokenize_datasets(raw_datasets, tokenizer)

    # 5. Load Model for Fine-tuning (with QLoRA)
    logger.info("Step 5: Loading model for fine-tuning...")
    model = load_model_for_finetuning(cfg.MODEL_ID)

    # 6. Run Custom Training Loop
    logger.info("Step 6: Starting training loop...")
    metrics_history = run_training_loop(model, tokenizer, tokenized_datasets)

    # 7. Plot Metrics (if history available)
    if metrics_history:
        logger.info("Step 7: Plotting training metrics...")
        plot_training_metrics(metrics_history, cfg.OUTPUT_DIR)
    else:
        logger.info("Step 7: No metrics history to plot.")

    logger.info("--- Fine-tuning Process Completed ---")

if __name__ == "__main__":
    main()