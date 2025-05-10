import torch
import pandas as pd
from tqdm.auto import tqdm
import os
import logging

from configs import inference_config as cfg
from src.model_utils import load_model_for_inference, get_inference_tokenizer
from src.utils import get_main_device, ensure_dir_exists

logger = logging.getLogger(__name__)

def create_inference_prompt(body_text):
    """Creates the prompt string for the model based on the email body for inference."""
    # Ensure this matches the format used during fine-tuning (user part)
    return (
        f"<|im_start|>system\n{cfg.GENERATION_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\nEmail Body:\n{body_text}\n\n"
        f"Subject:<|im_end|>\n<|im_start|>assistant\n"  # Model generates after this
    )

def generate_subjects_for_csv(model, tokenizer, main_data_device):
    """Loads data from CSV, generates subjects, and returns a DataFrame with results."""
    try:
        test_df = pd.read_csv(cfg.TEST_DATA_PATH)
        if cfg.EMAIL_BODY_COLUMN_NAME not in test_df.columns:
            logger.error(f"Email body column '{cfg.EMAIL_BODY_COLUMN_NAME}' not found in '{cfg.TEST_DATA_PATH}'.")
            logger.error(f"Available columns: {test_df.columns.tolist()}")
            raise ValueError(f"Column '{cfg.EMAIL_BODY_COLUMN_NAME}' not found.")
        logger.info(f"Successfully loaded test data from '{cfg.TEST_DATA_PATH}'. Emails: {len(test_df)}")
    except FileNotFoundError:
        logger.error(f"Test data file not found: '{cfg.TEST_DATA_PATH}'")
        raise
    except Exception as e:
        logger.error(f"Error loading test data CSV: {e}")
        raise

    results_list = []
    logger.info(f"Generating subjects for {len(test_df)} email bodies...")

    # Prepare EOS token IDs
    im_end_token_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
    eos_token_ids = []
    if isinstance(tokenizer.eos_token_id, int): eos_token_ids.append(tokenizer.eos_token_id)
    elif isinstance(tokenizer.eos_token_id, list): eos_token_ids.extend(tokenizer.eos_token_id)
    eos_token_ids.append(im_end_token_id)
    eos_token_ids = sorted(list(set(eos_token_ids))) # Unique and sorted

    for index, row in tqdm(test_df.iterrows(), total=test_df.shape[0], desc="Processing emails"):
        email_body = str(row[cfg.EMAIL_BODY_COLUMN_NAME])
        email_id = f"email_{index}"
        if cfg.ID_COLUMN_NAME and cfg.ID_COLUMN_NAME in row and pd.notna(row[cfg.ID_COLUMN_NAME]):
            email_id = str(row[cfg.ID_COLUMN_NAME])

        generated_subject = "ERROR: SKIPPED"
        if not email_body.strip():
            logger.warning(f"Skipping email ID '{email_id}' (index {index}) due to empty body.")
            generated_subject = "ERROR: SKIPPED - EMPTY BODY"
        else:
            prompt = create_inference_prompt(email_body)
            input_ids = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=cfg.MAX_INPUT_LENGTH # Max length for prompt
            ).input_ids.to(main_data_device)

            try:
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=input_ids,
                        max_new_tokens=cfg.MAX_NEW_TOKENS,
                        eos_token_id=eos_token_ids,
                        pad_token_id=tokenizer.pad_token_id,
                        do_sample=True,
                        top_p=0.9,
                        temperature=0.7,
                        num_return_sequences=1,
                    )
                
                # Decode only newly generated tokens
                generated_ids = outputs[0][input_ids.shape[1]:]
                generated_subject = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                
                # Basic post-processing
                if generated_subject.endswith(("...", "..", ".", ",")):
                    generated_subject = generated_subject[:-1] # Remove single trailing char
                if "Subject:" in generated_subject: # If model hallucinates "Subject:"
                    generated_subject = generated_subject.split("Subject:", 1)[-1].strip()
                if "<|im_end|>" in generated_subject: # Remove im_end if it appears
                    generated_subject = generated_subject.split("<|im_end|>")[0].strip()


            except Exception as e:
                logger.error(f"Error generating subject for email ID '{email_id}' (index {index}): {e}")
                generated_subject = f"ERROR: GENERATION FAILED - {type(e).__name__}"
        
        results_list.append({
            "id": email_id,
            "original_body": email_body,
            "generated_subject": generated_subject
        })

    return pd.DataFrame(results_list)

def run_inference():
    """Main function to orchestrate the inference process."""
    logger.info("--- Starting Subject Generation from CSV ---")
    logger.info(f"Using model adapter: {cfg.FINETUNED_MODEL_ADAPTER_ID}")
    logger.info(f"Input CSV: {cfg.TEST_DATA_PATH} (Body column: '{cfg.EMAIL_BODY_COLUMN_NAME}')")
    
    torch.cuda.empty_cache()

    try:
        model = load_model_for_inference(cfg.FINETUNED_MODEL_ADAPTER_ID)
        tokenizer = get_inference_tokenizer(cfg.FINETUNED_MODEL_ADAPTER_ID) # Load tokenizer from adapter path
    except Exception as e:
        logger.fatal(f"Fatal error loading model or tokenizer: {e}. Exiting.")
        return

    main_data_device = get_main_device(model)
    logger.info(f"Inference device for data: {main_data_device}")

    results_df = generate_subjects_for_csv(model, tokenizer, main_data_device)

    try:
        ensure_dir_exists(cfg.TEST_OUTPUT_FILE_PATH)
        results_df.to_csv(cfg.TEST_OUTPUT_FILE_PATH, index=False, encoding='utf-8')
        logger.info(f"Test output successfully saved to: {cfg.TEST_OUTPUT_FILE_PATH}")
        logger.info("First 5 generated results:")
        logger.info("\n" + results_df.head().to_string())
    except Exception as e:
        logger.error(f"Error saving output CSV: {e}")

    logger.info("--- CSV Processing Finished ---")