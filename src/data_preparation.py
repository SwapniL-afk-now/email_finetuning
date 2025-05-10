import os
import re
import logging
from datasets import load_dataset, Dataset
from configs import finetune_config as cfg # Using alias for brevity

logger = logging.getLogger(__name__)

REPLY_FWD_PREFIX_PATTERN = re.compile(r"^(?:(?:\{RE:\}|RE|FWD|FW)[:\s]*)+", re.IGNORECASE)

def clean_subject(subject_text):
    """Removes common reply/forward prefixes from an email subject."""
    if not subject_text:
        return subject_text
    cleaned_subject = REPLY_FWD_PREFIX_PATTERN.sub("", subject_text)
    cleaned_subject = cleaned_subject.replace("{RE:}", "") # Safeguard
    return cleaned_subject.strip()

def prepare_dataset_subset():
    """
    Loads the original dataset, processes it to extract and clean
    email bodies and subjects, selects a subset, and saves it to disk.
    Only runs if the processed subset doesn't already exist.
    """
    processed_dataset_path = cfg.get_processed_dataset_path()
    cfg.ensure_dir_exists(processed_dataset_path) # Ensure base directory exists

    if os.path.exists(processed_dataset_path):
        logger.info(f"Processed dataset already found at {processed_dataset_path}. Skipping preparation.")
        return

    logger.info(f"Subset dataset not found at {processed_dataset_path}. Preparing dataset...")
    try:
        original_dataset = load_dataset(cfg.ORIGINAL_DATASET_ID, split=cfg.ORIGINAL_DATASET_SPLIT)
        logger.info(f"Successfully loaded {len(original_dataset)} samples from {cfg.ORIGINAL_DATASET_ID}.")
    except Exception as e:
        logger.error(f"Error loading original dataset: {e}")
        raise

    processed_data = []
    stats = {"empty_email_lists": 0, "missing_keys": 0, "not_a_list": 0, "subjects_cleaned": 0}

    logger.info("Processing samples to extract and clean first email body/subject...")
    for i, sample in enumerate(original_dataset):
        if (i + 1) % 50000 == 0:
            logger.info(f"  Processed {i+1}/{len(original_dataset)} original samples...")

        email_list = sample.get('formatted_emails')
        if not isinstance(email_list, list) or not email_list:
            stats["empty_email_lists"] += 1 if not email_list else 0
            stats["not_a_list"] += 1 if not isinstance(email_list, list) else 0
            continue

        first_email = email_list[0]
        if isinstance(first_email, dict):
            body = first_email.get('body')
            original_subject = first_email.get('subject')

            if body is not None and original_subject is not None:
                cleaned_subject = clean_subject(original_subject)
                if original_subject != cleaned_subject:
                    stats["subjects_cleaned"] += 1
                processed_data.append({'body': body, 'subject': cleaned_subject})
            else:
                stats["missing_keys"] += 1
        else:
            stats["missing_keys"] += 1

    logger.info(f"Finished processing {len(original_dataset)} original samples.")
    logger.info(f"  Successfully extracted {len(processed_data)} email body/subject pairs.")
    for key, value in stats.items():
        if value > 0: logger.info(f"  {key.replace('_', ' ').capitalize()}: {value}")

    if not processed_data:
        logger.warning("No data was processed, intermediate dataset is empty. Not creating subset.")
        return

    new_dataset = Dataset.from_list(processed_data)
    logger.info(f"Intermediate dataset created with {len(new_dataset)} entries.")

    actual_num_to_take = min(cfg.NUM_SAMPLES_TO_KEEP, len(new_dataset))
    if actual_num_to_take == 0:
        logger.warning("Intermediate dataset is empty after filtering, cannot create a subset.")
        return

    if actual_num_to_take < cfg.NUM_SAMPLES_TO_KEEP:
        logger.warning(f"Intermediate dataset has only {len(new_dataset)} samples. Taking all {actual_num_to_take}.")

    logger.info(f"Selecting the first {actual_num_to_take} samples for the finetuning dataset...")
    finetuning_subset = new_dataset.select(range(actual_num_to_take))
    logger.info(f"Created finetuning subset with {len(finetuning_subset)} samples.")

    logger.info(f"Saving the finetuning subset to disk at: {processed_dataset_path}")
    try:
        finetuning_subset.save_to_disk(processed_dataset_path)
        logger.info("Finetuning subset saved successfully.")
    except Exception as e:
        logger.error(f"Error saving dataset to disk: {e}")
        raise

def inspect_subset_dataset():
    """Loads and prints a few samples from the processed subset for inspection."""
    processed_dataset_path = cfg.get_processed_dataset_path()
    if not os.path.exists(processed_dataset_path):
        logger.warning(f"Subset not found at {processed_dataset_path}, cannot inspect.")
        return

    logger.info(f"Loading the processed subset from {processed_dataset_path} for inspection...")
    try:
        loaded_subset = Dataset.load_from_disk(processed_dataset_path)
        logger.info(f"Loaded subset with {len(loaded_subset)} samples.")
        if len(loaded_subset) > 0:
            logger.info(f"First {min(3, len(loaded_subset))} sample subjects from the subset:")
            for i in range(min(3, len(loaded_subset))):
                logger.info(f"  Sample {i} Subject: '{loaded_subset[i]['subject']}'")
    except Exception as e:
        logger.error(f"Error loading subset for inspection: {e}")