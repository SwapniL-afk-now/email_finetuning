import logging
from datasets import load_from_disk, DatasetDict, Dataset
from transformers import AutoTokenizer
from configs import finetune_config as cfg
import os

logger = logging.getLogger(__name__)

def load_and_split_dataset(dataset_path, seed):
    """Loads dataset from disk and splits into train/validation sets."""
    if not os.path.exists(dataset_path):
        logger.warning(f"Dataset not found at {dataset_path}. Using dummy data.")
        dummy_data = [
            {'body': "Hey team, let's discuss the Q3 budget next week. Please come prepared.", 'subject': "Q3 Budget Meeting"},
            {'body': "Hi Alice, I've finished the report you asked for. It's attached.", 'subject': "Report Submission"}
        ] * (cfg.NUM_SAMPLES_TO_KEEP // 2 if cfg.NUM_SAMPLES_TO_KEEP > 2 else 500) # Ensure enough dummy data
        dataset = Dataset.from_list(dummy_data)
    else:
        logger.info(f"Loading dataset from {dataset_path}...")
        dataset = load_from_disk(dataset_path)

    logger.info(f"Loaded dataset with {len(dataset)} samples.")

    val_size = max(1, int(len(dataset) * cfg.VALIDATION_SET_SIZE))
    if len(dataset) - val_size <=0: # Ensure train set is not empty
        logger.warning("Dataset too small for specified validation split. Adjusting validation size.")
        val_size = 1 if len(dataset) > 1 else 0
    
    if val_size == 0 and len(dataset) > 0: # If val_size became 0 but dataset is not empty
        logger.warning("Validation set size is 0. Training will proceed without validation.")
        raw_datasets = DatasetDict({'train': dataset})
    elif val_size > 0:
        train_test_split = dataset.train_test_split(test_size=val_size, seed=seed)
        raw_datasets = DatasetDict({
            'train': train_test_split['train'],
            'validation': train_test_split['test']
        })
    else: # Dataset is empty
        logger.error("Cannot create dataset splits from an empty dataset.")
        raise ValueError("Empty dataset loaded.")


    logger.info(f"Train dataset size: {len(raw_datasets['train'])}")
    if 'validation' in raw_datasets:
        logger.info(f"Validation dataset size: {len(raw_datasets['validation'])}")
        logger.info(f"Sample from validation dataset: {raw_datasets['validation'][0]}")
    logger.info(f"Sample from train dataset: {raw_datasets['train'][0]}")
    return raw_datasets

def get_tokenizer(model_id):
    """Loads tokenizer, setting pad_token if necessary."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set tokenizer.pad_token to tokenizer.eos_token ({tokenizer.pad_token_id})")
    tokenizer.padding_side = "right" # Important for training, "left" for generation
    return tokenizer

def create_full_prompt_for_training(sample_dict):
    """Creates the full prompt string including the target subject for training."""
    return (
        f"<|im_start|>system\n{cfg.SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\nEmail Body:\n{sample_dict['body']}\n\n"
        f"Subject:<|im_end|>\n<|im_start|>assistant\n{sample_dict['subject']}<|im_end|>"
    )

def create_prompt_for_label_masking(body_text):
    """Creates the prompt string *without* the target subject, used for label masking."""
    return (
        f"<|im_start|>system\n{cfg.SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\nEmail Body:\n{body_text}\n\n"
        f"Subject:<|im_end|>\n<|im_start|>assistant\n"
    )

def preprocess_data_for_training(examples, tokenizer):
    """
    Tokenizes examples and creates labels for language modeling.
    Labels are masked for the prompt part.
    """
    num_samples_in_batch = len(examples['body'])
    
    prompts_with_completion = []
    for i in range(num_samples_in_batch):
        current_sample_as_dict = {'body': examples['body'][i], 'subject': examples['subject'][i]}
        prompts_with_completion.append(create_full_prompt_for_training(current_sample_as_dict))

    model_inputs = tokenizer(
        prompts_with_completion,
        max_length=cfg.MAX_LENGTH,
        truncation=True,
        padding="max_length"
    )
    
    labels_list_of_lists = [seq.copy() for seq in model_inputs["input_ids"]]

    for i in range(num_samples_in_batch):
        prompt_only_text = create_prompt_for_label_masking(examples['body'][i])
        # Important: Do not add special tokens again for prompt_only if they are already part of the template
        prompt_only_tokens = tokenizer(
            prompt_only_text,
            max_length=cfg.MAX_LENGTH, # Should not truncate here ideally
            truncation=True, # Safety for very long bodies, but usually prompt_only is shorter
            add_special_tokens=False # Qwen tokenizer might add bos implicitly, check this.
                                      # If create_prompt templates include all special tokens, this should be False.
        )["input_ids"]
        prompt_len = len(prompt_only_tokens)

        for j in range(len(labels_list_of_lists[i])):
            if j < prompt_len or labels_list_of_lists[i][j] == tokenizer.pad_token_id:
                labels_list_of_lists[i][j] = -100 # Mask prompt tokens and padding tokens

    model_inputs["labels"] = labels_list_of_lists
    return model_inputs


def tokenize_datasets(raw_datasets, tokenizer):
    """Applies preprocessing and tokenization to the datasets."""
    logger.info("Preprocessing and tokenizing datasets...")
    # num_proc can be increased for faster processing on multi-core CPUs
    tokenized_datasets = raw_datasets.map(
        lambda examples: preprocess_data_for_training(examples, tokenizer),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        num_proc= os.cpu_count() // 2 if os.cpu_count() > 1 else 1
    )
    logger.info("Dataset tokenization complete.")
    logger.info(f"Sample tokenized input_ids (first 30): {tokenized_datasets['train'][0]['input_ids'][:30]}")
    logger.info(f"Sample tokenized labels (first 30): {tokenized_datasets['train'][0]['labels'][:30]}")
    return tokenized_datasets