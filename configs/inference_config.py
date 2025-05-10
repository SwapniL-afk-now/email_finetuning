import torch
import os

# --- Input/Output Files ---
TEST_DATA_PATH = "./sample_test_emails.csv"  # Path to your input CSV test file
EMAIL_BODY_COLUMN_NAME = "email_body"      # Column name in CSV containing email bodies
ID_COLUMN_NAME = "email_id"                # Optional: Column for unique IDs in your CSV
TEST_OUTPUT_FILE_PATH = "./test_output_subjects_generated.csv"

# --- Model and Tokenizer ---
# This can be a Hugging Face Hub ID or a local path to the fine-tuned adapter
FINETUNED_MODEL_ADAPTER_ID = "exper1ment/email_swapnil" # Replace with your Hub ID or local path
# If FINETUNED_MODEL_ADAPTER_ID is a local path, BASE_MODEL_ID might be needed if tokenizer/config is not with adapter
# BASE_MODEL_ID_FOR_TOKENIZER = "Qwen/Qwen1.5-4B" # Only if tokenizer not with adapter

# --- Generation Parameters ---
MAX_INPUT_LENGTH = 462 # MAX_LENGTH (512) - MAX_NEW_TOKENS (50) = 462. Input prompt length.
MAX_NEW_TOKENS = 50     # Max tokens to generate for the subject
GENERATION_SYSTEM_PROMPT = "You are an expert email assistant. Your task is to generate a concise and relevant subject line for the given email body."


# --- BitsAndBytes Configuration (if loading a QLoRA model) ---
BNB_LOAD_IN_4BIT = True
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_COMPUTE_DTYPE = torch.bfloat16
BNB_4BIT_USE_DOUBLE_QUANT = True

# --- Device Configuration ---
# "auto" will try to use GPU if available
INFERENCE_DEVICE = "auto" # or "cuda", "cpu"