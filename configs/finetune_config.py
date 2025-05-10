import torch
import os

# --- General Settings ---
MODEL_ID = "Qwen/Qwen1.5-4B"  # Base model ID from Hugging Face Hub
OUTPUT_DIR = "./qwen1.5-4b-email-subject-finetuned"
NEW_ADAPTER_NAME = "qwen1.5-4b-email-subject-adapter" # Name for the saved adapter
SEED = 42

# --- Dataset Preparation ---
# Path to the original dataset for preprocessing (if needed)
ORIGINAL_DATASET_ID = "argilla/FinePersonas-Synthetic-Email-Conversations"
ORIGINAL_DATASET_SPLIT = "train"
NUM_SAMPLES_TO_KEEP = 20000  # Number of samples to subset from the original dataset
SUBSET_SAVE_PATH_BASE = "./finetuning_email_subset"

# Dynamically construct the full path for the processed dataset
def get_processed_dataset_path():
    return f"{SUBSET_SAVE_PATH_BASE}_{NUM_SAMPLES_TO_KEEP // 1000}k"

PROCESSED_DATASET_PATH = get_processed_dataset_path() # This will be used to load data for training

# --- Tokenization & Prompting ---
# System prompt for instructing the model
SYSTEM_PROMPT = "You are an expert email assistant. Your task is to generate a concise and relevant subject line for the given email body."
MAX_LENGTH = 512  # Max sequence length for tokenizer (prompt + completion)
VALIDATION_SET_SIZE = 0.01 # 1% for validation, or at least 1 sample

# --- Model Quantization (QLoRA) ---
BNB_LOAD_IN_4BIT = True
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_COMPUTE_DTYPE = torch.bfloat16 # or torch.float16 if bfloat16 not supported
BNB_4BIT_USE_DOUBLE_QUANT = True

# --- LoRA Configuration ---
LORA_R = 16
LORA_ALPHA = 32
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]
LORA_DROPOUT = 0.05
LORA_BIAS = "none"
PEFT_TASK_TYPE = "CAUSAL_LM"

# --- Custom Training Loop Hyperparameters ---
LEARNING_RATE = 5e-4
NUM_TRAIN_EPOCHS = 1
PER_DEVICE_TRAIN_BATCH_SIZE = 4
PER_DEVICE_EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
WARMUP_RATIO = 0.03 # Percentage of total training steps for warmup
MAX_GRAD_NORM = 1.0 # For gradient clipping

# How often to evaluate (in terms of optimizer steps)
# If set to 0 or None, evaluates only at the end of epochs.
# Set to a positive integer to evaluate every N optimizer steps.
# Default: evaluate twice per epoch (approx).
EVAL_EVERY_N_OPTIMIZER_STEPS_FACTOR = 2 # e.g. 2 means twice per epoch

# --- Directory Setup ---
# Ensure output directory exists (done in main script typically)
# os.makedirs(OUTPUT_DIR, exist_ok=True)
# os.makedirs(os.path.dirname(PROCESSED_DATASET_PATH) or ".", exist_ok=True)