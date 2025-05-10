
## Prerequisites

*   Python 3.8+
*   pip (Python package installer)
*   NVIDIA GPU with CUDA support (highly recommended for reasonable training times).
    *   Ensure CUDA toolkit and NVIDIA drivers are correctly installed.
*   Git (for cloning the repository)

## Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd email_llm_finetuning
    ```

2.  **Create and Activate a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Linux/macOS
    source venv/bin/activate
    # On Windows
    # venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```


    ```

4.  **Hugging Face Login (Optional, but recommended):**
    If you plan to use private models/datasets or push your fine-tuned model to the Hugging Face Hub:
    ```bash
    huggingface-cli login
    ```
    You might also need to set the `HF_TOKEN` environment variable for certain operations within scripts.

## Configuration

All major settings are managed through Python configuration files in the `configs/` directory.

### Fine-tuning Configuration (`configs/finetune_config.py`)

This file controls all aspects of the data preparation and model fine-tuning process. Key parameters include:

*   `MODEL_ID`: The base Hugging Face model ID (e.g., `"Qwen/Qwen1.5-4B"`).
*   `OUTPUT_DIR`: Directory to save fine-tuned model adapters, logs, and plots.
*   `NEW_ADAPTER_NAME`: Name for the final saved adapter.
*   `SEED`: Random seed for reproducibility.
*   `ORIGINAL_DATASET_ID`: Hugging Face ID of the dataset to process (e.g., `"argilla/FinePersonas-Synthetic-Email-Conversations"`).
*   `NUM_SAMPLES_TO_KEEP`: Number of samples to use from the original dataset for fine-tuning.
*   `SUBSET_SAVE_PATH_BASE`: Base path for saving the processed subset of data.
*   `SYSTEM_PROMPT`: The system instruction given to the LLM.
*   `MAX_LENGTH`: Maximum sequence length for tokenization.
*   `BNB_*` settings: For BitsAndBytes quantization (QLoRA).
*   `LORA_*` settings: For LoRA configuration (rank, alpha, target modules).
*   `LEARNING_RATE`, `NUM_TRAIN_EPOCHS`, `PER_DEVICE_TRAIN_BATCH_SIZE`, etc.: Standard training hyperparameters.
*   `EVAL_EVERY_N_OPTIMIZER_STEPS_FACTOR`: How frequently to run evaluation during training (e.g., `2` means twice per epoch).

### Inference Configuration (`configs/inference_config.py`)

This file controls the settings for generating subjects using a fine-tuned model. Key parameters include:

*   `TEST_DATA_PATH`: Path to your input CSV file containing email bodies for inference.
*   `EMAIL_BODY_COLUMN_NAME`: The name of the column in your CSV that holds the email body text.
*   `ID_COLUMN_NAME` (Optional): Name of a column containing unique identifiers for emails.
*   `TEST_OUTPUT_FILE_PATH`: Path where the CSV with generated subjects will be saved.
*   `FINETUNED_MODEL_ADAPTER_ID`: Hugging Face Hub ID or local path to your fine-tuned model adapter (e.g., `"your-username/your-finetuned-model-adapter"` or `./qwen1.5-4b-email-subject-finetuned/best_model_adapter`).
*   `MAX_INPUT_LENGTH`: Maximum length of the input prompt fed to the model.
*   `MAX_NEW_TOKENS`: Maximum number of tokens to generate for the subject line.
*   `GENERATION_SYSTEM_PROMPT`: System prompt for inference (should be consistent with training).
*   `BNB_*` settings: Quantization settings if loading a QLoRA model.

**Important:** Before running, review and adjust these configuration files to match your environment, model choices, and data paths.

## Usage

### 1. Data Preparation (Automatic)

The data preparation step (downloading, cleaning, and subsetting the dataset) is automatically triggered by the `main_finetune.py` script if the processed dataset (defined by `PROCESSED_DATASET_PATH` in `configs/finetune_config.py`) is not found.

*   **Source Dataset:** By default, it uses `"argilla/FinePersonas-Synthetic-Email-Conversations"`.
*   **Output:** A processed subset of this dataset is saved locally (e.g., `./finetuning_email_subset_20k/`).

You generally don't need to run a separate script for this unless you want to customize the `src/data_preparation.py` module significantly.

### 2. Fine-tuning the Model

This script handles the entire fine-tuning pipeline: loading data, tokenizing, setting up the QLoRA model, and running the custom training loop.

**Steps:**

1.  **Verify Configuration:** Ensure `configs/finetune_config.py` is set up correctly, especially `MODEL_ID`, `OUTPUT_DIR`, and dataset parameters.
2.  **Run Fine-tuning:**
    ```bash
    python main_finetune.py
    ```

**What happens:**

*   The script will first check for and prepare the dataset if needed.
*   It will load the base model and tokenizer.
*   The model will be configured with QLoRA.
*   The custom training loop will start, showing progress.
*   Evaluation will be performed periodically (if a validation set exists and is configured).
*   The best-performing model adapter (based on ROUGE-L on the validation set) will be saved to `<OUTPUT_DIR>/best_model_adapter/`.
*   The final model adapter will be saved to `<OUTPUT_DIR>/<NEW_ADAPTER_NAME>/`.
*   Training metrics (JSON) and plots (PNG) will be saved in `OUTPUT_DIR`.

### 3. Generating Subjects (Inference)

Once you have a fine-tuned model adapter, you can use it to generate subject lines for new email bodies.

**Steps:**

1.  **Prepare Test Data:**
    Create a CSV file (e.g., `my_emails.csv`) with at least one column containing the email bodies.
    Example CSV (`my_emails.csv`):
    ```csv
    email_id,email_body
    msg1,"Hey team, Just a reminder about our weekly sync-up call tomorrow at 10 AM. Please find the agenda attached. Looking forward to discussing the project updates."
    msg2,"Hi Sarah, I've reviewed the document you sent over. I have a few minor suggestions. Could we chat briefly about them sometime this afternoon?"
    ```

2.  **Verify Configuration:**
    Adjust `configs/inference_config.py`:
    *   Set `TEST_DATA_PATH` to the path of your CSV file (e.g., `"./my_emails.csv"`).
    *   Set `EMAIL_BODY_COLUMN_NAME` to the name of your email body column (e.g., `"email_body"`).
    *   Set `ID_COLUMN_NAME` (if you have one, e.g., `"email_id"`).
    *   Set `FINETUNED_MODEL_ADAPTER_ID` to your fine-tuned adapter's Hugging Face Hub ID or local path (e.g., `"your-username/qwen1.5-4b-email-adapter"` or `"./qwen1.5-4b-email-subject-finetuned/best_model_adapter"`).
    *   Set `TEST_OUTPUT_FILE_PATH` to where you want the results saved.

3.  **Run Inference:**
    ```bash
    python main_inference.py
    ```

**What happens:**

*   The script loads your fine-tuned model and tokenizer.
*   It reads each email body from your input CSV.
*   For each email body, it generates a subject line.
*   The results (original body, generated subject, and ID if provided) are saved to the `TEST_OUTPUT_FILE_PATH` CSV.

## Outputs

*   **Fine-tuning:**
    *   Fine-tuned model adapters (PEFT format) saved in subdirectories within `OUTPUT_DIR` (e.g., `best_model_adapter/`, `qwen1.5-4b-email-subject-adapter/`).
    *   `metrics_history.json`: A JSON file logging training and evaluation metrics per evaluation step.
    *   `training_metrics_plot.png`: A plot visualizing training/validation loss and ROUGE scores.
    *   Processed dataset (if generated) at `PROCESSED_DATASET_PATH`.
*   **Inference:**
    *   A CSV file (specified by `TEST_OUTPUT_FILE_PATH`) containing the original email bodies and their corresponding generated subject lines.

## Troubleshooting & Notes

*   **CUDA Out of Memory:**
    *   Reduce `PER_DEVICE_TRAIN_BATCH_SIZE` / `PER_DEVICE_EVAL_BATCH_SIZE`.
    *   Increase `GRADIENT_ACCUMULATION_STEPS`.
    *   Ensure no other processes are consuming GPU memory.
    *   Try a smaller base model if memory is severely constrained.
    *   For Qwen models, ensure `model.config.pretraining_tp = 1` is set if you see related warnings/errors.
*   **Slow Training:**
    *   Ensure you are using a GPU and PyTorch recognizes it (`torch.cuda.is_available()` should be `True`).
    *   Check GPU utilization (e.g., using `nvidia-smi`).
*   **`bitsandbytes` errors:**
    *   Ensure you have the correct version of `bitsandbytes` compatible with your CUDA version. Sometimes, specific installation instructions are needed for `bitsandbytes` depending on the OS and CUDA setup.
    *   Refer to the `bitsandbytes` GitHub repository for troubleshooting.
*   **Model Access (Private Models/Datasets):**
    *   If `MODEL_ID` or `ORIGINAL_DATASET_ID` points to a private resource on Hugging Face Hub, ensure you are logged in (`huggingface-cli login`) or have set the `HF_TOKEN` environment variable.
*   **Path Issues:**
    *   Double-check all file and directory paths in the configuration files. Paths are relative to the project root where you run the scripts.
*   **Tokenizer Behavior with Qwen ChatML format:**
    *   The Qwen tokenizer and prompt format (`<|im_start|>...<|im_end|>`) are specific. Ensure the `preprocess_data_for_training` function correctly masks only the prompt part and not the special tokens within the prompt structure if they are essential for the model's understanding. The current `add_special_tokens=False` in `create_prompt_for_label_masking` assumes the template itself adds all necessary special tokens.



