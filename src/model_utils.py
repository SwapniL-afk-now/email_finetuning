import torch
import logging
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from configs import finetune_config as cfg
from src.utils import get_device_map

logger = logging.getLogger(__name__)

def load_model_for_finetuning(model_id_or_path):
    """Loads the base model with QLoRA configuration for fine-tuning."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.BNB_LOAD_IN_4BIT,
        bnb_4bit_quant_type=cfg.BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=cfg.BNB_4BIT_COMPUTE_DTYPE,
        bnb_4bit_use_double_quant=cfg.BNB_4BIT_USE_DOUBLE_QUANT,
    )

    logger.info(f"Loading base model: {model_id_or_path} with QLoRA config...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        quantization_config=bnb_config,
        device_map=get_device_map(), # Handles multi-GPU or single device
        trust_remote_code=True,
    )
    model.config.use_cache = False  # Important for training
    model.config.pretraining_tp = 1 # Recommended for Qwen, adjust if needed

    logger.info("Preparing model for k-bit training...")
    model = prepare_model_for_kbit_training(model)

    lora_config_obj = LoraConfig(
        r=cfg.LORA_R,
        lora_alpha=cfg.LORA_ALPHA,
        target_modules=cfg.LORA_TARGET_MODULES,
        lora_dropout=cfg.LORA_DROPOUT,
        bias=cfg.LORA_BIAS,
        task_type=cfg.PEFT_TASK_TYPE,
    )

    logger.info("Applying PEFT (LoRA) to the model...")
    model = get_peft_model(model, lora_config_obj)
    model.print_trainable_parameters()
    logger.info("Model configured with LoRA and ready for fine-tuning.")
    return model

def load_model_for_inference(adapter_path_or_id, base_model_id_for_tokenizer=None):
    """
    Loads the fine-tuned model (base + adapter) for inference.
    Uses inference_config for quantization.
    """
    from configs import inference_config as infer_cfg # Local import to avoid circular dependency if utils imports this

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=infer_cfg.BNB_LOAD_IN_4BIT,
        bnb_4bit_quant_type=infer_cfg.BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=infer_cfg.BNB_4BIT_COMPUTE_DTYPE,
        bnb_4bit_use_double_quant=infer_cfg.BNB_4BIT_USE_DOUBLE_QUANT,
    )

    logger.info(f"Loading fine-tuned model from adapter: {adapter_path_or_id}")
    # PeftModel.from_pretrained handles loading the base model and applying the adapter.
    # It needs the base model name if the adapter was saved without it.
    # However, Qwen models often require the base model to be loaded first then adapter merged.
    # For QLoRA, loading base with quantization_config and then adapter is common.
    
    # Determine base model ID. If adapter_path_or_id is a local path,
    # it might not contain base_model_name.json.
    # Qwen typically needs base model path for from_pretrained.
    # If adapter_path_or_id is a Hub ID, HF handles it.
    # If adapter_path_or_id is a local path to an adapter, we need base model id.
    # The `AutoModelForCausalLM.from_pretrained(ADAPTER_PATH)` directly for PEFT models often works if
    # the adapter directory contains the full model structure or refers to it.
    # For QWen + QLoRA, it's safer to load base then adapter.
    
    # Heuristic: if adapter_path_or_id is a local path, use a base model ID
    # This assumes the 'adapter_path_or_id' is *just* the adapter.
    # If 'adapter_path_or_id' is a fully merged model saved with model.save_pretrained(),
    # then it can be loaded directly. The script saves adapters, so this path is for adapters.
    
    # Qwen often requires base model for tokenizer etc.
    # The adapter_path_or_id IS the HF Hub ID which points to the adapter and its base.
    # If it were a local path, we would need the original base model path.
    # The provided inference script uses `exper1ment/email_swapnil` which should work.

    model = AutoModelForCausalLM.from_pretrained(
        adapter_path_or_id, # This should be the Hub ID or a path to the *adapter*
        quantization_config=bnb_config,
        device_map=get_device_map(), # Or infer_cfg.INFERENCE_DEVICE
        trust_remote_code=True,
    )
    model.eval() # Set to evaluation mode
    logger.info("Fine-tuned model loaded successfully for inference.")
    return model

def get_inference_tokenizer(model_id_or_path):
    """Loads tokenizer for inference, setting pad_token and padding_side."""
    from configs import inference_config as infer_cfg # Local import

    tokenizer_path = model_id_or_path
    # if os.path.isdir(model_id_or_path) and infer_cfg.BASE_MODEL_ID_FOR_TOKENIZER:
    #     # If local adapter path and base model specified for tokenizer
    #     tokenizer_path = infer_cfg.BASE_MODEL_ID_FOR_TOKENIZER
    
    logger.info(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set tokenizer.pad_token to eos_token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
    
    tokenizer.padding_side = "left" # Important for generation
    logger.info("Tokenizer for inference loaded and configured.")
    return tokenizer