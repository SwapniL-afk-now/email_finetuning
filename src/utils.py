import random
import numpy as np
import torch

import logging
import os

logger = logging.getLogger(__name__)

def setup_logging(level=logging.INFO):
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

def set_seed(seed_value):
    """Sets the seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    logger.info(f"Set seed to {seed_value}")




def get_device_map():
    """Returns 'auto' for multi-GPU or specific device for single GPU/CPU."""
    if torch.cuda.is_available():
        return "auto" if torch.cuda.device_count() > 1 else None # None means transformers will pick default cuda:0
    return "cpu"

def get_main_device(model=None):
    """
    Determines the main device for placing data, especially with model parallelism or device_map.
    If a model with hf_device_map is provided, it tries to find the primary CUDA device.
    Otherwise, defaults to CUDA if available, else CPU.
    """
    if model and hasattr(model, 'hf_device_map') and model.hf_device_map:
        # Heuristic: Use the device of the first parameter or a common one.
        try:
            first_param_device_str = str(next(model.parameters()).device)
            if 'cuda' in first_param_device_str:
                return torch.device(first_param_device_str)
            else: # If first part is on CPU (e.g. embeddings offloaded)
                cuda_devices_used = {str(p.device) for p in model.parameters() if 'cuda' in str(p.device)}
                if cuda_devices_used:
                    return torch.device(sorted(list(cuda_devices_used))[0]) # Smallest CUDA index
        except StopIteration: # No parameters
            pass
        except Exception as e:
            logger.warning(f"Error determining main device from model: {e}. Falling back.")

    # Fallback or if no model provided
    if torch.cuda.is_available():
        return torch.device("cuda:0" if torch.cuda.device_count() == 1 else "cuda") # "cuda" will pick default
    return torch.device("cpu")


def ensure_dir_exists(path):
    """Ensures that a directory exists, creating it if necessary."""
    dir_name = os.path.dirname(path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        logger.info(f"Created directory: {dir_name}")