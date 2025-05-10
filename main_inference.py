import logging
from src.utils import setup_logging
from src.inference import run_inference # Import the main orchestrator

def main():
    setup_logging() # Basic logging setup
    logger = logging.getLogger(__name__)

    logger.info("--- Starting Inference Process ---")
    run_inference()
    logger.info("--- Inference Process Completed ---")

if __name__ == "__main__":
    main()