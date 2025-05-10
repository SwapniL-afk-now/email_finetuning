import matplotlib.pyplot as plt
import pandas as pd
import os
import logging
from configs import finetune_config as cfg

logger = logging.getLogger(__name__)

def plot_training_metrics(metrics_history, output_dir):
    """Plots training and evaluation metrics."""
    if not metrics_history:
        logger.warning("No metrics history to plot.")
        return

    df_history = pd.DataFrame(metrics_history)
    plot_path = os.path.join(output_dir, "training_metrics_plot.png")
    cfg.ensure_dir_exists(plot_path)


    plt.figure(figsize=(18, 6))

    # Plotting Eval Loss
    plt.subplot(1, 3, 1)
    if 'optimizer_step' in df_history.columns and 'eval_loss' in df_history.columns:
        plt.plot(df_history['optimizer_step'], df_history['eval_loss'], label='Validation Loss', marker='o', linestyle='-')
    plt.title('Validation Loss vs. Optimizer Steps')
    plt.xlabel('Optimizer Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plotting ROUGE Scores (R1, R2)
    plt.subplot(1, 3, 2)
    if 'optimizer_step' in df_history.columns and 'eval_rouge1' in df_history.columns:
        plt.plot(df_history['optimizer_step'], df_history['eval_rouge1'], label='Validation ROUGE-1', marker='o', linestyle='-')
    if 'optimizer_step' in df_history.columns and 'eval_rouge2' in df_history.columns:
        plt.plot(df_history['optimizer_step'], df_history['eval_rouge2'], label='Validation ROUGE-2', marker='s', linestyle='-')
    plt.title('Validation ROUGE-1 & ROUGE-2')
    plt.xlabel('Optimizer Steps')
    plt.ylabel('F-measure')
    plt.legend()
    plt.grid(True)
    
    # Plotting ROUGE-L Score
    plt.subplot(1, 3, 3)
    if 'optimizer_step' in df_history.columns and 'eval_rougeL' in df_history.columns:
        plt.plot(df_history['optimizer_step'], df_history['eval_rougeL'], label='Validation ROUGE-L', marker='^', color='green', linestyle='-')
    plt.title('Validation ROUGE-L Score')
    plt.xlabel('Optimizer Steps')
    plt.ylabel('F-measure')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    try:
        plt.savefig(plot_path)
        logger.info(f"Training metrics plot saved to {plot_path}")
        # plt.show() # Uncomment if running interactively and want to display plot
    except Exception as e:
        logger.error(f"Error saving metrics plot: {e}")
    plt.close()