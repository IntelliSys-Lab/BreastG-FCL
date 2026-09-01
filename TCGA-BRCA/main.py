import logging
import os
import random

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")

from configs.TCGA_BRCA import parse_args
from breastgfcl import ParallelServerGFedCL
from utils.plot_utils import plot_all_tasks_accuracy, plot_results


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logging(opt):
    os.makedirs(opt.output_dir, exist_ok=True)
    log_file_path = opt.log_path or os.path.join(opt.output_dir, "run.log")

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file_path, mode="w")],
    )

    logger = logging.getLogger("GFedCL-TCGA-BRCA-Server")
    logger.info(f"Logging initialized. Log file: {log_file_path}")
    return logger


def main():
    opt = parse_args()
    set_seed(opt.seed)
    logger = setup_logging(opt)

    logger.info("Initializing Parallel Server-based GFedCL for TCGA-BRCA...")
    logger.info(f"Dataset: {opt.dataset}")
    logger.info(f"Number of clients: {opt.num_clients}")
    logger.info(f"Number of tasks per client: {opt.num_task}")
    logger.info(f"Classes: {opt.num_classes}")
    logger.info(f"Expression column: {opt.expression_value_col}")
    logger.info(f"TCIA MRI features: {opt.tcia_mri_features_path}")
    logger.info(f"Max genes/features: {opt.max_genes}")
    logger.info(f"Output directory: {opt.output_dir}")
    logger.info(f"Log file: {opt.log_path}")

    if torch.cuda.is_available():
        logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.warning("CUDA not available, using CPU")

    breastgfcl = ParallelServerGFedCL(opt)
    logger.info("Starting TCGA-BRCA GFedCL training...")
    accuracy_results, all_tasks_accuracy, quality_summary = breastgfcl.train_GFedCL()

    logger.info("Training completed.")
    plots_dir = plot_results(accuracy_results, opt.output_dir)
    if all_tasks_accuracy:
        plot_all_tasks_accuracy(opt, all_tasks_accuracy, plots_dir)

    logger.info("===== TRAINING SUMMARY =====")
    logger.info(f"Dataset: {opt.dataset}")
    logger.info(f"Number of clients: {opt.num_clients}")
    logger.info(f"Number of tasks per client: {opt.num_task}")
    logger.info(f"Overall average accuracy: {accuracy_results['overall_avg_acc']:.2f}%")
    logger.info(f"Results saved to {opt.output_dir}")
    logger.info(f"Log file: {opt.log_path}")
    logger.info(f"Accuracy plots: {plots_dir}")

    logger.info("Performance by task:")
    for task_id, acc in accuracy_results["task_avg_acc"].items():
        logger.info(f"  Task {task_id + 1}: {acc:.2f}%")
    logger.info("===========================")


if __name__ == "__main__":
    os.environ["RAY_memory_monitor_refresh_ms"] = "0"
    main()
