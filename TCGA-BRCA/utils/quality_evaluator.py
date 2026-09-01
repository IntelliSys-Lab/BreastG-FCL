import csv
import logging
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import ray
import torch
from torch.utils.data import DataLoader, TensorDataset

from utils.fid_calculation import SimplifiedFID
from utils.latent_space_evaluator import FixedLatentSpaceEvaluator

logger = logging.getLogger("GFedCL")


@ray.remote
def generate_fid_data_remote(client, task_id, relational_graphs, dataloader, max_batches, generate_synthetic):
    logger.info(f"Generating FID data from client {client.getId()} for task {task_id}")

    client.task_ID = task_id
    client.relational_graph = relational_graphs
    client.eval()

    real_images = []
    real_labels = []
    synthetic_images = []
    synthetic_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            inputs, targets = batch
            inputs = inputs.to(client.device)
            targets = targets.to(client.device)

            real_images.append(inputs.cpu())
            real_labels.append(targets.cpu())

            if generate_synthetic and task_id > 0:
                synthetic_images.append(torch.randn_like(inputs).cpu())
                synthetic_labels.append(targets.cpu())

    if not real_images:
        logger.warning(f"No real images collected for client {client.getId()}")
        return {"real_data": None, "synthetic_data": None}

    if generate_synthetic and not synthetic_images:
        logger.warning(f"No synthetic images generated for client {client.getId()}")
        return {"real_data": None, "synthetic_data": None}

    real_data = {
        "images": torch.cat(real_images, dim=0),
        "labels": torch.cat(real_labels, dim=0),
    }
    synthetic_data = {
        "images": torch.cat(synthetic_images, dim=0),
        "labels": torch.cat(synthetic_labels, dim=0),
    }

    return {"real_data": real_data, "synthetic_data": synthetic_data}


@ray.remote
def calculate_client_fid_remote(real_data, synthetic_data, device):
    if real_data is None or synthetic_data is None:
        return None

    if real_data["images"] is None or synthetic_data["images"] is None:
        return None

    if len(real_data["images"]) < 2 or len(synthetic_data["images"]) < 2:
        logger.warning(
            "Not enough samples for FID calculation. "
            f"Real: {len(real_data['images'])}, Synthetic: {len(synthetic_data['images'])}"
        )
        return None

    batch_size = min(64, len(real_data["images"]))
    real_loader = DataLoader(
        TensorDataset(real_data["images"], real_data["labels"]),
        batch_size=batch_size,
        shuffle=False,
    )
    synthetic_loader = DataLoader(
        TensorDataset(synthetic_data["images"], synthetic_data["labels"]),
        batch_size=batch_size,
        shuffle=False,
    )

    fid_calculator = SimplifiedFID(device=device)
    return fid_calculator.calculate_fid(real_loader, synthetic_loader)


class QualityEvaluator:
    def __init__(self, opt):
        self.opt = opt
        self.enable_fid = bool(getattr(opt, "eval_fid", False))
        self.enable_is = bool(getattr(opt, "eval_is", False))

        self.round_fid_scores = []
        self.round_inception_scores = []

        self.fid_scores_dir = os.path.join(opt.output_dir, "fid_scores")
        self.inception_scores_dir = os.path.join(opt.output_dir, "inception_scores")

        if self.enable_fid:
            os.makedirs(self.fid_scores_dir, exist_ok=True)
        if self.enable_is:
            os.makedirs(self.inception_scores_dir, exist_ok=True)
            self.latent_evaluator = FixedLatentSpaceEvaluator(opt)

    def evaluate_round(self, task_id, round_id, clients, dataloaders, relational_graphs):
        if self.enable_fid:
            self._evaluate_fid_round(task_id, round_id, clients, dataloaders, relational_graphs)
        if self.enable_is:
            self._evaluate_inception_round(task_id, round_id, clients, dataloaders)

    def _evaluate_fid_round(self, task_id, round_id, clients, dataloaders, relational_graphs):
        if task_id <= 0:
            return

        logger.info(f"Calculating FID scores for Task {task_id + 1}, Round {round_id + 1}...")

        client_subset = min(getattr(self.opt, "fid_num_clients", 3), len(clients))
        selected_client_ids = random.sample(range(len(clients)), client_subset)
        max_batches = getattr(self.opt, "fid_max_batches", 5)

        fid_data_results = []
        for i in range(0, len(selected_client_ids), self.opt.ray_max_in_flight):
            futures = []
            batch_ids = selected_client_ids[i : i + self.opt.ray_max_in_flight]
            for client_id in batch_ids:
                futures.append(
                    generate_fid_data_remote.options(
                        num_gpus=self.opt.ray_num_gpus_per_task,
                        num_cpus=self.opt.ray_num_cpus_per_task,
                    ).remote(
                        clients[client_id],
                        task_id,
                        relational_graphs,
                        dataloaders[client_id][task_id]["train"],
                        max_batches,
                        True,
                    )
                )
            fid_data_results.extend(ray.get(futures))

        fid_scores = []
        for i in range(0, len(fid_data_results), self.opt.ray_max_in_flight):
            futures = []
            batch_results = fid_data_results[i : i + self.opt.ray_max_in_flight]
            for result in batch_results:
                if result["real_data"] is None or result["synthetic_data"] is None:
                    continue
                futures.append(
                    calculate_client_fid_remote.options(
                        num_gpus=self.opt.ray_num_gpus_per_task,
                        num_cpus=self.opt.ray_num_cpus_per_task,
                    ).remote(
                        result["real_data"],
                        result["synthetic_data"],
                        self.opt.device,
                    )
                )

            if futures:
                fid_scores.extend(ray.get(futures))

        if not fid_scores:
            logger.warning(f"No FID futures created for Task {task_id + 1}, Round {round_id + 1}")
            self.round_fid_scores.append(
                {"round": f"Task {task_id + 1}, Round {round_id + 1}", "fid_score": float("nan")}
            )
            return
        valid_scores = [score for score in fid_scores if score is not None and not np.isnan(score)]

        if valid_scores:
            avg_fid_score = float(sum(valid_scores) / len(valid_scores))
            logger.info(
                f"Task {task_id + 1}, Round {round_id + 1} - Average FID Score: {avg_fid_score:.4f}"
            )
            self.round_fid_scores.append(
                {
                    "round": f"Task {task_id + 1}, Round {round_id + 1}",
                    "fid_score": avg_fid_score,
                    "num_clients": len(valid_scores),
                }
            )
        else:
            logger.warning(
                f"No valid FID scores calculated for Task {task_id + 1}, Round {round_id + 1}"
            )
            self.round_fid_scores.append(
                {
                    "round": f"Task {task_id + 1}, Round {round_id + 1}",
                    "fid_score": float("nan"),
                }
            )

        self._save_fid_scores()

    def _save_fid_scores(self):
        if not self.round_fid_scores:
            return

        csv_path = os.path.join(self.opt.output_dir, "round_fid_scores.csv")
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Round", "FID Score", "Number of Clients"])
            for score_data in self.round_fid_scores:
                fid_score = score_data.get("fid_score", float("nan"))
                num_clients = score_data.get("num_clients", 0)
                if np.isnan(fid_score):
                    score_str = "nan"
                else:
                    score_str = f"{fid_score:.4f}"
                writer.writerow([score_data["round"], score_str, num_clients])

        logger.info(f"Saved round FID score data to {csv_path}")

    def _plot_fid_scores(self):
        if not self.round_fid_scores:
            return None

        rounds = [data["round"] for data in self.round_fid_scores]
        fid_scores = [data["fid_score"] for data in self.round_fid_scores]

        valid_indices = [i for i, score in enumerate(fid_scores) if not np.isnan(score)]
        if not valid_indices:
            logger.warning("No valid FID scores to plot")
            return None

        valid_scores = [fid_scores[i] for i in valid_indices]
        valid_rounds = [rounds[i] for i in valid_indices]

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(valid_scores) + 1), valid_scores, "o-", color="#e41a1c")
        plt.ylabel("FID Score (lower is better)")
        plt.xlabel("Round")
        plt.title("FID Score Trend")
        plt.xticks(range(1, len(valid_scores) + 1), valid_rounds, rotation=45, ha="right")
        plt.grid(alpha=0.3)

        file_path = os.path.join(self.opt.output_dir, "fid_scores.png")
        plt.tight_layout()
        plt.savefig(file_path, dpi=300)
        plt.close()

        logger.info(f"Saved FID score plot to {file_path}")
        return file_path

    def _evaluate_inception_round(self, task_id, round_id, clients, dataloaders):
        if task_id <= 0:
            return

        logger.info(
            f"Calculating inception scores for Task {task_id + 1}, Round {round_id + 1}..."
        )
        scores = self.latent_evaluator.evaluate_inception_scores(
            clients,
            dataloaders,
            task_id,
            generate_synthetic=True,
            num_samples_per_client=getattr(self.opt, "inception_samples_per_client", 500),
        )

        scores["round_label"] = f"Task {task_id + 1}, Round {round_id + 1}"
        scores["round_number"] = round_id + 1
        self.round_inception_scores.append(scores)

        round_csv_path = os.path.join(
            self.inception_scores_dir, f"inception_scores_task{task_id + 1}_round{round_id + 1}.csv"
        )
        with open(round_csv_path, "w", newline="") as csvfile:
            fieldnames = [
                "task_id",
                "round_id",
                "client_id",
                "real_score",
                "real_score_std",
                "synthetic_score",
                "synthetic_score_std",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for client_id, client_scores in scores["clients"].items():
                writer.writerow(
                    {
                        "task_id": task_id,
                        "round_id": round_id,
                        "client_id": client_id,
                        "real_score": client_scores["real_score"],
                        "real_score_std": client_scores["real_score_std"],
                        "synthetic_score": client_scores["synthetic_score"],
                        "synthetic_score_std": client_scores["synthetic_score_std"],
                    }
                )

        logger.info(f"Saved round inception scores to {round_csv_path}")

    def _save_all_inception_scores(self):
        if not self.round_inception_scores:
            return None

        combined_csv_path = os.path.join(self.opt.output_dir, "all_inception_scores.csv")
        with open(combined_csv_path, "w", newline="") as csvfile:
            fieldnames = [
                "task_id",
                "round_id",
                "round_label",
                "avg_real_score",
                "avg_real_score_std",
                "avg_synthetic_score",
                "avg_synthetic_score_std",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for score_data in self.round_inception_scores:
                writer.writerow(
                    {
                        "task_id": score_data["task_id"],
                        "round_id": score_data["round_number"],
                        "round_label": score_data["round_label"],
                        "avg_real_score": score_data["avg_real_score"],
                        "avg_real_score_std": score_data["avg_real_score_std"],
                        "avg_synthetic_score": score_data["avg_synthetic_score"],
                        "avg_synthetic_score_std": score_data["avg_synthetic_score_std"],
                    }
                )

        logger.info(f"Saved all inception scores to {combined_csv_path}")
        return combined_csv_path

    def _plot_inception_score_trends(self):
        if not self.round_inception_scores:
            return None

        rounds = [score_data["round_number"] for score_data in self.round_inception_scores]
        real_scores = [score_data["avg_real_score"] for score_data in self.round_inception_scores]
        synthetic_scores = [
            score_data["avg_synthetic_score"] for score_data in self.round_inception_scores
        ]

        plt.figure(figsize=(10, 6))
        plt.plot(rounds, real_scores, "o-", label="Real")
        plt.plot(rounds, synthetic_scores, "o-", label="Synthetic")
        plt.xlabel("Round")
        plt.ylabel("Inception Score")
        plt.title("Inception Score Trends")
        plt.legend()
        plt.grid(alpha=0.3)

        file_path = os.path.join(self.opt.output_dir, "inception_score_trends.png")
        plt.tight_layout()
        plt.savefig(file_path, dpi=300)
        plt.close()

        logger.info(f"Saved inception score trend visualization to {file_path}")
        return file_path

    def finalize(self):
        summary = {}

        if self.enable_fid:
            self._save_fid_scores()
            self._plot_fid_scores()
            summary["fid_scores"] = list(self.round_fid_scores)

        if self.enable_is:
            self._save_all_inception_scores()
            self._plot_inception_score_trends()
            self.latent_evaluator.plot_inception_scores()
            summary["inception_scores"] = list(self.round_inception_scores)

        return summary
