import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import ray
import torch
import torch.nn as nn
from scipy.stats import entropy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.nn import functional as F
from torchvision.models import inception_v3

logger = logging.getLogger("GFedCL")


class FixedLatentSpaceEvaluator:
    """
    Evaluate latent space representations and compute inception scores.
    """

    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device(opt.device)
        self.vis_dir = os.path.join(opt.output_dir, "latent_vis")
        os.makedirs(self.vis_dir, exist_ok=True)
        self.inception_model = None

    def _load_inception_model(self):
        if self.inception_model is None:
            logger.info("Loading Inception V3 model for inception score calculation")
            try:
                model = inception_v3(pretrained=True, transform_input=False).to(self.device)
                model.eval()
                model.fc = nn.Identity()
                self.inception_model = model
            except Exception as exc:
                logger.error(f"Failed to load Inception model: {exc}")
                return False
        return True

    def visualize_latent_space(self, clients, dataloaders, task_id, num_samples=500, method="tsne"):
        logger.info(f"Visualizing latent space for task {task_id} using {method}")

        real_encodings = []
        synthetic_encodings = []
        real_labels = []
        synthetic_labels = []
        client_ids = []

        with torch.no_grad():
            for client_id, client in enumerate(clients):
                try:
                    dataloader = dataloaders[client_id][task_id]["train"]
                    for data, labels in dataloader:
                        data = data.to(self.device)
                        labels = labels.to(self.device)

                        if hasattr(client, "client_relations"):
                            graph_embedding = client.client_relations
                        else:
                            one_hot = torch.zeros(1, self.opt.num_clients, device=self.device)
                            one_hot[0, client_id] = 1.0
                            graph_embedding = one_hot

                        z_seq = client.netG(graph_embedding)
                        e_seq = client.netE(data, labels, z_seq)

                        real_encodings.append(e_seq.cpu().numpy())
                        real_labels.extend(labels.cpu().numpy())
                        client_ids.extend([client_id] * len(labels))

                        if len(real_labels) >= num_samples:
                            break

                    if task_id > 0:
                        noise = torch.randn(
                            min(dataloader.batch_size, num_samples // self.opt.num_clients),
                            self.opt.input_dim,
                            device=self.device,
                        )
                        syn_labels = labels[
                            : min(dataloader.batch_size, num_samples // self.opt.num_clients)
                        ]
                        z_seq = client.netG(graph_embedding)
                        e_seq = client.netE(noise, syn_labels, z_seq)
                        synthetic_encodings.append(e_seq.cpu().numpy())
                        synthetic_labels.extend(syn_labels.cpu().numpy())
                except Exception as exc:
                    logger.error(f"Error processing client {client_id}: {exc}")

        if real_encodings:
            real_encodings = np.vstack(real_encodings)
            real_labels = np.array(real_labels)
            client_ids = np.array(client_ids)
        else:
            logger.error("No real encodings collected")
            return None

        if synthetic_encodings:
            synthetic_encodings = np.vstack(synthetic_encodings)
            synthetic_labels = np.array(synthetic_labels)

        if method == "tsne":
            logger.info(f"Applying t-SNE to {len(real_encodings)} real samples")
            real_embeddings = TSNE(n_components=2, random_state=self.opt.seed).fit_transform(
                real_encodings
            )
            if len(synthetic_encodings) > 0:
                logger.info(f"Applying t-SNE to {len(synthetic_encodings)} synthetic samples")
                synthetic_embeddings = TSNE(n_components=2, random_state=self.opt.seed).fit_transform(
                    synthetic_encodings
                )
        else:
            logger.info(f"Applying PCA to {len(real_encodings)} real samples")
            pca = PCA(n_components=2, random_state=self.opt.seed)
            real_embeddings = pca.fit_transform(real_encodings)
            if len(synthetic_encodings) > 0:
                logger.info(f"Applying PCA to {len(synthetic_encodings)} synthetic samples")
                synthetic_embeddings = pca.transform(synthetic_encodings)

        plt.figure(figsize=(12, 10))
        plt.subplot(1, 1 if len(synthetic_encodings) == 0 else 2, 1)
        scatter = plt.scatter(
            real_embeddings[:, 0],
            real_embeddings[:, 1],
            c=real_labels,
            cmap="tab10",
            alpha=0.7,
            s=50,
        )
        plt.colorbar(scatter, label="Class Label")
        plt.title(f"Real Samples Latent Space - Task {task_id + 1} ({method.upper()})")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.grid(alpha=0.3)

        if len(synthetic_encodings) > 0:
            plt.subplot(1, 2, 2)
            scatter = plt.scatter(
                synthetic_embeddings[:, 0],
                synthetic_embeddings[:, 1],
                c=synthetic_labels,
                cmap="tab10",
                alpha=0.7,
                s=50,
            )
            plt.colorbar(scatter, label="Class Label")
            plt.title(
                f"Synthetic Samples Latent Space - Task {task_id + 1} ({method.upper()})"
            )
            plt.xlabel("Dimension 1")
            plt.ylabel("Dimension 2")
            plt.grid(alpha=0.3)

        plt.tight_layout()
        file_path = os.path.join(self.vis_dir, f"latent_space_task{task_id + 1}_{method}.png")
        plt.savefig(file_path, bbox_inches="tight", dpi=300)
        plt.close()

        logger.info(f"Saved latent space visualization to {file_path}")
        return file_path

    @ray.remote
    def _calculate_inception_score_batch(self, images):
        if not self._load_inception_model():
            return None

        self.inception_model.eval()

        with torch.no_grad():
            try:
                if images.shape[1] == 1:
                    images = images.repeat(1, 3, 1, 1)
                if images.shape[2] != 299 or images.shape[3] != 299:
                    images = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)

                preds = F.softmax(self.inception_model(images), dim=1).cpu().numpy()
                return preds
            except Exception as exc:
                logger.error(f"Error in inception score calculation: {exc}")
                return None

    def calculate_inception_score(self, images, splits=10, batch_size=32):
        if not self._load_inception_model():
            return None, None

        predictions = []
        n_images = len(images)

        results = []
        for i in range(0, n_images, batch_size * self.opt.ray_max_in_flight):
            futures = []
            chunk = images[i : i + batch_size * self.opt.ray_max_in_flight]
            for j in range(0, len(chunk), batch_size):
                batch = chunk[j : j + batch_size].to(self.device)
                futures.append(
                    self._calculate_inception_score_batch.options(
                        num_gpus=self.opt.ray_num_gpus_per_task,
                        num_cpus=self.opt.ray_num_cpus_per_task,
                    ).remote(self, batch)
                )
            results.extend(ray.get(futures))
        for result in results:
            if result is not None:
                predictions.append(result)

        if not predictions:
            logger.warning("No predictions for inception score calculation")
            return 0.0, 0.0

        predictions = np.concatenate(predictions, axis=0)

        scores = []
        split_size = predictions.shape[0] // splits if splits > 0 else predictions.shape[0]
        for i in range(splits):
            part = predictions[i * split_size : (i + 1) * split_size]
            if part.shape[0] == 0:
                continue
            p_y = np.expand_dims(part.mean(axis=0), 0)
            kl_div = part * (np.log(part + 1e-10) - np.log(p_y + 1e-10))
            sum_kl_div = kl_div.sum(axis=1)
            scores.append(np.exp(sum_kl_div.mean()))

        if not scores:
            return 0.0, 0.0
        return float(np.mean(scores)), float(np.std(scores))

    @ray.remote
    def collect_client_samples(self, client, dataloader, task_id, generate_synthetic, num_samples):
        client.eval()
        real_features = []
        synthetic_features = []

        with torch.no_grad():
            for data, labels in dataloader:
                data = data.to(self.device)
                labels = labels.to(self.device)
                real_features.append(data)
                if sum(len(batch) for batch in real_features) >= num_samples:
                    break

            real_features = torch.cat(real_features, dim=0)[:num_samples]

            if generate_synthetic and task_id > 0:
                batch_size = min(dataloader.batch_size, num_samples)
                generated = 0
                for i in range((num_samples + batch_size - 1) // batch_size):
                    for data, labels in dataloader:
                        batch_labels = labels[
                            : min(batch_size, num_samples - i * batch_size)
                        ].to(self.device)
                        break
                    current_batch = batch_labels.size(0)
                    noise = torch.randn(
                        current_batch,
                        self.opt.input_dim,
                        device=self.device,
                    )

                    if hasattr(client, "client_relations"):
                        graph_embedding = client.client_relations
                    else:
                        one_hot = torch.zeros(1, self.opt.num_clients, device=self.device)
                        one_hot[0, client.client_id] = 1.0
                        graph_embedding = one_hot

                    z_seq = client.netG(graph_embedding)
                    client.netE(noise, batch_labels, z_seq)
                    synthetic_features.append(noise)
                    generated += current_batch
                    if generated >= num_samples:
                        break

                synthetic_features = torch.cat(synthetic_features, dim=0)[:num_samples]

        return {
            "client_id": client.client_id,
            "real_features": real_features,
            "synthetic_features": synthetic_features if generate_synthetic and task_id > 0 else None,
        }

    def evaluate_inception_scores(
        self, clients, dataloaders, task_id, generate_synthetic=False, num_samples_per_client=500
    ):
        logger.info(f"Calculating average inception scores for task {task_id}")

        results = []
        for i in range(0, len(clients), self.opt.ray_max_in_flight):
            futures = []
            batch_clients = clients[i : i + self.opt.ray_max_in_flight]
            for j, client in enumerate(batch_clients):
                client_id = i + j
                if task_id in dataloaders[client_id]:
                    futures.append(
                        self.collect_client_samples.options(
                            num_gpus=self.opt.ray_num_gpus_per_task,
                            num_cpus=self.opt.ray_num_cpus_per_task,
                        ).remote(
                            self,
                            client,
                            dataloaders[client_id][task_id]["train"],
                            task_id,
                            generate_synthetic,
                            num_samples_per_client,
                        )
                    )
            if futures:
                results.extend(ray.get(futures))
        all_real_features = []
        all_synthetic_features = []

        for result in results:
            all_real_features.append(result["real_features"])
            if result["synthetic_features"] is not None:
                all_synthetic_features.append(result["synthetic_features"])

        combined_real_features = (
            torch.cat(all_real_features, dim=0) if all_real_features else None
        )
        combined_synthetic_features = (
            torch.cat(all_synthetic_features, dim=0) if all_synthetic_features else None
        )

        real_score = (0.0, 0.0)
        synthetic_score = (0.0, 0.0)

        if combined_real_features is not None and len(combined_real_features) > 0:
            logger.info("Calculating global inception score for real images")
            real_score = self.calculate_inception_score(combined_real_features)

        if combined_synthetic_features is not None and len(combined_synthetic_features) > 0:
            logger.info("Calculating global inception score for synthetic images")
            synthetic_score = self.calculate_inception_score(combined_synthetic_features)

        scores = {
            "task_id": task_id,
            "avg_real_score": real_score[0],
            "avg_real_score_std": real_score[1],
            "avg_synthetic_score": synthetic_score[0],
            "avg_synthetic_score_std": synthetic_score[1],
            "clients": {},
        }

        for client_id in range(len(clients)):
            scores["clients"][client_id] = {
                "real_score": real_score[0],
                "real_score_std": real_score[1],
                "synthetic_score": synthetic_score[0],
                "synthetic_score_std": synthetic_score[1],
            }

        self._save_inception_scores(scores)
        return scores

    def _save_inception_scores(self, scores):
        import csv

        csv_path = os.path.join(self.opt.output_dir, "inception_scores.csv")
        file_exists = os.path.isfile(csv_path)

        with open(csv_path, "a", newline="") as csvfile:
            fieldnames = [
                "task_id",
                "client_id",
                "real_score",
                "real_score_std",
                "synthetic_score",
                "synthetic_score_std",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()

            for client_id, client_scores in scores["clients"].items():
                writer.writerow(
                    {
                        "task_id": scores["task_id"],
                        "client_id": client_id,
                        "real_score": client_scores["real_score"],
                        "real_score_std": client_scores["real_score_std"],
                        "synthetic_score": client_scores["synthetic_score"],
                        "synthetic_score_std": client_scores["synthetic_score_std"],
                    }
                )

        logger.info(f"Saved inception scores to {csv_path}")

        global_csv_path = os.path.join(self.opt.output_dir, "global_inception_scores.csv")
        global_file_exists = os.path.isfile(global_csv_path)

        with open(global_csv_path, "a", newline="") as csvfile:
            fieldnames = [
                "task_id",
                "avg_real_score",
                "avg_real_score_std",
                "avg_synthetic_score",
                "avg_synthetic_score_std",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not global_file_exists:
                writer.writeheader()

            writer.writerow(
                {
                    "task_id": scores["task_id"],
                    "avg_real_score": scores["avg_real_score"],
                    "avg_real_score_std": scores["avg_real_score_std"],
                    "avg_synthetic_score": scores["avg_synthetic_score"],
                    "avg_synthetic_score_std": scores["avg_synthetic_score_std"],
                }
            )

        logger.info(f"Saved global inception scores to {global_csv_path}")

    def plot_inception_scores(self):
        import pandas as pd

        global_csv_path = os.path.join(self.opt.output_dir, "global_inception_scores.csv")
        if not os.path.isfile(global_csv_path):
            logger.warning("No global inception scores file found for plotting")
            return None

        df = pd.read_csv(global_csv_path)
        if df.empty:
            logger.warning("No data found for inception score plotting")
            return None

        plt.figure(figsize=(10, 6))
        plt.errorbar(
            df["task_id"] + 1,
            df["avg_real_score"],
            yerr=df["avg_real_score_std"],
            label="Real",
            marker="o",
        )
        plt.errorbar(
            df["task_id"] + 1,
            df["avg_synthetic_score"],
            yerr=df["avg_synthetic_score_std"],
            label="Synthetic",
            marker="o",
        )
        plt.xlabel("Task")
        plt.ylabel("Inception Score")
        plt.title("Inception Scores by Task")
        plt.legend()
        plt.grid(alpha=0.3)

        file_path = os.path.join(self.opt.output_dir, "inception_scores.png")
        plt.savefig(file_path, bbox_inches="tight", dpi=300)
        plt.close()

        logger.info(f"Saved inception score plot to {file_path}")
        return file_path
