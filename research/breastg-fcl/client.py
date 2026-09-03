# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Site-local BreastG-FCL operations published through the Collab API."""

import torch
import torch.nn.functional as F
from nvflare.collab import collab

from config import ExperimentConfig
from data import SyntheticSiteData
from model import get_model_state, load_model_state, make_discriminator, make_model, resolve_device


def _laplace_noise_like(values: torch.Tensor, scale: float) -> torch.Tensor:
    if scale == 0:
        return torch.zeros_like(values)
    distribution = torch.distributions.Laplace(
        torch.tensor(0.0, device=values.device),
        torch.tensor(scale, device=values.device),
    )
    return distribution.sample(values.shape)


class BreastGFCLClient:
    """Persistent client object with remotely callable research operations."""

    def __init__(self, config: ExperimentConfig):
        self.config = config.validate()
        self.device = None
        self.site_index = None
        self.data = None
        self.model = None
        self.discriminator = None
        self.optimizer = None

    @collab.init
    def initialize(self):
        self.site_index = int(collab.get_app_prop("site_index"))
        torch.manual_seed(self.config.seed + self.site_index)
        self.device = resolve_device()
        self.data = SyntheticSiteData(self.config, self.site_index)
        self.model = make_model(self.config, self.device)
        self.discriminator = make_discriminator(self.config, self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        print(f"[{collab.site_name}] initialized on {self.device}")

    def _graph_row(self, site_order: list[str], relational_graph: torch.Tensor) -> torch.Tensor:
        if collab.site_name not in site_order:
            raise ValueError(f"{collab.site_name} is missing from site_order={site_order}")
        matrix = torch.as_tensor(relational_graph, dtype=torch.float32, device=self.device)
        expected_shape = (self.config.num_clients, self.config.num_clients)
        if tuple(matrix.shape) != expected_shape:
            raise ValueError(f"relational_graph must have shape {expected_shape}, got {tuple(matrix.shape)}")
        return matrix[site_order.index(collab.site_name)]

    def _load_global_model(self, global_model: dict[str, torch.Tensor]) -> None:
        load_model_state(self.model, global_model)

    @collab.publish
    def get_graph_summary(self, task_id: int) -> dict:
        spatial, temporal = self.data.graph_summary(task_id)
        if self.config.summary_noise_scale:
            spatial += _laplace_noise_like(spatial, self.config.summary_noise_scale)
            temporal += _laplace_noise_like(temporal, self.config.summary_noise_scale)
        task = self.data.tasks[task_id]
        return {
            "spatial": spatial.cpu(),
            "temporal": temporal.cpu(),
            "train_size": len(task.train),
            "test_size": len(task.test),
        }

    @collab.publish
    def collect_encodings(
        self,
        task_id: int,
        global_model: dict[str, torch.Tensor],
        site_order: list[str],
        relational_graph: torch.Tensor,
    ) -> dict:
        """Encode local samples and perturb them before leaving the site."""

        self._load_global_model(global_model)
        graph_row = self._graph_row(site_order, relational_graph)
        task_ids = [task_id]
        if self.config.replay_previous_task and task_id > 0:
            task_ids.append(task_id - 1)

        encodings = []
        graph_targets = []
        self.model.eval()
        with torch.no_grad():
            for local_task_id in task_ids:
                loader = self.data.loader(local_task_id, train=True, shuffle=False)
                for batch_index, (inputs, _labels) in enumerate(loader):
                    if collab.is_aborted:
                        raise RuntimeError("Encoding collection aborted")
                    if batch_index >= self.config.max_encoding_batches:
                        break
                    inputs = inputs.to(self.device)
                    latent, graph_embedding = self.model.encode(inputs, graph_row)
                    encodings.append(latent.cpu())
                    graph_targets.append(graph_embedding.cpu())

        if not encodings:
            raise RuntimeError(f"{collab.site_name} produced no encodings for task {task_id}")
        encoded = torch.cat(encodings, dim=0)
        encoded += _laplace_noise_like(encoded, self.config.encoding_noise_scale)
        return {
            "encodings": encoded.cpu(),
            "graph_targets": torch.cat(graph_targets, dim=0).cpu(),
            "num_examples": encoded.shape[0],
        }

    @collab.publish
    def train(
        self,
        task_id: int,
        global_model: dict[str, torch.Tensor],
        discriminator_state: dict[str, torch.Tensor],
        site_order: list[str],
        relational_graph: torch.Tensor,
    ) -> dict:
        self._load_global_model(global_model)
        load_model_state(self.discriminator, discriminator_state)
        self.discriminator.eval()
        for parameter in self.discriminator.parameters():
            parameter.requires_grad_(False)

        graph_row = self._graph_row(site_order, relational_graph)
        task_ids = [task_id]
        if self.config.replay_previous_task and task_id > 0:
            task_ids.append(task_id - 1)

        num_steps = 0
        num_examples = 0
        loss_sum = 0.0
        self.model.train()
        for _epoch in range(self.config.local_epochs):
            for local_task_id in task_ids:
                for inputs, labels in self.data.loader(local_task_id, train=True):
                    if collab.is_aborted:
                        raise RuntimeError("Client training aborted")
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    self.optimizer.zero_grad()
                    logits, latent, graph_embedding = self.model(inputs, graph_row)
                    prediction_loss = F.cross_entropy(logits, labels)
                    graph_prediction = self.discriminator(latent)
                    privacy_loss = -F.mse_loss(graph_prediction, graph_embedding.detach())
                    loss = prediction_loss + self.config.privacy_weight * privacy_loss
                    loss.backward()
                    self.optimizer.step()
                    batch_size = labels.shape[0]
                    num_steps += 1
                    num_examples += batch_size
                    loss_sum += loss.detach().item() * batch_size

        if num_examples == 0:
            raise RuntimeError(f"{collab.site_name} trained on no examples")
        print(f"[{collab.site_name}] task={task_id} steps={num_steps} " f"loss={loss_sum / num_examples:.4f}")
        return {
            "model": get_model_state(self.model),
            "num_steps": num_steps,
            "num_examples": num_examples,
            "loss": loss_sum / num_examples,
        }

    @collab.publish
    def evaluate(
        self,
        current_task_id: int,
        global_model: dict[str, torch.Tensor],
        site_order: list[str],
        relational_graphs: list[torch.Tensor],
    ) -> dict:
        self._load_global_model(global_model)
        self.model.eval()
        task_metrics = {}
        with torch.no_grad():
            for task_id in range(current_task_id + 1):
                graph_row = self._graph_row(site_order, relational_graphs[task_id])
                correct = 0
                total = 0
                loss_sum = 0.0
                for inputs, labels in self.data.loader(task_id, train=False):
                    if collab.is_aborted:
                        raise RuntimeError("Client evaluation aborted")
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    logits, _latent, _graph_embedding = self.model(inputs, graph_row)
                    loss_sum += F.cross_entropy(logits, labels, reduction="sum").item()
                    correct += logits.argmax(dim=1).eq(labels).sum().item()
                    total += labels.shape[0]
                task_metrics[str(task_id)] = {
                    "loss_sum": loss_sum,
                    "correct": correct,
                    "total": total,
                }
        return {"tasks": task_metrics}
