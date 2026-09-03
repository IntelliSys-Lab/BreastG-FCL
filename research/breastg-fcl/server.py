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

"""Server-side BreastG-FCL workflow expressed as Collab remote calls."""

import torch
import torch.nn.functional as F
from nvflare.collab import collab

from aggregation import aggregate_evaluation, aggregate_model_results, require_complete_results
from config import ExperimentConfig
from graph import BreastGraphGenerator
from model import get_model_state, make_discriminator, make_model, resolve_device


class BreastGFCLServer:
    """Coordinate the multi-task, multi-phase BreastG-FCL algorithm."""

    def __init__(self, config: ExperimentConfig):
        self.config = config.validate()

    def _clients(self):
        options = {"timeout": self.config.call_timeout}
        if self.config.max_parallel_calls:
            options["parallel"] = self.config.max_parallel_calls
        return collab.clients(**options)

    def _fit_discriminator(
        self,
        discriminator,
        optimizer,
        encoding_results: list[tuple[str, dict]],
        device: torch.device,
    ) -> float:
        encodings = torch.cat([result["encodings"] for _site, result in encoding_results], dim=0).to(device)
        graph_targets = torch.cat([result["graph_targets"] for _site, result in encoding_results], dim=0).to(device)
        if encodings.shape[0] != graph_targets.shape[0]:
            raise ValueError(
                f"encoding and graph-target counts differ: {encodings.shape[0]} != {graph_targets.shape[0]}"
            )

        discriminator.train()
        final_loss = 0.0
        for _epoch in range(self.config.discriminator_epochs):
            optimizer.zero_grad()
            prediction = discriminator(encodings.detach())
            loss = F.mse_loss(prediction, graph_targets.detach())
            loss.backward()
            optimizer.step()
            final_loss = loss.detach().item()
        return final_loss

    @collab.main
    def run(self) -> dict:
        torch.manual_seed(self.config.seed)
        device = resolve_device()
        model = make_model(self.config, device)
        discriminator = make_discriminator(self.config, device)
        discriminator_optimizer = torch.optim.Adam(
            discriminator.parameters(),
            lr=self.config.discriminator_learning_rate,
        )
        graph_generator = BreastGraphGenerator(
            num_clients=self.config.num_clients,
            temporal_window=self.config.temporal_window,
            temperature=self.config.attention_temperature,
            epsilon=self.config.graph_epsilon,
        )

        global_model = get_model_state(model)
        site_order = None
        graphs = []
        graph_artifacts = []
        history = []
        global_round = 0

        for task_id in range(self.config.num_tasks):
            summary_call = self._clients().get_graph_summary(task_id)
            summary_results = require_complete_results(summary_call, site_order)
            if len(summary_results) != self.config.num_clients:
                raise RuntimeError(
                    f"expected {self.config.num_clients} clients, received {len(summary_results)} summaries"
                )
            if site_order is None:
                site_order = sorted(site_name for site_name, _result in summary_results)

            summaries_by_site = dict(summary_results)
            spatial = torch.stack([summaries_by_site[site]["spatial"] for site in site_order])
            temporal = torch.stack([summaries_by_site[site]["temporal"] for site in site_order])
            artifacts = graph_generator.build(task_id, spatial, temporal)
            relational_graph = artifacts.fused.cpu()
            graphs.append(relational_graph)
            graph_artifacts.append(
                {
                    "fused": relational_graph,
                    "spatial_attention": artifacts.spatial_attention.cpu(),
                    "temporal_attention": artifacts.temporal_attention.cpu(),
                    "temporal_window": artifacts.temporal_window.cpu(),
                }
            )

            for task_round in range(self.config.rounds_per_task):
                encoding_call = self._clients().collect_encodings(
                    task_id,
                    global_model,
                    site_order,
                    relational_graph,
                )
                encoding_results = require_complete_results(encoding_call, site_order)
                discriminator_loss = self._fit_discriminator(
                    discriminator,
                    discriminator_optimizer,
                    encoding_results,
                    device,
                )

                training_call = self._clients().train(
                    task_id,
                    global_model,
                    get_model_state(discriminator),
                    site_order,
                    relational_graph,
                )
                global_model = aggregate_model_results(training_call, global_round, site_order)

                evaluation_call = self._clients().evaluate(
                    task_id,
                    global_model,
                    site_order,
                    graphs,
                )
                evaluation_results = require_complete_results(evaluation_call, site_order)
                metrics = aggregate_evaluation(evaluation_results)
                record = {
                    "global_round": global_round,
                    "task_id": task_id,
                    "task_round": task_round,
                    "discriminator_loss": discriminator_loss,
                    "metrics": metrics,
                }
                history.append(record)
                metric_text = ", ".join(
                    f"task-{metric_task} accuracy={values['accuracy']:.4f}" for metric_task, values in metrics.items()
                )
                print(f"Round {global_round + 1}: discriminator_loss={discriminator_loss:.4f}; " f"{metric_text}")
                global_round += 1

        return {
            "config": self.config.to_dict(),
            "site_order": site_order,
            "model": global_model,
            "discriminator": get_model_state(discriminator),
            "graphs": graph_artifacts,
            "history": history,
        }
