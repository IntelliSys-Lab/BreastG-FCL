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

"""PyTorch modules used by the Collab BreastG-FCL workflow."""

from collections.abc import Mapping

import torch
import torch.nn as nn

from config import ExperimentConfig


def get_model_state(module: nn.Module) -> dict[str, torch.Tensor]:
    """Return a detached CPU state dictionary safe for cross-site transport."""

    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def load_model_state(module: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    """Load transported CPU tensors into a module with strict key checking."""

    module.load_state_dict(dict(state), strict=True)


class GraphEncoder(nn.Module):
    def __init__(self, num_clients: int, hidden_dim: int, graph_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(num_clients, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, graph_dim),
        )

    def forward(self, graph_row: torch.Tensor) -> torch.Tensor:
        if graph_row.ndim == 1:
            graph_row = graph_row.unsqueeze(0)
        return self.network(graph_row.float())


class BreastGFCLModel(nn.Module):
    """Graph-conditioned feature encoder and task predictor."""

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.graph_encoder = GraphEncoder(config.num_clients, config.hidden_dim, config.graph_dim)
        self.feature_encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim * 2),
            nn.LayerNorm(config.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_dim + config.graph_dim, config.hidden_dim),
            nn.ReLU(),
        )
        self.predictor = nn.Linear(config.hidden_dim, config.num_classes)

    def encode(self, inputs: torch.Tensor, graph_row: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_encoder(inputs.float())
        graph_embedding = self.graph_encoder(graph_row)
        if graph_embedding.shape[0] == 1 and features.shape[0] > 1:
            graph_embedding = graph_embedding.expand(features.shape[0], -1)
        if graph_embedding.shape[0] != features.shape[0]:
            raise ValueError(
                "graph embedding batch does not match input batch: "
                f"{graph_embedding.shape[0]} != {features.shape[0]}"
            )
        latent = self.fusion(torch.cat([features, graph_embedding], dim=1))
        return latent, graph_embedding

    def forward(self, inputs: torch.Tensor, graph_row: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent, graph_embedding = self.encode(inputs, graph_row)
        return self.predictor(latent), latent, graph_embedding


class GraphDiscriminator(nn.Module):
    """Predict a client's graph embedding from its encoded samples."""

    def __init__(self, hidden_dim: int, graph_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, graph_dim),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.network(latent.float())


def make_model(config: ExperimentConfig, device: torch.device) -> BreastGFCLModel:
    return BreastGFCLModel(config).to(device)


def make_discriminator(config: ExperimentConfig, device: torch.device) -> GraphDiscriminator:
    return GraphDiscriminator(config.hidden_dim, config.graph_dim).to(device)


def resolve_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
