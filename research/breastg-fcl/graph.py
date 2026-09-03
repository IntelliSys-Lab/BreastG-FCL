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

"""Disease-aware spatial-temporal relationship graph construction."""

from dataclasses import dataclass

import torch


@dataclass
class GraphArtifacts:
    fused: torch.Tensor
    spatial_attention: torch.Tensor
    temporal_attention: torch.Tensor
    temporal_window: torch.Tensor


class BreastGraphGenerator:
    """Build a row-normalized graph from client spatial and temporal summaries."""

    def __init__(
        self,
        num_clients: int,
        temporal_window: int = 2,
        temperature: float = 1.0,
        epsilon: float = 1.0e-8,
    ):
        if num_clients <= 0:
            raise ValueError("num_clients must be positive")
        if temporal_window <= 0:
            raise ValueError("temporal_window must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if epsilon < 0:
            raise ValueError("epsilon cannot be negative")
        self.num_clients = num_clients
        self.temporal_window = temporal_window
        self.temperature = temperature
        self.epsilon = epsilon
        self.temporal_history: list[torch.Tensor] = []

    def _validate(self, values: torch.Tensor, name: str) -> torch.Tensor:
        matrix = torch.as_tensor(values, dtype=torch.float32, device="cpu")
        if matrix.ndim != 2 or matrix.shape[0] != self.num_clients:
            raise ValueError(f"{name} must have shape [{self.num_clients}, feature_dim], got {tuple(matrix.shape)}")
        if matrix.shape[1] == 0:
            raise ValueError(f"{name} must contain at least one feature")
        if not torch.isfinite(matrix).all():
            raise ValueError(f"{name} contains non-finite values")
        return matrix

    def _attention(self, values: torch.Tensor, name: str) -> torch.Tensor:
        matrix = self._validate(values, name)
        standardized = (matrix - matrix.mean(dim=0, keepdim=True)) / matrix.std(
            dim=0,
            keepdim=True,
            correction=0,
        ).clamp_min(1.0e-6)
        differences = standardized[:, None, :] - standardized[None, :, :]
        scores = -differences.square().mean(dim=-1) / self.temperature
        return torch.softmax(scores, dim=1)

    def build(
        self,
        task_id: int,
        spatial_summaries: torch.Tensor,
        temporal_summaries: torch.Tensor,
    ) -> GraphArtifacts:
        spatial = self._validate(spatial_summaries, "spatial summaries")
        temporal = self._validate(temporal_summaries, "temporal summaries")
        if task_id == len(self.temporal_history):
            self.temporal_history.append(temporal)
        elif 0 <= task_id < len(self.temporal_history):
            self.temporal_history[task_id] = temporal
        else:
            raise ValueError(
                "Graph tasks must be generated sequentially: "
                f"received task {task_id} with {len(self.temporal_history)} stored"
            )

        start = max(0, task_id - self.temporal_window + 1)
        temporal_window = torch.cat(self.temporal_history[start : task_id + 1], dim=1)
        spatial_attention = self._attention(spatial, "spatial summaries")
        temporal_attention = self._attention(temporal_window, "windowed temporal summaries")
        product = spatial_attention * temporal_attention
        fused = product / (product.sum(dim=1, keepdim=True) + self.epsilon)
        return GraphArtifacts(
            fused=fused.float(),
            spatial_attention=spatial_attention.float(),
            temporal_attention=temporal_attention.float(),
            temporal_window=temporal_window.float(),
        )
