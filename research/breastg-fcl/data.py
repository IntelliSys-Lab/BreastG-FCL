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

"""Site-local synthetic data for a runnable BreastG-FCL smoke experiment."""

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, TensorDataset

from config import ExperimentConfig


@dataclass
class TaskData:
    train: TensorDataset
    test: TensorDataset
    spatial_summary: torch.Tensor
    temporal_summary: torch.Tensor


class SyntheticSiteData:
    """Deterministic non-IID task data owned by one simulated site."""

    def __init__(self, config: ExperimentConfig, site_index: int):
        if not 0 <= site_index < config.num_clients:
            raise ValueError(f"site_index must be in [0, {config.num_clients}), got {site_index}")
        self.config = config
        self.site_index = site_index
        self.tasks = [self._make_task(task_id) for task_id in range(config.num_tasks)]

    def _make_task(self, task_id: int) -> TaskData:
        config = self.config
        generator = torch.Generator().manual_seed(config.seed + 10_000 * task_id + 101 * self.site_index)
        weight_generator = torch.Generator().manual_seed(config.seed + 1_003 * task_id + 17)
        class_weights = torch.randn(
            config.num_classes,
            config.input_dim,
            generator=weight_generator,
        )

        train_x, train_y = self._make_split(
            config.train_samples_per_task,
            task_id,
            class_weights,
            generator,
        )
        test_x, test_y = self._make_split(
            config.test_samples_per_task,
            task_id,
            class_weights,
            generator,
        )

        summary_width = min(8, config.input_dim)
        spatial_summary = torch.cat(
            [
                train_x[:, :summary_width].mean(dim=0),
                train_x[:, :summary_width].std(dim=0),
            ]
        )
        class_histogram = torch.bincount(train_y, minlength=config.num_classes).float()
        class_histogram /= class_histogram.sum().clamp_min(1.0)
        temporal_summary = torch.cat(
            [
                train_x[:, -summary_width:].mean(dim=0),
                class_histogram,
            ]
        )

        return TaskData(
            train=TensorDataset(train_x, train_y),
            test=TensorDataset(test_x, test_y),
            spatial_summary=spatial_summary,
            temporal_summary=temporal_summary,
        )

    def _make_split(
        self,
        num_samples: int,
        task_id: int,
        class_weights: torch.Tensor,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        config = self.config
        site_shift = (self.site_index - (config.num_clients - 1) / 2.0) * 0.35
        task_shift = task_id * 0.20
        inputs = torch.randn(num_samples, config.input_dim, generator=generator)
        inputs[:, : min(8, config.input_dim)] += site_shift
        inputs[:, -min(8, config.input_dim) :] += task_shift
        logits = inputs @ class_weights.t()
        logits[:, self.site_index % config.num_classes] += 0.25
        logits += 0.15 * torch.randn(logits.shape, generator=generator)
        labels = logits.argmax(dim=1)
        return inputs.float(), labels.long()

    def loader(self, task_id: int, train: bool, shuffle: bool | None = None) -> DataLoader:
        task = self.tasks[task_id]
        dataset = task.train if train else task.test
        if shuffle is None:
            shuffle = train
        generator = torch.Generator().manual_seed(
            self.config.seed + 100_000 + self.site_index * 1_000 + task_id * 10 + int(train)
        )
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            generator=generator,
        )

    def graph_summary(self, task_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        task = self.tasks[task_id]
        return task.spatial_summary.clone(), task.temporal_summary.clone()
