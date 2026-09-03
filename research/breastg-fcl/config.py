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

"""Configuration shared by the BreastG-FCL server and clients."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    """Serializable configuration for the initial Collab implementation."""

    num_clients: int = 2
    num_tasks: int = 2
    rounds_per_task: int = 2
    local_epochs: int = 1
    discriminator_epochs: int = 2
    batch_size: int = 32
    input_dim: int = 64
    hidden_dim: int = 64
    graph_dim: int = 16
    num_classes: int = 2
    train_samples_per_task: int = 96
    test_samples_per_task: int = 48
    learning_rate: float = 1.0e-3
    discriminator_learning_rate: float = 1.0e-3
    privacy_weight: float = 0.05
    encoding_noise_scale: float = 0.0
    summary_noise_scale: float = 0.0
    temporal_window: int = 2
    attention_temperature: float = 1.0
    graph_epsilon: float = 1.0e-8
    replay_previous_task: bool = True
    max_encoding_batches: int = 2
    call_timeout: int = 3600
    max_parallel_calls: int = 0
    seed: int = 0

    def validate(self) -> "ExperimentConfig":
        positive_ints = {
            "num_clients": self.num_clients,
            "num_tasks": self.num_tasks,
            "rounds_per_task": self.rounds_per_task,
            "local_epochs": self.local_epochs,
            "discriminator_epochs": self.discriminator_epochs,
            "batch_size": self.batch_size,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "graph_dim": self.graph_dim,
            "num_classes": self.num_classes,
            "train_samples_per_task": self.train_samples_per_task,
            "test_samples_per_task": self.test_samples_per_task,
            "temporal_window": self.temporal_window,
            "max_encoding_batches": self.max_encoding_batches,
            "call_timeout": self.call_timeout,
        }
        invalid = [name for name, value in positive_ints.items() if value <= 0]
        if invalid:
            raise ValueError(f"Configuration values must be positive: {', '.join(invalid)}")
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if self.learning_rate <= 0 or self.discriminator_learning_rate <= 0:
            raise ValueError("learning rates must be positive")
        if self.privacy_weight < 0:
            raise ValueError("privacy_weight cannot be negative")
        if self.encoding_noise_scale < 0 or self.summary_noise_scale < 0:
            raise ValueError("noise scales cannot be negative")
        if self.attention_temperature <= 0:
            raise ValueError("attention_temperature must be positive")
        if self.graph_epsilon < 0:
            raise ValueError("graph_epsilon cannot be negative")
        if self.max_parallel_calls < 0:
            raise ValueError("max_parallel_calls cannot be negative")
        return self

    def to_dict(self) -> dict:
        return asdict(self)
