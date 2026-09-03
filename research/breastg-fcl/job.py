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

"""Run the BreastG-FCL Collab workflow in NVFlare simulation mode."""

import argparse

from nvflare.collab import CollabRecipe
from nvflare.recipe import SimEnv

from client import BreastGFCLClient
from config import ExperimentConfig
from server import BreastGFCLServer


def define_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BreastG-FCL NVFlare Collab simulation")
    parser.add_argument("--num-clients", type=int, default=2)
    parser.add_argument("--num-tasks", type=int, default=2)
    parser.add_argument("--rounds-per-task", type=int, default=2)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--discriminator-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--input-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--graph-dim", type=int, default=16)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--train-samples-per-task", type=int, default=96)
    parser.add_argument("--test-samples-per-task", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--discriminator-learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--encoding-noise-scale", type=float, default=0.0)
    parser.add_argument("--summary-noise-scale", type=float, default=0.0)
    parser.add_argument("--privacy-weight", type=float, default=0.05)
    parser.add_argument("--temporal-window", type=int, default=2)
    parser.add_argument("--attention-temperature", type=float, default=1.0)
    parser.add_argument("--max-encoding-batches", type=int, default=2)
    parser.add_argument("--call-timeout", type=int, default=3600)
    parser.add_argument("--max-parallel-calls", type=int, default=0)
    parser.add_argument("--no-replay", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workspace", default="/tmp/nvflare/simulation")
    parser.add_argument("--gpu", default=None, help='NVFlare simulator GPU mapping, for example "[0],[1]"')
    return parser


def make_recipe(config: ExperimentConfig) -> CollabRecipe:
    recipe = CollabRecipe(
        job_name="breastg_fcl_collab",
        server=BreastGFCLServer(config),
        client=BreastGFCLClient(config),
        min_clients=config.num_clients,
    )
    recipe.set_per_site_config(
        {f"site-{site_index + 1}": {"site_index": site_index} for site_index in range(config.num_clients)}
    )
    return recipe


def main():
    args = define_parser().parse_args()
    config = ExperimentConfig(
        num_clients=args.num_clients,
        num_tasks=args.num_tasks,
        rounds_per_task=args.rounds_per_task,
        local_epochs=args.local_epochs,
        discriminator_epochs=args.discriminator_epochs,
        batch_size=args.batch_size,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        graph_dim=args.graph_dim,
        num_classes=args.num_classes,
        train_samples_per_task=args.train_samples_per_task,
        test_samples_per_task=args.test_samples_per_task,
        learning_rate=args.learning_rate,
        discriminator_learning_rate=args.discriminator_learning_rate,
        encoding_noise_scale=args.encoding_noise_scale,
        summary_noise_scale=args.summary_noise_scale,
        privacy_weight=args.privacy_weight,
        temporal_window=args.temporal_window,
        attention_temperature=args.attention_temperature,
        max_encoding_batches=args.max_encoding_batches,
        call_timeout=args.call_timeout,
        max_parallel_calls=args.max_parallel_calls,
        replay_previous_task=not args.no_replay,
        seed=args.seed,
    ).validate()
    site_names = [f"site-{site_index + 1}" for site_index in range(config.num_clients)]
    environment = SimEnv(
        clients=site_names,
        gpu_config=args.gpu,
        workspace_root=args.workspace,
    )
    run = make_recipe(config).execute(environment)
    print("Job Status:", run.get_status())
    print("Results at:", run.get_result())


if __name__ == "__main__":
    main()
