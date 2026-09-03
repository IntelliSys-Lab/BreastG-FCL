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

import unittest

import torch

from config import ExperimentConfig
from data import SyntheticSiteData
from model import BreastGFCLModel, GraphDiscriminator


class ModelAndDataTest(unittest.TestCase):
    def test_synthetic_data_is_deterministic_and_site_specific(self):
        config = ExperimentConfig(num_clients=2, num_tasks=1, input_dim=12)
        first = SyntheticSiteData(config, site_index=0)
        repeated = SyntheticSiteData(config, site_index=0)
        other = SyntheticSiteData(config, site_index=1)

        torch.testing.assert_close(first.tasks[0].train.tensors[0], repeated.tasks[0].train.tensors[0])
        self.assertFalse(torch.equal(first.tasks[0].train.tensors[0], other.tasks[0].train.tensors[0]))

    def test_model_and_discriminator_shapes(self):
        config = ExperimentConfig(num_clients=2, input_dim=12, hidden_dim=16, graph_dim=4)
        model = BreastGFCLModel(config)
        discriminator = GraphDiscriminator(config.hidden_dim, config.graph_dim)
        inputs = torch.randn(5, config.input_dim)
        graph_row = torch.tensor([0.7, 0.3])

        logits, latent, graph_embedding = model(inputs, graph_row)
        self.assertEqual(tuple(logits.shape), (5, config.num_classes))
        self.assertEqual(tuple(latent.shape), (5, config.hidden_dim))
        self.assertEqual(tuple(graph_embedding.shape), (5, config.graph_dim))
        self.assertEqual(tuple(discriminator(latent).shape), (5, config.graph_dim))


if __name__ == "__main__":
    unittest.main()
