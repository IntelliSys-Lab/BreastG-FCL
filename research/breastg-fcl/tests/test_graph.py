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

from graph import BreastGraphGenerator


class BreastGraphGeneratorTest(unittest.TestCase):
    def test_graph_is_normalized_and_temporal_history_is_windowed(self):
        generator = BreastGraphGenerator(num_clients=3, temporal_window=2)
        spatial = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
        temporal_0 = torch.tensor([[0.0, 0.0], [0.5, 0.0], [0.0, 1.0]])
        first = generator.build(0, spatial, temporal_0)

        torch.testing.assert_close(first.fused.sum(dim=1), torch.ones(3))
        self.assertTrue(torch.all(first.fused >= 0))

        temporal_1 = temporal_0 + 0.1
        second = generator.build(1, spatial, temporal_1)
        self.assertEqual(tuple(second.temporal_window.shape), (3, 4))
        torch.testing.assert_close(second.temporal_window[:, :2], temporal_0)
        torch.testing.assert_close(second.temporal_window[:, 2:], temporal_1)

    def test_invalid_task_order_is_rejected(self):
        generator = BreastGraphGenerator(num_clients=2)
        values = torch.zeros(2, 2)
        with self.assertRaises(ValueError):
            generator.build(1, values, values)


if __name__ == "__main__":
    unittest.main()
