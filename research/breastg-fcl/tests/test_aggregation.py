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

from aggregation import aggregate_evaluation, aggregate_model_results, require_complete_results


class FakeResults(list):
    def __init__(self, values, failures=None):
        super().__init__(values)
        self.failures = failures or {}


class AggregationTest(unittest.TestCase):
    def test_model_aggregation_uses_example_counts(self):
        results = FakeResults(
            [
                ("site-1", {"model": {"weight": torch.tensor([1.0])}, "num_examples": 1}),
                ("site-2", {"model": {"weight": torch.tensor([3.0])}, "num_examples": 3}),
            ]
        )
        aggregated = aggregate_model_results(results, round_number=0, expected_sites=["site-1", "site-2"])
        torch.testing.assert_close(aggregated["weight"], torch.tensor([2.5]))

    def test_failures_are_not_silently_aggregated(self):
        results = FakeResults([("site-1", {})], failures={"site-2": RuntimeError("failed")})
        with self.assertRaises(RuntimeError):
            require_complete_results(results)

    def test_evaluation_aggregates_counts(self):
        results = [
            ("site-1", {"tasks": {"0": {"loss_sum": 2.0, "correct": 1, "total": 2}}}),
            ("site-2", {"tasks": {"0": {"loss_sum": 2.0, "correct": 3, "total": 4}}}),
        ]
        aggregated = aggregate_evaluation(results)
        self.assertAlmostEqual(aggregated["0"]["loss"], 4.0 / 6.0)
        self.assertAlmostEqual(aggregated["0"]["accuracy"], 4.0 / 6.0)
        self.assertEqual(aggregated["0"]["num_examples"], 6)


if __name__ == "__main__":
    unittest.main()
