import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np


TEST_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from model.modules import BreastGraphGenerator


def make_opt(**overrides):
    values = {
        "num_clients": 3,
        "temporal_window": 2,
        "attention_temperature": 1.0,
        "graph_epsilon": 1e-8,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class BreastGraphGeneratorTest(unittest.TestCase):
    def test_attention_is_row_normalized_and_deterministic(self):
        generator = BreastGraphGenerator(make_opt())
        summaries = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]],
            dtype=np.float32,
        )

        first = generator._attention(summaries, "test summaries")
        second = generator._attention(summaries, "test summaries")

        np.testing.assert_allclose(first, second, atol=0.0)
        np.testing.assert_allclose(first.sum(axis=1), np.ones(3), atol=1e-6)
        self.assertTrue(np.all(first >= 0.0))

    def test_temporal_window_and_multiplicative_fusion_follow_paper(self):
        opt = make_opt()
        generator = BreastGraphGenerator(opt)
        spatial = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]],
            dtype=np.float32,
        )
        temporal_0 = np.asarray(
            [[0.0, 0.0], [0.5, 0.0], [0.0, 1.0]],
            dtype=np.float32,
        )

        graph = generator.learn(
            0,
            task_id=0,
            spatial_features=spatial,
            temporal_features=temporal_0,
        )
        product = (
            generator.last_spatial_attention
            * generator.last_temporal_patterns
        )
        expected = product / (
            product.sum(axis=1, keepdims=True) + opt.graph_epsilon
        )
        np.testing.assert_allclose(graph, expected, atol=1e-7)

        temporal_1 = temporal_0 + np.asarray(
            [[0.2, 0.0], [0.0, 0.1], [0.1, 0.2]],
            dtype=np.float32,
        )
        generator.learn(
            0,
            task_id=1,
            spatial_features=spatial,
            temporal_features=temporal_1,
        )
        self.assertEqual(generator.last_temporal_window.shape, (3, 4))
        np.testing.assert_allclose(
            generator.last_temporal_window[:, :2],
            temporal_0,
        )
        np.testing.assert_allclose(
            generator.last_temporal_window[:, 2:],
            temporal_1,
        )

    def test_invalid_or_missing_summaries_fail_closed(self):
        generator = BreastGraphGenerator(make_opt())
        with self.assertRaises(RuntimeError):
            generator.learn(0, task_id=0, spatial_features=None, temporal_features=None)
        with self.assertRaises(ValueError):
            generator.learn(
                0,
                task_id=0,
                spatial_features=np.zeros((2, 3), dtype=np.float32),
                temporal_features=np.zeros((3, 2), dtype=np.float32),
            )


if __name__ == "__main__":
    unittest.main()
