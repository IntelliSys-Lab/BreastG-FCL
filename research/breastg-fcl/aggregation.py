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

"""Result validation and aggregation for blocking Collab group calls."""

from collections.abc import Iterable

from nvflare.app_common.aggregators.weighted_aggregation_helper import WeightedAggregationHelper


def require_complete_results(client_results, expected_sites: Iterable[str] | None = None) -> list[tuple[str, dict]]:
    """Materialize successful results and reject failed or incomplete calls."""

    failures = getattr(client_results, "failures", {})
    if failures:
        details = ", ".join(f"{site}: {error}" for site, error in sorted(failures.items()))
        raise RuntimeError(f"Collab client call failed: {details}")

    results = list(client_results)
    if not results:
        raise RuntimeError("Collab client call returned no successful results")
    site_names = [site_name for site_name, _result in results]
    if len(site_names) != len(set(site_names)):
        raise RuntimeError(f"Collab client call returned duplicate sites: {site_names}")

    if expected_sites is not None:
        expected = set(expected_sites)
        actual = set(site_names)
        if actual != expected:
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            raise RuntimeError(f"Collab client roster mismatch: missing={missing}, unexpected={unexpected}")
    return results


def aggregate_model_results(
    client_results,
    round_number: int,
    expected_sites: Iterable[str],
) -> dict:
    """Aggregate client model states weighted by locally processed examples."""

    results = require_complete_results(client_results, expected_sites)
    helper = WeightedAggregationHelper()
    for site_name, result in results:
        num_examples = int(result["num_examples"])
        if num_examples <= 0:
            raise ValueError(f"{site_name} reported invalid num_examples={num_examples}")
        helper.add(
            data=result["model"],
            weight=num_examples,
            contributor_name=site_name,
            contribution_round=round_number,
        )
    return helper.get_result()


def aggregate_evaluation(results: list[tuple[str, dict]]) -> dict[str, dict[str, float]]:
    """Combine per-site evaluation counts without averaging percentages."""

    totals: dict[str, dict[str, float]] = {}
    for _site_name, site_result in results:
        for task_id, metrics in site_result["tasks"].items():
            task_totals = totals.setdefault(str(task_id), {"loss_sum": 0.0, "correct": 0.0, "total": 0.0})
            task_totals["loss_sum"] += float(metrics["loss_sum"])
            task_totals["correct"] += float(metrics["correct"])
            task_totals["total"] += float(metrics["total"])

    aggregated = {}
    for task_id, task_totals in sorted(totals.items(), key=lambda item: int(item[0])):
        total = task_totals["total"]
        aggregated[task_id] = {
            "loss": task_totals["loss_sum"] / total if total else 0.0,
            "accuracy": task_totals["correct"] / total if total else 0.0,
            "num_examples": int(total),
        }
    return aggregated
