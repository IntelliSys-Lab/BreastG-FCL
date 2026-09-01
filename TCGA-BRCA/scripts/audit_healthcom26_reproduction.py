#!/usr/bin/env python3
"""Audit whether source-level or exact-author HealthCom'26 reproduction is ready."""

import argparse
import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = PROJECT_DIR / "configs" / "healthcom26_reproduction_protocol.json"
DEFAULT_MANIFEST = (
    PROJECT_DIR
    / "data"
    / "tcia_official_radiogenomics"
    / "official_source_manifest.json"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--require-exact",
        action="store_true",
        help="Exit with status 2 while any exact-author blocker remains.",
    )
    return parser.parse_args()


def audit(protocol_path, source_manifest_path):
    protocol = json.loads(protocol_path.read_text())
    errors = []
    if not source_manifest_path.exists():
        errors.append(f"missing source manifest: {source_manifest_path}")
    else:
        manifest = json.loads(source_manifest_path.read_text())
        derived = manifest.get("derived_tables", {})
        expected = protocol["source_reproducible_data_protocol"]
        checks = {
            "official_radiomics_rows": expected["official_radiomics_rows"],
            "matched_cohort_rows": expected[
                "matched_radiomics_pam50_clinical_rows"
            ],
            "spatial_feature_dim": expected["spatial_feature_dimension"],
            "temporal_feature_dim": expected["temporal_feature_dimension"],
        }
        for field, expected_value in checks.items():
            if derived.get(field) != expected_value:
                errors.append(
                    f"manifest {field}: expected {expected_value}, "
                    f"found {derived.get(field)}"
                )
        artifacts = manifest.get("artifacts", {})
        if len(artifacts) != 5:
            errors.append(f"expected 5 source artifacts, found {len(artifacts)}")
        for name, artifact in artifacts.items():
            if len(str(artifact.get("sha256", ""))) != 64:
                errors.append(f"artifact {name} has no valid SHA-256 digest")

    blockers = protocol.get("exact_author_experiment_blockers", [])
    return {
        "source_data_reproduction_ready": not errors,
        "source_data_errors": errors,
        "exact_author_experiment_ready": not errors and not blockers,
        "exact_author_blocker_count": len(blockers),
        "exact_author_blockers": blockers,
        "protocol_status": protocol.get("status"),
        "source_manifest": str(source_manifest_path),
    }


def main():
    args = parse_args()
    report = audit(args.protocol, args.source_manifest)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
        print(f"Wrote audit report: {args.output}")
    print(payload, end="")
    if not report["source_data_reproduction_ready"]:
        raise SystemExit(1)
    if args.require_exact and not report["exact_author_experiment_ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
