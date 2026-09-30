import csv
import os
from pathlib import Path

import numpy as np


ID_COLUMNS = (
    "case_submitter_id",
    "case_id",
    "sample_submitter_id",
    "sample_id",
    "patient_id",
    "PatientID",
    "SubjectID",
)


def _detect_delimiter(path):
    suffix = Path(path).suffix.lower()
    if suffix == ".tsv":
        return "\t"
    return ","


def _normalize(values):
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("TCIA MRI feature matrix must be 2D")
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return ((arr - mean) / std).astype(np.float32)


def load_tcia_mri_features(path, id_column=None, normalize=True):
    """Load TCIA MRI embeddings/radiomics keyed by TCGA case or sample id.

    Supported formats:
    - CSV/TSV: one id column plus numeric feature columns.
    - NPZ: arrays named `features` and one of `case_ids`, `sample_ids`, `ids`.
    """
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"TCIA MRI feature file not found: {path}")

    suffix = Path(path).suffix.lower()
    if suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        features = data["features"].astype(np.float32)
        ids = None
        for key in ("case_ids", "sample_ids", "ids"):
            if key in data:
                ids = [str(x) for x in data[key].tolist()]
                break
        if ids is None:
            raise ValueError("NPZ TCIA MRI features must include case_ids, sample_ids, or ids")
        if normalize:
            features = _normalize(features)
        return {identifier: features[i] for i, identifier in enumerate(ids)}

    delimiter = _detect_delimiter(path)
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        fieldnames = reader.fieldnames or []
        selected_id_column = id_column
        if selected_id_column is None:
            selected_id_column = next((col for col in ID_COLUMNS if col in fieldnames), None)
        if selected_id_column is None:
            raise ValueError(
                "TCIA MRI feature table needs an id column. "
                f"Tried: {', '.join(ID_COLUMNS)}"
            )

        feature_columns = [col for col in fieldnames if col != selected_id_column]
        for row in reader:
            identifier = row.get(selected_id_column, "").strip()
            if not identifier:
                continue
            values = []
            for col in feature_columns:
                value = row.get(col, "")
                if value == "":
                    continue
                try:
                    values.append(float(value))
                except ValueError:
                    continue
            if values:
                rows.append((identifier, values))

    if not rows:
        raise RuntimeError(f"No numeric TCIA MRI features found in {path}")

    width = min(len(values) for _, values in rows)
    ids = [identifier for identifier, _ in rows]
    features = np.asarray([values[:width] for _, values in rows], dtype=np.float32)
    if normalize:
        features = _normalize(features)
    return {identifier: features[i] for i, identifier in enumerate(ids)}


def aggregate_mri_features(records, indices, feature_by_id, feature_dim=None):
    """Mean-pool MRI features for the samples assigned to one client/task."""
    vectors = []
    matched_ids = []
    for idx in indices:
        record = records[idx]
        raw_candidates = (
            record.get("case_id"),
            record.get("case_submitter_id"),
            record.get("sample_id"),
            record.get("sample_submitter_id"),
            record.get("file_id"),
        )
        candidates = []
        for identifier in raw_candidates:
            if not identifier:
                continue
            identifier = str(identifier)
            candidates.append(identifier)
            if identifier.startswith("TCGA-") and len(identifier) >= 12:
                candidates.append(identifier[:12])

        for identifier in candidates:
            if identifier in feature_by_id:
                vectors.append(feature_by_id[identifier])
                matched_ids.append(identifier)
                break

    if vectors:
        matrix = np.vstack(vectors).astype(np.float32)
        return matrix.mean(axis=0), matched_ids

    if feature_dim is None:
        first = next(iter(feature_by_id.values()))
        feature_dim = int(first.shape[0])
    return np.zeros(feature_dim, dtype=np.float32), matched_ids
