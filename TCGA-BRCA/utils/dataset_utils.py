import csv
import os
import pickle
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from utils.tcia_mri_utils import aggregate_mri_features, load_tcia_mri_features


LABEL_MAP = {
    "Normal": 0,
    "Tumor": 1,
}


CLINICAL_STAGE_TASKS = [
    {
        "id": 0,
        "name": "early_stage",
        "description": "AJCC Stage 0/I breast cancer tumors vs normal controls",
        "stage_prefixes": ("stage 0", "stage i"),
    },
    {
        "id": 1,
        "name": "intermediate_stage",
        "description": "AJCC Stage II breast cancer tumors vs normal controls",
        "stage_prefixes": ("stage ii",),
    },
    {
        "id": 2,
        "name": "advanced_stage",
        "description": "AJCC Stage III/IV or metastatic breast cancer tumors vs normal controls",
        "stage_prefixes": ("stage iii", "stage iv"),
    },
]


def _load_clinical_cases(path):
    clinical = {}
    if not path or not os.path.exists(path):
        return clinical
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            case_submitter_id = row.get("case_submitter_id", "")
            if case_submitter_id:
                clinical[case_submitter_id] = row
    return clinical


def _normalize_stage(stage):
    return " ".join(str(stage or "").strip().lower().split())


def _clinical_stage_task_id(stage, sample_type=""):
    sample_type = str(sample_type or "")
    normalized = _normalize_stage(stage)
    if sample_type == "Metastatic" or normalized.startswith("stage iv"):
        return 2
    if normalized.startswith("stage iii"):
        return 2
    if normalized.startswith("stage ii"):
        return 1
    if normalized.startswith("stage 0") or normalized.startswith("stage i"):
        return 0
    return None


def _split_index_list(indices, train_split, rng):
    shuffled = list(indices)
    rng.shuffle(shuffled)
    if len(shuffled) > 1:
        split = int(round(len(shuffled) * train_split))
        split = min(max(split, 1), len(shuffled) - 1)
    else:
        split = len(shuffled)
    train = shuffled[:split]
    test = shuffled[split:] or shuffled[:1]
    return train, test


def _distribute_task_label_indices(task_label_indices, opt, seed):
    rng = np.random.default_rng(seed)
    client_task_indices = defaultdict(dict)
    all_indices = [idx for by_label in task_label_indices.values() for values in by_label.values() for idx in values]

    for client_id in range(opt.num_clients):
        for task_id in range(opt.num_task):
            client_task_indices[client_id][task_id] = []

    for task_id in range(opt.num_task):
        for indices in task_label_indices.get(task_id, {}).values():
            label_buckets = _assign_to_buckets(indices, opt.num_clients, rng)
            for client_id, label_indices in enumerate(label_buckets):
                client_task_indices[client_id][task_id].extend(label_indices)

        task_pool = [idx for values in task_label_indices.get(task_id, {}).values() for idx in values]
        for client_id in range(opt.num_clients):
            bucket = client_task_indices[client_id][task_id]
            if not bucket and task_pool:
                bucket.append(int(rng.choice(task_pool)))
            elif not bucket and all_indices:
                bucket.append(int(rng.choice(all_indices)))
            rng.shuffle(bucket)

    return client_task_indices


def _task_label_summary(indices_by_task_label):
    summary = {}
    for task_id, by_label in indices_by_task_label.items():
        summary[str(int(task_id))] = {
            label_name: int(len(by_label.get(label_id, [])))
            for label_name, label_id in LABEL_MAP.items()
        }
    return summary


def _make_clinical_stage_task_indices(records, targets, opt):
    if opt.num_task != len(CLINICAL_STAGE_TASKS):
        raise ValueError(
            "clinical_stage task split expects --num-task 3: "
            "early_stage, intermediate_stage, advanced_stage"
        )

    clinical = _load_clinical_cases(getattr(opt, "clinical_path", None))
    if not clinical:
        raise RuntimeError(
            "clinical_stage task split requires GDC clinical metadata. "
            "Provide --clinical-path pointing to gdc_clinical_cases.tsv."
        )

    normal_indices = []
    tumor_by_task = {task["id"]: [] for task in CLINICAL_STAGE_TASKS}
    excluded_unknown_stage = []

    for idx, record in enumerate(records):
        label = int(targets[idx])
        if label == LABEL_MAP["Normal"]:
            normal_indices.append(idx)
            continue

        case_submitter_id = record.get("case_submitter_id", "")
        clinical_row = clinical.get(case_submitter_id, {})
        stage = clinical_row.get("ajcc_pathologic_stage", "")
        task_id = _clinical_stage_task_id(stage, record.get("sample_type", ""))
        if task_id is None:
            if getattr(opt, "include_unknown_stage", False):
                task_id = len(CLINICAL_STAGE_TASKS) - 1
            else:
                excluded_unknown_stage.append(idx)
                continue
        tumor_by_task[task_id].append(idx)
        record["ajcc_pathologic_stage"] = stage
        record["clinical_task_id"] = task_id
        record["clinical_task_name"] = CLINICAL_STAGE_TASKS[task_id]["name"]

    rng = np.random.default_rng(opt.seed)
    normal_buckets = _assign_to_buckets(normal_indices, opt.num_task, rng)

    train_by_task_label = defaultdict(lambda: defaultdict(list))
    test_by_task_label = defaultdict(lambda: defaultdict(list))

    for task in CLINICAL_STAGE_TASKS:
        task_id = task["id"]
        task_normals = normal_buckets[task_id]
        normal_train, normal_test = _split_index_list(task_normals, opt.train_split, rng)
        tumor_train, tumor_test = _split_index_list(tumor_by_task[task_id], opt.train_split, rng)
        train_by_task_label[task_id][LABEL_MAP["Normal"]] = normal_train
        train_by_task_label[task_id][LABEL_MAP["Tumor"]] = tumor_train
        test_by_task_label[task_id][LABEL_MAP["Normal"]] = normal_test
        test_by_task_label[task_id][LABEL_MAP["Tumor"]] = tumor_test

    train_indices = _distribute_task_label_indices(train_by_task_label, opt, opt.seed)
    test_indices = _distribute_task_label_indices(test_by_task_label, opt, opt.seed + 1)

    opt.task_metadata = [
        {
            "task_id": task["id"],
            "task_name": task["name"],
            "description": task["description"],
            "stage_prefixes": list(task["stage_prefixes"]),
        }
        for task in CLINICAL_STAGE_TASKS
    ]
    opt.task_label_counts = _task_label_summary(train_by_task_label)
    opt.task_test_label_counts = _task_label_summary(test_by_task_label)
    opt.excluded_unknown_stage_count = int(len(excluded_unknown_stage))
    return train_indices, test_indices


def _make_random_task_indices(targets, opt):
    train_by_label, test_by_label = _split_indices_by_label(
        targets,
        opt.train_split,
        opt.seed,
    )
    train_indices = _make_client_task_indices(train_by_label, opt, opt.seed)
    test_indices = _make_client_task_indices(test_by_label, opt, opt.seed + 1)
    opt.task_metadata = [
        {
            "task_id": task_id,
            "task_name": f"random_task_{task_id}",
            "description": "Random stratified TCGA-BRCA Normal vs Tumor split",
        }
        for task_id in range(opt.num_task)
    ]
    return train_indices, test_indices


class TCGABRCADataset(Dataset):
    """TCGA-BRCA RNA-seq expression dataset backed by an in-memory matrix."""

    def __init__(self, features, targets, sample_ids=None):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.long)
        self.sample_ids = sample_ids or [str(i) for i in range(len(self.targets))]

    def __len__(self):
        return int(self.targets.shape[0])

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def _read_manifest(manifest_path, raw_dir):
    records = []
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            tissue_type = row.get("tissue_type", "")
            if tissue_type not in LABEL_MAP:
                continue

            file_id = row["file_id"]
            file_name = row["file_name"]
            file_path = os.path.join(raw_dir, file_id, file_name)
            if not os.path.exists(file_path):
                continue

            records.append(
                {
                    "file_id": file_id,
                    "file_name": file_name,
                    "file_path": file_path,
                    "sample_id": row.get("sample_submitter_id", file_id),
                    "sample_submitter_id": row.get("sample_submitter_id", file_id),
                    "case_id": row.get("case_submitter_id", ""),
                    "case_submitter_id": row.get("case_submitter_id", ""),
                    "sample_type": row.get("sample_type", ""),
                    "label_name": tissue_type,
                    "label": LABEL_MAP[tissue_type],
                }
            )
    if not records:
        raise RuntimeError(
            f"No usable TCGA-BRCA expression files found from manifest: {manifest_path}"
        )
    return records


def _read_expression_vector(path, value_column, max_genes):
    values = []
    gene_ids = []
    with open(path, newline="") as f:
        reader = csv.DictReader(
            (line for line in f if not line.startswith("#")),
            delimiter="\t",
        )
        if value_column not in reader.fieldnames:
            raise ValueError(f"{value_column} not found in {path}")

        for row in reader:
            if row.get("gene_type") != "protein_coding":
                continue
            raw_value = row.get(value_column, "")
            value = 0.0 if raw_value == "" else float(raw_value)
            gene_ids.append(row.get("gene_id", ""))
            values.append(value)
            if len(values) >= max_genes:
                break

    if len(values) < max_genes:
        values.extend([0.0] * (max_genes - len(values)))
        gene_ids.extend([""] * (max_genes - len(gene_ids)))

    return np.asarray(values, dtype=np.float32), gene_ids


def _build_expression_cache(opt, records):
    cache_dir = os.path.join(opt.data_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    cache_name = (
        f"tcga_brca_{opt.expression_value_col}_{opt.max_genes}_"
        f"{len(records)}samples.npz"
    )
    cache_path = os.path.join(cache_dir, cache_name)

    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        return (
            data["features"].astype(np.float32),
            data["targets"].astype(np.int64),
            data["sample_ids"].tolist(),
            data["gene_ids"].tolist(),
        )

    feature_rows = []
    targets = []
    sample_ids = []
    gene_ids = None

    for record in records:
        vector, current_gene_ids = _read_expression_vector(
            record["file_path"],
            opt.expression_value_col,
            opt.max_genes,
        )
        if gene_ids is None:
            gene_ids = current_gene_ids
        feature_rows.append(np.log1p(vector))
        targets.append(record["label"])
        sample_ids.append(record["sample_id"])

    features = np.vstack(feature_rows).astype(np.float32)
    targets = np.asarray(targets, dtype=np.int64)

    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    features = ((features - mean) / std).astype(np.float32)

    np.savez_compressed(
        cache_path,
        features=features,
        targets=targets,
        sample_ids=np.asarray(sample_ids, dtype=object),
        gene_ids=np.asarray(gene_ids or [], dtype=object),
    )

    return features, targets, sample_ids, gene_ids or []


def _split_indices_by_label(targets, train_split, seed):
    rng = np.random.default_rng(seed)
    train_by_label = {}
    test_by_label = {}

    for label in sorted(set(targets.tolist())):
        indices = np.where(targets == label)[0]
        rng.shuffle(indices)
        if len(indices) > 1:
            split = int(round(len(indices) * train_split))
            split = min(max(split, 1), len(indices) - 1)
        else:
            split = len(indices)
        train_by_label[label] = indices[:split].tolist()
        test_by_label[label] = indices[split:].tolist() or indices[:1].tolist()

    return train_by_label, test_by_label


def _assign_to_buckets(indices, num_buckets, rng):
    shuffled = list(indices)
    rng.shuffle(shuffled)
    buckets = [[] for _ in range(num_buckets)]
    for i, idx in enumerate(shuffled):
        buckets[i % num_buckets].append(idx)
    return buckets


def _make_client_task_indices(indices_by_label, opt, seed):
    rng = np.random.default_rng(seed)
    num_buckets = opt.num_clients * opt.num_task
    buckets = [[] for _ in range(num_buckets)]

    for indices in indices_by_label.values():
        label_buckets = _assign_to_buckets(indices, num_buckets, rng)
        for bucket_id, label_indices in enumerate(label_buckets):
            buckets[bucket_id].extend(label_indices)

    all_indices = [idx for indices in indices_by_label.values() for idx in indices]
    for bucket in buckets:
        if not bucket and all_indices:
            bucket.append(int(rng.choice(all_indices)))

    client_task_indices = defaultdict(dict)
    for client_id in range(opt.num_clients):
        for task_id in range(opt.num_task):
            bucket_id = client_id * opt.num_task + task_id
            rng.shuffle(buckets[bucket_id])
            client_task_indices[client_id][task_id] = buckets[bucket_id]

    return client_task_indices


def setup_tcga_brca_loaders(opt):
    """Create TCGA-BRCA federated loaders from GDC STAR gene-count files."""
    os.makedirs(opt.output_dir, exist_ok=True)

    manifest_path = getattr(
        opt,
        "manifest_path",
        os.path.join(opt.data_dir, "..", "metadata", "gdc_files_manifest.tsv"),
    )
    raw_dir = getattr(opt, "raw_dir", os.path.join(opt.data_dir, "raw"))

    records = _read_manifest(manifest_path, raw_dir)
    features, targets, sample_ids, gene_ids = _build_expression_cache(opt, records)

    opt.input_dim = int(features.shape[1])
    opt.num_classes = int(len(set(targets.tolist())))
    opt.nc = opt.num_classes

    dataset = TCGABRCADataset(features, targets, sample_ids)
    if getattr(opt, "task_split_strategy", "clinical_stage") == "clinical_stage":
        train_indices, test_indices = _make_clinical_stage_task_indices(records, targets, opt)
    else:
        train_indices, test_indices = _make_random_task_indices(targets, opt)

    feature_by_id = None
    opt.client_spatial_features = None
    opt.client_spatial_feature_matches = None
    spatial_path = getattr(opt, "tcia_mri_features_path", None)
    if not spatial_path or not os.path.exists(spatial_path):
        raise RuntimeError(
            "BreastG-FCL requires TCIA spatial features; "
            "provide --tcia-mri-features-path"
        )
    feature_by_id = load_tcia_mri_features(
        spatial_path,
        id_column=getattr(opt, "tcia_mri_id_column", None),
        normalize=True,
    )

    if feature_by_id:
        feature_dim = int(next(iter(feature_by_id.values())).shape[0])
        spatial_by_task = []
        match_counts_by_task = []
        for task_id in range(opt.num_task):
            task_vectors = []
            task_match_counts = []
            for client_id in range(opt.num_clients):
                vector, matched_ids = aggregate_mri_features(
                    records,
                    train_indices[client_id][task_id],
                    feature_by_id,
                    feature_dim=feature_dim,
                )
                task_vectors.append(vector)
                task_match_counts.append(len(matched_ids))
            spatial_by_task.append(np.vstack(task_vectors).astype(np.float32))
            match_counts_by_task.append(task_match_counts)
        opt.client_spatial_features = spatial_by_task
        opt.client_spatial_feature_matches = match_counts_by_task

    temporal_feature_by_id = None
    opt.client_temporal_features = None
    opt.client_temporal_feature_matches = None
    kinetics_path = getattr(opt, "tcia_dce_kinetics_path", None)
    if not kinetics_path or not os.path.exists(kinetics_path):
        raise RuntimeError(
            "BreastG-FCL requires DCE temporal features; "
            "provide --tcia-dce-kinetics-path"
        )
    temporal_feature_by_id = load_tcia_mri_features(
        kinetics_path,
        id_column=getattr(opt, "tcia_dce_id_column", None),
        normalize=True,
    )

    if temporal_feature_by_id:
        temporal_dim = int(next(iter(temporal_feature_by_id.values())).shape[0])
        temporal_by_task = []
        temporal_match_counts_by_task = []
        for task_id in range(opt.num_task):
            task_vectors = []
            task_match_counts = []
            for client_id in range(opt.num_clients):
                vector, matched_ids = aggregate_mri_features(
                    records,
                    train_indices[client_id][task_id],
                    temporal_feature_by_id,
                    feature_dim=temporal_dim,
                )
                task_vectors.append(vector)
                task_match_counts.append(len(matched_ids))
            temporal_by_task.append(np.vstack(task_vectors).astype(np.float32))
            temporal_match_counts_by_task.append(task_match_counts)
        opt.client_temporal_features = temporal_by_task
        opt.client_temporal_feature_matches = temporal_match_counts_by_task

    client_loaders = defaultdict(dict)
    for client_id in range(opt.num_clients):
        for task_id in range(opt.num_task):
            client_loaders[client_id][task_id] = {
                "train": DataLoader(
                    Subset(dataset, train_indices[client_id][task_id]),
                    batch_size=opt.batch_size,
                    shuffle=opt.shuffle,
                    num_workers=opt.num_workers,
                    pin_memory=opt.pin_memory,
                ),
                "test": DataLoader(
                    Subset(dataset, test_indices[client_id][task_id]),
                    batch_size=opt.batch_size,
                    shuffle=False,
                    num_workers=opt.num_workers,
                    pin_memory=opt.pin_memory,
                ),
            }

    label_counts = {
        label_name: int((targets == label_id).sum())
        for label_name, label_id in LABEL_MAP.items()
    }
    partitioning = {
        "dataset": "TCGA-BRCA",
        "label_map": LABEL_MAP,
        "label_counts": label_counts,
        "num_clients": opt.num_clients,
        "tasks_per_client": opt.num_task,
        "task_split_strategy": getattr(opt, "task_split_strategy", None),
        "task_metadata": getattr(opt, "task_metadata", None),
        "task_label_counts": getattr(opt, "task_label_counts", None),
        "task_test_label_counts": getattr(opt, "task_test_label_counts", None),
        "excluded_unknown_stage_count": getattr(opt, "excluded_unknown_stage_count", 0),
        "train_split": opt.train_split,
        "input_dim": opt.input_dim,
        "expression_value_col": opt.expression_value_col,
        "max_genes": opt.max_genes,
        "gene_ids": gene_ids,
        "spatial_attention_source": getattr(opt, "spatial_attention_source", None),
        "tcia_mri_features_path": getattr(opt, "tcia_mri_features_path", None),
        "client_spatial_feature_matches": getattr(opt, "client_spatial_feature_matches", None),
        "temporal_attention_source": getattr(opt, "temporal_attention_source", None),
        "tcia_dce_kinetics_path": getattr(opt, "tcia_dce_kinetics_path", None),
        "client_temporal_feature_matches": getattr(opt, "client_temporal_feature_matches", None),
        "client_task_train_indices": dict(train_indices),
        "client_task_test_indices": dict(test_indices),
        "seed": opt.seed,
    }

    partition_path = os.path.join(
        opt.output_dir,
        f"tcga_brca_partitioning_seed{opt.seed}.pkl",
    )
    with open(partition_path, "wb") as f:
        pickle.dump(partitioning, f)

    print("Created TCGA-BRCA dataloaders:")
    print(f"  - samples: {len(dataset)}")
    print(f"  - labels: {label_counts}")
    print(f"  - clients: {opt.num_clients}")
    print(f"  - tasks per client: {opt.num_task}")
    print(f"  - task split strategy: {getattr(opt, 'task_split_strategy', None)}")
    if getattr(opt, "task_metadata", None):
        for task in opt.task_metadata:
            task_id = task["task_id"]
            task_key = str(task_id)
            train_counts = getattr(opt, "task_label_counts", {}).get(task_key, {})
            test_counts = getattr(opt, "task_test_label_counts", {}).get(task_key, {})
            print(
                f"  - task {task_id} {task['task_name']}: "
                f"train={train_counts}, test={test_counts}"
            )
    if getattr(opt, "excluded_unknown_stage_count", 0):
        print(f"  - excluded unknown-stage tumor samples: {opt.excluded_unknown_stage_count}")
    print(f"  - input_dim: {opt.input_dim}")
    if getattr(opt, "client_spatial_features", None) is not None:
        print(f"  - TCIA MRI spatial feature dim: {opt.client_spatial_features[0].shape[1]}")
    if getattr(opt, "client_temporal_features", None) is not None:
        print(f"  - DCE kinetic temporal feature dim: {opt.client_temporal_features[0].shape[1]}")

    return client_loaders
