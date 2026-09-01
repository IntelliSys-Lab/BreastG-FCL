#!/usr/bin/env python3
"""Extract HealthCom spatial morphology features from TCIA TCGA-BRCA DCE-MRI.

Features approximate the four HealthCom spatial regions:
- tumor core
- enhancing rim
- peritumoral ring
- background parenchymal enhancement (BPE)

Without lesion masks, tumor regions are estimated from robust peak-vs-baseline
DCE enhancement maps. The output is patient-level and suitable as the
spatial-attention feature table.
"""

import argparse
import csv
import io
import math
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pydicom


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-dir", default="TCGA-BRCA/data/tcia_mri/raw/series_zip")
    parser.add_argument(
        "--study-kinetics",
        default="TCGA-BRCA/data/tcia_mri/features/dce_kinetics_study.csv",
        help="Study-level DCE kinetics CSV produced by extract_dce_kinetics.py.",
    )
    parser.add_argument("--output-dir", default="TCGA-BRCA/data/tcia_mri/features")
    parser.add_argument("--max-slices-per-study", type=int, default=16)
    parser.add_argument("--central-crop", type=float, default=0.8)
    parser.add_argument("--core-percentile", type=float, default=98.0)
    parser.add_argument("--rim-iterations", type=int, default=2)
    parser.add_argument("--peri-iterations", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def safe_float(value, default=np.nan):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def build_zip_index(zip_dir):
    index = {}
    for path in Path(zip_dir).glob("*.zip"):
        if "__" not in path.stem:
            continue
        uid = path.stem.split("__", 1)[1]
        index[uid] = path
    return index


def sample_names(names, max_count):
    names = sorted(names)
    if max_count <= 0 or len(names) <= max_count:
        return names
    idx = np.linspace(0, len(names) - 1, max_count).round().astype(int)
    return [names[i] for i in sorted(set(idx.tolist()))]


def crop_center(arr, fraction):
    if arr.ndim > 2:
        arr = arr.reshape(-1, arr.shape[-2], arr.shape[-1])[0]
    h, w = arr.shape[-2:]
    frac = min(max(float(fraction), 0.1), 1.0)
    y0 = int((1.0 - frac) * h / 2.0)
    y1 = h - y0
    x0 = int((1.0 - frac) * w / 2.0)
    x1 = w - x0
    return arr[y0:y1, x0:x1]


def dicom_image(ds, central_crop):
    arr = ds.pixel_array.astype(np.float32, copy=False)
    slope = safe_float(getattr(ds, "RescaleSlope", 1.0), 1.0)
    intercept = safe_float(getattr(ds, "RescaleIntercept", 0.0), 0.0)
    arr = arr * slope + intercept
    return crop_center(arr, central_crop)


def read_series_images(zip_path, max_slices, central_crop):
    images = []
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".dcm")]
        for name in sample_names(names, max_slices):
            ds = pydicom.dcmread(io.BytesIO(zf.read(name)), force=True)
            try:
                image = dicom_image(ds, central_crop)
            except Exception:
                continue
            if np.isfinite(image).any():
                images.append(image)
    return images


def match_image_pairs(base_images, peak_images):
    n = min(len(base_images), len(peak_images))
    if n == 0:
        return []
    pairs = []
    for i in range(n):
        b = base_images[i]
        p = peak_images[i]
        h = min(b.shape[-2], p.shape[-2])
        w = min(b.shape[-1], p.shape[-1])
        pairs.append((b[:h, :w], p[:h, :w]))
    return pairs


def shift_or_zero(mask, dy, dx):
    out = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    src_y0 = max(0, -dy)
    src_y1 = min(h, h - dy)
    src_x0 = max(0, -dx)
    src_x1 = min(w, w - dx)
    dst_y0 = max(0, dy)
    dst_y1 = min(h, h + dy)
    dst_x0 = max(0, dx)
    dst_x1 = min(w, w + dx)
    if src_y1 > src_y0 and src_x1 > src_x0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = mask[src_y0:src_y1, src_x0:src_x1]
    return out


def dilate(mask, iterations):
    out = mask.astype(bool)
    for _ in range(max(0, int(iterations))):
        expanded = out.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                expanded |= shift_or_zero(out, dy, dx)
        out = expanded
    return out


def perimeter(mask):
    if not mask.any():
        return 0.0
    eroded = mask.copy()
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        eroded &= shift_or_zero(mask, dy, dx)
    return float(np.logical_and(mask, ~eroded).sum())


def mask_stats(values, mask):
    vals = values[mask & np.isfinite(values)]
    if vals.size == 0:
        return 0.0, 0.0, 0.0
    return float(np.mean(vals)), float(np.std(vals)), float(np.percentile(vals, 90))


def foreground_mask(baseline, peak):
    signal = np.maximum(np.abs(baseline), np.abs(peak))
    if signal.size == 0:
        return np.zeros_like(signal, dtype=bool)
    threshold = np.percentile(signal[np.isfinite(signal)], 10)
    return np.isfinite(signal) & (signal > threshold)


def slice_features(baseline, peak, args):
    denom = np.maximum(np.abs(baseline), 1.0)
    enh = (peak - baseline) / denom
    fg = foreground_mask(baseline, peak)
    valid = fg & np.isfinite(enh)
    if valid.sum() < 64:
        return None

    fg_values = enh[valid]
    threshold = max(0.05, float(np.percentile(fg_values, args.core_percentile)))
    core = valid & (enh >= threshold)
    if core.sum() < 8:
        threshold = max(0.01, float(np.percentile(fg_values, 99.0)))
        core = valid & (enh >= threshold)
    if core.sum() < 4:
        return None

    rim_outer = dilate(core, args.rim_iterations)
    peri_outer = dilate(core, args.peri_iterations)
    rim = rim_outer & ~core & valid
    peri = peri_outer & ~rim_outer & valid
    bpe = valid & ~peri_outer

    total = max(float(valid.sum()), 1.0)
    core_area = float(core.sum())
    rim_area = float(rim.sum())
    peri_area = float(peri.sum())
    bpe_area = float(bpe.sum())

    ys, xs = np.where(core)
    bbox_area = core_area
    if ys.size:
        bbox_area = float((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))
    core_mean, core_std, core_p90 = mask_stats(enh, core)
    rim_mean, rim_std, rim_p90 = mask_stats(enh, rim)
    peri_mean, peri_std, peri_p90 = mask_stats(enh, peri)
    bpe_mean, bpe_std, bpe_p90 = mask_stats(enh, bpe)

    left = bpe.copy()
    right = bpe.copy()
    mid = bpe.shape[1] // 2
    left[:, mid:] = False
    right[:, :mid] = False
    left_mean, _, _ = mask_stats(enh, left)
    right_mean, _, _ = mask_stats(enh, right)
    bpe_asym = abs(left_mean - right_mean)
    bpe_high_frac = float(((enh > 0.2) & bpe).sum()) / max(bpe_area, 1.0)

    return {
        "core_area_fraction": core_area / total,
        "core_mean_enhancement": core_mean,
        "core_compactness_proxy": core_area / max(bbox_area, 1.0),
        "core_texture_heterogeneity": core_std,
        "rim_area_fraction": rim_area / total,
        "rim_mean_enhancement": rim_mean,
        "rim_irregularity_proxy": perimeter(core) / max(math.sqrt(core_area), 1.0),
        "rim_to_core_ratio": rim_mean / max(core_mean, 1e-6),
        "peri_area_fraction": peri_area / total,
        "peri_mean_enhancement": peri_mean,
        "peri_texture_heterogeneity": peri_std,
        "peri_to_core_ratio": peri_mean / max(core_mean, 1e-6),
        "bpe_area_fraction": bpe_area / total,
        "bpe_mean_enhancement": bpe_mean,
        "bpe_variance": bpe_std * bpe_std,
        "bpe_left_right_asymmetry": bpe_asym,
        "bpe_high_enhancement_fraction": bpe_high_frac,
        "core_p90_enhancement": core_p90,
        "rim_p90_enhancement": rim_p90,
        "peri_p90_enhancement": peri_p90,
        "bpe_p90_enhancement": bpe_p90,
    }


def summarize_study(row, zip_index, args):
    baseline_uid = row.get("baseline_series_uid")
    peak_uid = row.get("peak_series_uid")
    if baseline_uid not in zip_index or peak_uid not in zip_index:
        return None, "missing_zip"
    base_images = read_series_images(zip_index[baseline_uid], args.max_slices_per_study, args.central_crop)
    peak_images = read_series_images(zip_index[peak_uid], args.max_slices_per_study, args.central_crop)
    features = []
    for baseline, peak in match_image_pairs(base_images, peak_images):
        feat = slice_features(baseline, peak, args)
        if feat is not None:
            features.append(feat)
    if not features:
        return None, "no_usable_tumor_proxy"

    keys = list(features[0].keys())
    out = {
        "PatientID": row.get("PatientID", ""),
        "StudyInstanceUID": row.get("StudyInstanceUID", ""),
        "baseline_series_uid": baseline_uid,
        "peak_series_uid": peak_uid,
        "usable_slices": len(features),
    }
    for key in keys:
        values = np.asarray([f[key] for f in features], dtype=np.float32)
        out[key] = float(np.mean(values))
    return out, ""


def patient_features(study_rows):
    grouped = defaultdict(list)
    for row in study_rows:
        grouped[row["PatientID"]].append(row)
    out = []
    numeric_cols = [
        c for c in study_rows[0].keys()
        if c not in {"PatientID", "StudyInstanceUID", "baseline_series_uid", "peak_series_uid"}
    ] if study_rows else []
    for patient, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda r: float(r.get("core_mean_enhancement", 0.0)), reverse=True)
        best = rows[0]
        record = {"PatientID": patient, "n_spatial_studies": len(rows)}
        for col in numeric_cols:
            values = np.asarray([float(r[col]) for r in rows], dtype=np.float32)
            record[col] = float(best[col])
            record[f"{col}_mean"] = float(np.mean(values))
        out.append(record)
    return out


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    kinetics = read_csv(args.study_kinetics)
    if args.limit:
        kinetics = kinetics[: args.limit]
    zip_index = build_zip_index(args.zip_dir)
    print(f"Processing {len(kinetics)} DCE studies", flush=True)

    study_rows = []
    failures = []
    for i, row in enumerate(kinetics, 1):
        try:
            features, error = summarize_study(row, zip_index, args)
        except Exception as exc:
            features, error = None, repr(exc)
        if features is None:
            failures.append({
                "PatientID": row.get("PatientID", ""),
                "StudyInstanceUID": row.get("StudyInstanceUID", ""),
                "error": error,
            })
        else:
            study_rows.append(features)
        if i % 10 == 0 or i == len(kinetics):
            print(f"[{i}/{len(kinetics)}] studies={len(study_rows)} failures={len(failures)}", flush=True)

    patient_rows = patient_features(study_rows)
    out_dir = Path(args.output_dir)
    write_csv(out_dir / "spatial_morphology_study_features.csv", study_rows)
    write_csv(out_dir / "spatial_morphology_patient_features.csv", patient_rows)
    write_csv(out_dir / "spatial_morphology_failures.csv", failures)
    print(f"Wrote {len(study_rows)} study spatial rows")
    print(f"Wrote {len(patient_rows)} patient spatial rows")
    print(f"Wrote {len(failures)} failures")


if __name__ == "__main__":
    raise SystemExit(main())
