#!/usr/bin/env python3
"""Extract coarse DCE-MRI kinetic features from TCIA TCGA-BRCA DICOM zips.

The output is intended as a patient/study-level temporal feature table for
attention modules. It does not replace lesion-level pharmacokinetic analysis:
without lesion masks, features are computed from robust foreground intensity
statistics over sampled slices.
"""

import argparse
import csv
import io
import math
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pydicom


DCE_TOKENS = (
    "vibrant",
    "dyn",
    "dce",
    "multiphase",
    "post",
    "pre",
    "sub",
    "subtract",
    "cad",
    "contrast",
)
EXCLUDE_TOKENS = ("t2", "localizer", "scout", "survey", "calibration")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zip-dir",
        default="TCGA-BRCA/data/tcia_mri/raw/series_zip",
        help="Directory containing TCIA series zip files.",
    )
    parser.add_argument(
        "--manifest",
        default="TCGA-BRCA/data/tcia_mri/metadata/tcia_tcga_brca_mri_series.csv",
        help="TCIA series manifest CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="TCGA-BRCA/data/tcia_mri/features",
        help="Directory for extracted kinetic feature CSVs.",
    )
    parser.add_argument(
        "--max-images-per-series",
        type=int,
        default=32,
        help="Evenly sample at most this many DICOM images per series.",
    )
    parser.add_argument(
        "--central-crop",
        type=float,
        default=0.8,
        help="Central crop fraction used to reduce background dominance.",
    )
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=0.2,
        help="Fraction of foreground pixels used for robust enhanced-signal mean.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N zips for a quick smoke test.",
    )
    return parser.parse_args()


def read_manifest(path):
    rows = {}
    if not Path(path).exists():
        return rows
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row.get("SeriesInstanceUID")
            if uid:
                rows[uid] = row
    return rows


def parse_dicom_time(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    match = re.match(r"^(\d{2})(\d{2})(\d{2})(?:\.(\d+))?", text)
    if not match:
        return None
    hh, mm, ss, frac = match.groups()
    seconds = int(hh) * 3600 + int(mm) * 60 + int(ss)
    if frac:
        seconds += float("0." + frac)
    return float(seconds)


def unwrap_times(times):
    out = []
    prev = None
    day = 0.0
    for t in times:
        if t is None:
            out.append(None)
            continue
        value = float(t) + day
        if prev is not None and value + 3600 < prev:
            day += 24 * 3600
            value = float(t) + day
        out.append(value)
        prev = value
    return out


def safe_float(value, default=np.nan):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def image_signal(pixel_array, ds, central_crop=0.8, top_fraction=0.2):
    arr = pixel_array.astype(np.float32, copy=False)
    slope = safe_float(getattr(ds, "RescaleSlope", 1.0), 1.0)
    intercept = safe_float(getattr(ds, "RescaleIntercept", 0.0), 0.0)
    arr = arr * slope + intercept

    if arr.ndim > 2:
        arr = arr.reshape(-1, arr.shape[-2], arr.shape[-1])[0]
    h, w = arr.shape[-2:]
    crop = min(max(float(central_crop), 0.1), 1.0)
    y0 = int((1.0 - crop) * h / 2.0)
    y1 = h - y0
    x0 = int((1.0 - crop) * w / 2.0)
    x1 = w - x0
    arr = arr[y0:y1, x0:x1]
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None

    positive = arr[arr > np.percentile(arr, 5)]
    if positive.size < 128:
        positive = arr
    p90 = float(np.percentile(positive, 90))
    p95 = float(np.percentile(positive, 95))
    mean = float(np.mean(positive))
    std = float(np.std(positive))

    n_top = max(1, int(positive.size * min(max(top_fraction, 0.01), 1.0)))
    top_values = np.partition(positive, positive.size - n_top)[-n_top:]
    top_mean = float(np.mean(top_values))
    return mean, std, p90, p95, top_mean


def sample_names(names, max_count):
    names = sorted(names)
    if max_count <= 0 or len(names) <= max_count:
        return names
    idx = np.linspace(0, len(names) - 1, max_count).round().astype(int)
    return [names[i] for i in sorted(set(idx.tolist()))]


def get_text(ds, name, default=""):
    value = getattr(ds, name, default)
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return "\\".join(str(x) for x in value)
    return str(value)


def is_dce_candidate(row):
    text = " ".join(
        str(row.get(k, "") or "")
        for k in ("SeriesDescription", "ProtocolName", "ContrastBolusAgent", "ScanningSequence")
    ).lower()
    has_dce = any(token in text for token in DCE_TOKENS)
    excluded = any(token in text for token in EXCLUDE_TOKENS) and not has_dce
    return bool(has_dce and not excluded)


def summarize_zip(zip_path, manifest_row, args):
    series_uid = zip_path.stem.split("__", 1)[-1]
    with zipfile.ZipFile(zip_path) as zf:
        dicom_names = [n for n in zf.namelist() if n.lower().endswith(".dcm")]
        if not dicom_names:
            return None
        selected = sample_names(dicom_names, args.max_images_per_series)

        signals = []
        times = []
        instance_numbers = []
        temporal_positions = []
        first = None
        pixel_errors = 0
        for name in selected:
            raw = zf.read(name)
            ds = pydicom.dcmread(io.BytesIO(raw), force=True)
            if first is None:
                first = ds
            times.append(
                parse_dicom_time(getattr(ds, "AcquisitionTime", None))
                or parse_dicom_time(getattr(ds, "ContentTime", None))
                or parse_dicom_time(getattr(ds, "SeriesTime", None))
            )
            instance_numbers.append(safe_float(getattr(ds, "InstanceNumber", np.nan)))
            temporal_positions.append(safe_float(getattr(ds, "TemporalPositionIdentifier", np.nan)))
            try:
                signal = image_signal(
                    ds.pixel_array,
                    ds,
                    central_crop=args.central_crop,
                    top_fraction=args.top_fraction,
                )
                if signal is not None:
                    signals.append(signal)
            except Exception:
                pixel_errors += 1

    if first is None or not signals:
        return None

    arr = np.asarray(signals, dtype=np.float32)
    acq_time = (
        parse_dicom_time(getattr(first, "AcquisitionTime", None))
        or parse_dicom_time(getattr(first, "ContentTime", None))
        or parse_dicom_time(getattr(first, "SeriesTime", None))
    )
    row = {
        "PatientID": get_text(first, "PatientID", manifest_row.get("PatientID", "")),
        "StudyInstanceUID": get_text(first, "StudyInstanceUID", manifest_row.get("StudyInstanceUID", "")),
        "SeriesInstanceUID": get_text(first, "SeriesInstanceUID", series_uid),
        "SeriesDescription": get_text(first, "SeriesDescription", manifest_row.get("SeriesDescription", "")),
        "ProtocolName": get_text(first, "ProtocolName", manifest_row.get("ProtocolName", "")),
        "ContrastBolusAgent": get_text(first, "ContrastBolusAgent", ""),
        "ImageType": get_text(first, "ImageType", ""),
        "ScanningSequence": get_text(first, "ScanningSequence", ""),
        "AcquisitionTime": get_text(first, "AcquisitionTime", ""),
        "SeriesTime": get_text(first, "SeriesTime", ""),
        "time_seconds": acq_time if acq_time is not None else "",
        "image_count_manifest": manifest_row.get("ImageCount", ""),
        "sampled_images": len(selected),
        "usable_images": len(signals),
        "pixel_errors": pixel_errors,
        "mean_signal": float(np.mean(arr[:, 0])),
        "std_signal": float(np.mean(arr[:, 1])),
        "p90_signal": float(np.mean(arr[:, 2])),
        "p95_signal": float(np.mean(arr[:, 3])),
        "top_signal": float(np.mean(arr[:, 4])),
        "temporal_position_min": np.nanmin(temporal_positions) if np.isfinite(temporal_positions).any() else "",
        "temporal_position_max": np.nanmax(temporal_positions) if np.isfinite(temporal_positions).any() else "",
        "instance_min": np.nanmin(instance_numbers) if np.isfinite(instance_numbers).any() else "",
        "instance_max": np.nanmax(instance_numbers) if np.isfinite(instance_numbers).any() else "",
    }
    row["is_dce_candidate"] = int(is_dce_candidate(row))
    return row


def rel_enhancement(signal, baseline):
    eps = max(abs(float(baseline)), 1e-6)
    return (float(signal) - float(baseline)) / eps


def build_study_features(series_rows):
    groups = defaultdict(list)
    for row in series_rows:
        if int(row.get("is_dce_candidate", 0)) != 1:
            continue
        groups[(row["PatientID"], row["StudyInstanceUID"])].append(row)

    study_rows = []
    for (patient, study), rows in groups.items():
        rows = sorted(
            rows,
            key=lambda r: (
                float(r["time_seconds"]) if r.get("time_seconds") != "" else math.inf,
                r["SeriesInstanceUID"],
            ),
        )
        if len(rows) < 2:
            continue
        times = unwrap_times([
            float(r["time_seconds"]) if r.get("time_seconds") != "" else None
            for r in rows
        ])
        if all(t is None for t in times):
            times = [float(i) * 60.0 for i in range(len(rows))]

        baseline_idx = 0
        for i, row in enumerate(rows):
            desc = f"{row.get('SeriesDescription', '')} {row.get('ProtocolName', '')}".lower()
            contrast = str(row.get("ContrastBolusAgent", "") or "").strip()
            if "pre" in desc or not contrast:
                baseline_idx = i
                break

        baseline = rows[baseline_idx]
        baseline_signal = float(baseline["top_signal"])
        enhancements = np.asarray(
            [rel_enhancement(r["top_signal"], baseline_signal) for r in rows],
            dtype=np.float32,
        )
        peak_idx = int(np.nanargmax(enhancements))
        late_idx = len(rows) - 1
        early_idx = min(max(baseline_idx + 1, 0), len(rows) - 1)

        t0 = times[baseline_idx] if times[baseline_idx] is not None else float(baseline_idx) * 60.0
        t_peak = times[peak_idx] if times[peak_idx] is not None else float(peak_idx) * 60.0
        t_late = times[late_idx] if times[late_idx] is not None else float(late_idx) * 60.0
        peak_dt_min = max((t_peak - t0) / 60.0, 1e-6)
        washout_dt_min = max((t_late - t_peak) / 60.0, 1e-6)

        peak_enh = float(enhancements[peak_idx])
        late_enh = float(enhancements[late_idx])
        early_enh = float(enhancements[early_idx])
        wash_in = peak_enh / peak_dt_min
        wash_out = (late_enh - peak_enh) / washout_dt_min
        ser = early_enh / max(late_enh, 1e-6)

        valid_times = np.asarray([
            t if t is not None else float(i) * 60.0 for i, t in enumerate(times)
        ], dtype=np.float32)
        rel_minutes = (valid_times - valid_times[baseline_idx]) / 60.0
        order = np.argsort(rel_minutes)
        auc = float(np.trapz(enhancements[order], rel_minutes[order])) if len(rows) > 1 else 0.0

        study_rows.append({
            "PatientID": patient,
            "StudyInstanceUID": study,
            "n_dce_series": len(rows),
            "baseline_series_uid": baseline["SeriesInstanceUID"],
            "peak_series_uid": rows[peak_idx]["SeriesInstanceUID"],
            "late_series_uid": rows[late_idx]["SeriesInstanceUID"],
            "baseline_signal": baseline_signal,
            "early_enhancement": early_enh,
            "peak_enhancement": peak_enh,
            "late_enhancement": late_enh,
            "wash_in_slope_per_min": wash_in,
            "wash_out_slope_per_min": wash_out,
            "time_to_peak_min": peak_dt_min,
            "signal_enhancement_ratio": ser,
            "enhancement_auc": auc,
        })
    return study_rows


def build_patient_features(study_rows):
    by_patient = defaultdict(list)
    for row in study_rows:
        by_patient[row["PatientID"]].append(row)

    out = []
    numeric_cols = (
        "n_dce_series",
        "baseline_signal",
        "early_enhancement",
        "peak_enhancement",
        "late_enhancement",
        "wash_in_slope_per_min",
        "wash_out_slope_per_min",
        "time_to_peak_min",
        "signal_enhancement_ratio",
        "enhancement_auc",
    )
    for patient, rows in sorted(by_patient.items()):
        # Prefer the study with strongest peak enhancement as the representative.
        rows = sorted(rows, key=lambda r: float(r["peak_enhancement"]), reverse=True)
        best = rows[0]
        record = {"PatientID": patient, "n_dce_studies": len(rows)}
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
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    zip_dir = Path(args.zip_dir)
    out_dir = Path(args.output_dir)
    manifest = read_manifest(args.manifest)

    zip_paths = sorted(zip_dir.glob("*.zip"))
    if args.limit:
        zip_paths = zip_paths[: args.limit]
    print(f"Processing {len(zip_paths)} series zips", flush=True)

    series_rows = []
    failures = []
    for i, zip_path in enumerate(zip_paths, 1):
        uid = zip_path.stem.split("__", 1)[-1]
        try:
            row = summarize_zip(zip_path, manifest.get(uid, {}), args)
            if row is not None:
                series_rows.append(row)
        except Exception as exc:
            failures.append({
                "zip_file": str(zip_path),
                "SeriesInstanceUID": uid,
                "error": repr(exc),
            })
        if i % 25 == 0 or i == len(zip_paths):
            print(
                f"[{i}/{len(zip_paths)}] series={len(series_rows)} failures={len(failures)}",
                flush=True,
            )

    study_rows = build_study_features(series_rows)
    patient_rows = build_patient_features(study_rows)

    write_csv(out_dir / "dce_series_signals.csv", series_rows)
    write_csv(out_dir / "dce_kinetics_study.csv", study_rows)
    write_csv(out_dir / "dce_kinetics_patient_features.csv", patient_rows)
    write_csv(out_dir / "dce_kinetics_failures.csv", failures)

    print(f"Wrote {len(series_rows)} series rows")
    print(f"Wrote {len(study_rows)} study kinetic rows")
    print(f"Wrote {len(patient_rows)} patient feature rows")
    print(f"Wrote {len(failures)} failures")


if __name__ == "__main__":
    raise SystemExit(main())
