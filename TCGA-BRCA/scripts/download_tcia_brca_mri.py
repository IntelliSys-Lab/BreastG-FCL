#!/usr/bin/env python3
"""Download TCIA TCGA-BRCA MRI DICOM series via the public NBIA API."""

import argparse
import csv
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

API_BASE = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
COLLECTION = "TCGA-BRCA"


def fetch_json(url, timeout=120):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def get_series(collection=COLLECTION):
    query = urllib.parse.urlencode({"Collection": collection})
    url = f"{API_BASE}/getSeries?{query}"
    return fetch_json(url)


def write_manifest(series, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "tcia_tcga_brca_mri_series.json"
    csv_path = out_dir / "tcia_tcga_brca_mri_series.csv"

    with json_path.open("w") as f:
        json.dump(series, f, indent=2)

    fieldnames = [
        "PatientID",
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "Modality",
        "BodyPartExamined",
        "StudyDesc",
        "SeriesDescription",
        "ProtocolName",
        "Manufacturer",
        "ManufacturerModelName",
        "ImageCount",
        "FileSize",
        "StudyDate",
        "SeriesDate",
        "CollectionURI",
        "LicenseName",
        "LicenseURI",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in series:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    return json_path, csv_path


def load_state(path):
    if path.exists():
        with path.open() as f:
            return json.load(f)
    return {"completed": {}, "failed": {}}


def save_state(path, state):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    tmp_path.replace(path)


def download_series_zip(series_uid, out_path, timeout=600, chunk_size=1024 * 1024):
    query = urllib.parse.urlencode({"SeriesInstanceUID": series_uid})
    url = f"{API_BASE}/getImage?{query}"
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()

    request = urllib.request.Request(url, headers={"User-Agent": "TCGA-GFedCL-TCIA-Downloader/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response, tmp_path.open("wb") as f:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)

    if tmp_path.stat().st_size == 0:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded empty response for {series_uid}")
    tmp_path.replace(out_path)
    return out_path.stat().st_size


def format_size(num_bytes):
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="TCGA-BRCA/data/tcia_mri")
    parser.add_argument("--collection", default=COLLECTION)
    parser.add_argument("--modality", default="MR")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    manifest_dir = output_dir / "metadata"
    raw_dir = output_dir / "raw" / "series_zip"
    state_path = manifest_dir / "download_state.json"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    print(f"Querying TCIA NBIA API for collection={args.collection} ...", flush=True)
    all_series = get_series(args.collection)
    series = [row for row in all_series if row.get("Modality") == args.modality]
    series.sort(key=lambda row: (row.get("PatientID", ""), row.get("StudyInstanceUID", ""), row.get("SeriesInstanceUID", "")))
    if args.limit is not None:
        series = series[: args.limit]

    total_size = sum(int(row.get("FileSize") or 0) for row in series)
    total_images = sum(int(row.get("ImageCount") or 0) for row in series)
    patients = {row.get("PatientID") for row in series}

    json_path, csv_path = write_manifest(series, manifest_dir)
    print(f"MRI series: {len(series)}", flush=True)
    print(f"Patients: {len(patients)}", flush=True)
    print(f"Images: {total_images}", flush=True)
    print(f"Reported total size: {format_size(total_size)}", flush=True)
    print(f"Manifest JSON: {json_path}", flush=True)
    print(f"Manifest CSV: {csv_path}", flush=True)

    if args.manifest_only:
        return 0

    state = load_state(state_path)
    completed = state.setdefault("completed", {})
    failed = state.setdefault("failed", {})

    for index, row in enumerate(series, start=1):
        series_uid = row["SeriesInstanceUID"]
        patient_id = row.get("PatientID", "unknown")
        out_path = raw_dir / f"{patient_id}__{series_uid}.zip"

        if series_uid in completed and out_path.exists() and out_path.stat().st_size > 0:
            print(f"[{index}/{len(series)}] skip existing {patient_id} {series_uid}", flush=True)
            continue
        if out_path.exists() and out_path.stat().st_size > 0:
            completed[series_uid] = {
                "path": str(out_path),
                "bytes": out_path.stat().st_size,
                "patient_id": patient_id,
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            failed.pop(series_uid, None)
            save_state(state_path, state)
            print(f"[{index}/{len(series)}] registered existing {patient_id} {series_uid}", flush=True)
            continue

        print(
            f"[{index}/{len(series)}] downloading {patient_id} "
            f"images={row.get('ImageCount')} reported={format_size(int(row.get('FileSize') or 0))}",
            flush=True,
        )
        last_error = None
        for attempt in range(1, args.retries + 1):
            try:
                bytes_written = download_series_zip(series_uid, out_path, timeout=args.timeout)
                completed[series_uid] = {
                    "path": str(out_path),
                    "bytes": bytes_written,
                    "patient_id": patient_id,
                    "image_count": row.get("ImageCount"),
                    "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                failed.pop(series_uid, None)
                save_state(state_path, state)
                print(f"    done {format_size(bytes_written)} -> {out_path}", flush=True)
                break
            except Exception as exc:
                last_error = str(exc)
                print(f"    attempt {attempt}/{args.retries} failed: {last_error}", flush=True)
                time.sleep(min(30, args.sleep * (2 ** attempt)))
        else:
            failed[series_uid] = {
                "patient_id": patient_id,
                "error": last_error,
                "failed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            save_state(state_path, state)
            print(f"    failed permanently: {series_uid}", flush=True)

        time.sleep(args.sleep)

    save_state(state_path, state)
    print(f"Completed: {len(completed)}", flush=True)
    print(f"Failed: {len(failed)}", flush=True)
    print(f"State: {state_path}", flush=True)
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
