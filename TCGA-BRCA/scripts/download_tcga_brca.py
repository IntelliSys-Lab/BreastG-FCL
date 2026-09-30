#!/usr/bin/env python3
"""Download public TCGA-BRCA STAR count files from the NCI GDC API."""

from __future__ import annotations

import argparse
import json
import os
import tarfile
import tempfile
import time
from pathlib import Path

import requests


GDC_API = "https://api.gdc.cancer.gov"
PROJECT_ID = "TCGA-BRCA"
FIELDS = [
    "file_id",
    "file_name",
    "file_size",
    "md5sum",
    "access",
    "data_category",
    "data_type",
    "cases.submitter_id",
    "cases.case_id",
    "cases.samples.submitter_id",
    "cases.samples.sample_type",
    "cases.samples.tissue_type",
    "analysis.workflow_type",
]


def build_filters() -> dict:
    return {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {
                    "field": "cases.project.project_id",
                    "value": [PROJECT_ID],
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "files.data_category",
                    "value": ["Transcriptome Profiling"],
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "files.data_type",
                    "value": ["Gene Expression Quantification"],
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "files.analysis.workflow_type",
                    "value": ["STAR - Counts"],
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "files.access",
                    "value": ["open"],
                },
            },
        ],
    }


def request_json(session: requests.Session, url: str, **kwargs) -> dict:
    response = session.get(url, timeout=90, **kwargs)
    response.raise_for_status()
    return response.json()


def fetch_status(session: requests.Session) -> dict:
    return request_json(session, f"{GDC_API}/status")


def fetch_manifest(session: requests.Session, page_size: int = 1000) -> list[dict]:
    filters = build_filters()
    params = {
        "filters": json.dumps(filters),
        "fields": ",".join(FIELDS),
        "format": "JSON",
        "size": page_size,
        "from": 0,
        "sort": "file_id:asc",
    }

    files: list[dict] = []
    while True:
        data = request_json(session, f"{GDC_API}/files", params=params)["data"]
        files.extend(data["hits"])
        pagination = data["pagination"]
        if len(files) >= pagination["total"]:
            break
        params["from"] += page_size
    return files


def first_nested(hit: dict, path: list[str], default: str = "") -> str:
    value = hit
    for key in path:
        if isinstance(value, list):
            value = value[0] if value else {}
        if not isinstance(value, dict):
            return default
        value = value.get(key, {})
    if isinstance(value, list):
        value = value[0] if value else default
    return str(value) if value not in ({}, None) else default


def write_manifest(files: list[dict], path: Path) -> None:
    columns = [
        "file_id",
        "file_name",
        "file_size",
        "md5sum",
        "case_submitter_id",
        "case_id",
        "sample_submitter_id",
        "sample_type",
        "tissue_type",
        "workflow_type",
    ]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(columns) + "\n")
        for hit in files:
            row = {
                "file_id": hit.get("file_id", hit.get("id", "")),
                "file_name": hit.get("file_name", ""),
                "file_size": str(hit.get("file_size", "")),
                "md5sum": hit.get("md5sum", ""),
                "case_submitter_id": first_nested(hit, ["cases", "submitter_id"]),
                "case_id": first_nested(hit, ["cases", "case_id"]),
                "sample_submitter_id": first_nested(
                    hit, ["cases", "samples", "submitter_id"]
                ),
                "sample_type": first_nested(hit, ["cases", "samples", "sample_type"]),
                "tissue_type": first_nested(hit, ["cases", "samples", "tissue_type"]),
                "workflow_type": first_nested(hit, ["analysis", "workflow_type"]),
            }
            handle.write("\t".join(row[column] for column in columns) + "\n")


def chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def load_downloaded(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as handle:
        return set(json.load(handle))


def save_downloaded(path: Path, downloaded: set[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(sorted(downloaded), handle, indent=2)


def extract_download_tar(tar_path: Path, raw_dir: Path) -> list[str]:
    extracted_ids: list[str] = []
    with tarfile.open(tar_path, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            parts = Path(member.name).parts
            if len(parts) < 2:
                continue
            file_id = parts[0]
            target_dir = raw_dir / file_id
            target_dir.mkdir(parents=True, exist_ok=True)
            member.name = parts[-1]
            archive.extract(member, target_dir)
            extracted_ids.append(file_id)
    return extracted_ids


def download_files(
    session: requests.Session,
    files: list[dict],
    raw_dir: Path,
    metadata_dir: Path,
    chunk_size: int,
    pause_seconds: float,
) -> None:
    downloaded_path = metadata_dir / "downloaded_files.json"
    downloaded = load_downloaded(downloaded_path)
    file_ids = [hit.get("file_id", hit.get("id")) for hit in files]
    pending = [file_id for file_id in file_ids if file_id and file_id not in downloaded]

    for index, ids in enumerate(chunked(pending, chunk_size), start=1):
        print(f"Downloading chunk {index}: {len(ids)} files")
        response = session.post(
            f"{GDC_API}/data",
            json={"ids": ids},
            timeout=600,
            stream=True,
        )
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(
            prefix="gdc_", suffix=".tar.gz", delete=False
        ) as handle:
            tmp_path = Path(handle.name)
            for block in response.iter_content(chunk_size=1024 * 1024):
                if block:
                    handle.write(block)

        extracted_ids = extract_download_tar(tmp_path, raw_dir)
        tmp_path.unlink(missing_ok=True)
        downloaded.update(extracted_ids)
        save_downloaded(downloaded_path, downloaded)
        print(f"Downloaded {len(downloaded)} / {len(file_ids)} files")
        if pause_seconds:
            time.sleep(pause_seconds)


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=project_dir)
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--pause-seconds", type=float, default=0.5)
    parser.add_argument("--metadata-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    metadata_dir = root / "metadata"
    raw_dir = root / "data" / "raw"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    status = fetch_status(session)
    files = fetch_manifest(session)

    (metadata_dir / "gdc_status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    (metadata_dir / "gdc_query.json").write_text(
        json.dumps(
            {
                "api": GDC_API,
                "project_id": PROJECT_ID,
                "filters": build_filters(),
                "fields": FIELDS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_manifest(files, metadata_dir / "gdc_files_manifest.tsv")

    total_size = sum(int(hit.get("file_size", 0)) for hit in files)
    print(f"Found {len(files)} files ({total_size / 1024**3:.2f} GiB raw).")
    print(f"GDC data release: {status.get('data_release', 'unknown')}")

    if args.metadata_only:
        return

    download_files(
        session=session,
        files=files,
        raw_dir=raw_dir,
        metadata_dir=metadata_dir,
        chunk_size=args.chunk_size,
        pause_seconds=args.pause_seconds,
    )


if __name__ == "__main__":
    main()

