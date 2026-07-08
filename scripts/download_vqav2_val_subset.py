#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Download and prepare a small VQAv2 val2014 subset.

The script downloads the official VQAv2 validation question/annotation files
and only the COCO val2014 images referenced by the selected examples. It writes
the local ``metadata.jsonl`` format consumed by the Qwen3-VL J-lens scripts.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jlens.qwen3vl import write_vqa_splits

DEFAULT_QUESTION_URL = (
    "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/"
    "v2_Questions_Val_mscoco.zip"
)
DEFAULT_ANNOTATION_URL = (
    "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/"
    "v2_Annotations_Val_mscoco.zip"
)
DEFAULT_IMAGE_BASE_URL = "http://images.cocodataset.org/val2014"
QUESTION_JSON = "v2_OpenEnded_mscoco_val2014_questions.json"
ANNOTATION_JSON = "v2_mscoco_val2014_annotations.json"
USER_AGENT = "jacobian-lens-vqav2-subset/1.0"


def _download(url: str, dest: Path, *, overwrite: bool, timeout: float) -> None:
    if dest.exists() and dest.stat().st_size > 0 and not overwrite:
        print(f"[skip] {dest}", file=sys.stderr)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    print(f"[download] {url} -> {dest}", file=sys.stderr)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            with tmp.open("wb") as f:
                shutil.copyfileobj(response, f, length=1024 * 1024)
        if tmp.stat().st_size == 0:
            raise RuntimeError(f"downloaded empty file: {url}")
        tmp.replace(dest)
    finally:
        if tmp.exists():
            tmp.unlink()


def _extract_json(
    zip_path: Path,
    filename: str,
    dest: Path,
    *,
    overwrite: bool,
) -> Path:
    if dest.exists() and dest.stat().st_size > 0 and not overwrite:
        print(f"[skip] {dest}", file=sys.stderr)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with zipfile.ZipFile(zip_path) as zf:
        matches = [name for name in zf.namelist() if Path(name).name == filename]
        if not matches:
            raise FileNotFoundError(f"{filename} not found in {zip_path}")
        print(f"[extract] {matches[0]} -> {dest}", file=sys.stderr)
        with zf.open(matches[0]) as src:
            with tmp.open("wb") as f:
                shutil.copyfileobj(src, f)
    tmp.replace(dest)
    return dest


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _image_name(image_id: int) -> str:
    return f"COCO_val2014_{image_id:012d}.jpg"


def _select_pairs(
    questions: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    *,
    limit: int,
    sample_mode: str,
    seed: int,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    by_question_id = {int(a["question_id"]): a for a in annotations}
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for question in questions:
        question_id = int(question["question_id"])
        annotation = by_question_id.get(question_id)
        if annotation is not None:
            pairs.append((question, annotation))
    if limit > len(pairs):
        raise ValueError(f"requested {limit} examples, but only {len(pairs)} are available")
    if sample_mode == "random":
        rng = random.Random(seed)
        pairs = list(pairs)
        rng.shuffle(pairs)
    return pairs[:limit]


def _unique_image_ids(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]]
) -> list[int]:
    seen: set[int] = set()
    image_ids: list[int] = []
    for question, _ in pairs:
        image_id = int(question["image_id"])
        if image_id in seen:
            continue
        seen.add(image_id)
        image_ids.append(image_id)
    return image_ids


def _download_images(
    image_ids: list[int],
    *,
    image_base_url: str,
    image_dir: Path,
    overwrite: bool,
    timeout: float,
) -> None:
    base = image_base_url.rstrip("/")
    for i, image_id in enumerate(image_ids, start=1):
        name = _image_name(image_id)
        print(f"[images] {i}/{len(image_ids)} {name}", file=sys.stderr)
        _download(
            f"{base}/{name}",
            image_dir / name,
            overwrite=overwrite,
            timeout=timeout,
        )


def _write_metadata(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    metadata_path: Path,
    *,
    overwrite: bool,
) -> None:
    if metadata_path.exists() and not overwrite:
        print(f"[skip] {metadata_path}", file=sys.stderr)
        return
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = metadata_path.with_name(metadata_path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for index, (question, annotation) in enumerate(pairs):
            image_id = int(question["image_id"])
            record = {
                "index": index,
                "question_id": int(question["question_id"]),
                "image_id": image_id,
                "image_file": f"images/{_image_name(image_id)}",
                "question": question["question"],
                "multiple_choice_answer": annotation.get("multiple_choice_answer", ""),
                "answer_type": annotation.get("answer_type"),
                "question_type": annotation.get("question_type"),
                "answers": annotation.get("answers", []),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp.replace(metadata_path)


def _count_jsonl(path: Path) -> int:
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download a VQAv2 val2014 subset and write J-lens metadata."
    )
    parser.add_argument("--out", required=True, help="Output dataset directory.")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--sample-mode", choices=["first", "random"], default="first")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fit-count", type=int, default=900)
    parser.add_argument("--val-count", type=int, default=100)
    parser.add_argument("--question-url", default=DEFAULT_QUESTION_URL)
    parser.add_argument("--annotation-url", default=DEFAULT_ANNOTATION_URL)
    parser.add_argument("--image-base-url", default=DEFAULT_IMAGE_BASE_URL)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-splits", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out = Path(args.out).expanduser().resolve()
    raw_dir = out / "raw"
    image_dir = out / "images"
    metadata_path = out / "metadata.jsonl"

    question_zip = raw_dir / Path(args.question_url).name
    annotation_zip = raw_dir / Path(args.annotation_url).name
    _download(args.question_url, question_zip, overwrite=args.overwrite, timeout=args.timeout)
    _download(
        args.annotation_url,
        annotation_zip,
        overwrite=args.overwrite,
        timeout=args.timeout,
    )

    question_json = _extract_json(
        question_zip, QUESTION_JSON, raw_dir / QUESTION_JSON, overwrite=args.overwrite
    )
    annotation_json = _extract_json(
        annotation_zip,
        ANNOTATION_JSON,
        raw_dir / ANNOTATION_JSON,
        overwrite=args.overwrite,
    )
    question_data = _load_json(question_json)
    annotation_data = _load_json(annotation_json)
    pairs = _select_pairs(
        question_data["questions"],
        annotation_data["annotations"],
        limit=args.limit,
        sample_mode=args.sample_mode,
        seed=args.seed,
    )

    image_ids = _unique_image_ids(pairs)
    _download_images(
        image_ids,
        image_base_url=args.image_base_url,
        image_dir=image_dir,
        overwrite=args.overwrite,
        timeout=args.timeout,
    )
    _write_metadata(pairs, metadata_path, overwrite=args.overwrite)

    fit_path = None
    val_path = None
    if not args.no_splits:
        required = args.fit_count + args.val_count
        available = _count_jsonl(metadata_path)
        if available < required:
            raise ValueError(
                f"need at least {required} metadata rows for the requested split, "
                f"found {available}"
            )
        existing_fit = out / "splits" / f"fit_{args.fit_count}.jsonl"
        existing_val = out / "splits" / f"val_{args.val_count}.jsonl"
        if existing_fit.exists() and existing_val.exists() and not args.overwrite:
            fit_path, val_path = existing_fit, existing_val
            print(f"[skip] {existing_fit}", file=sys.stderr)
            print(f"[skip] {existing_val}", file=sys.stderr)
        else:
            fit_path, val_path = write_vqa_splits(
                out, fit_count=args.fit_count, val_count=args.val_count
            )

    summary = {
        "out": str(out),
        "metadata": str(metadata_path),
        "records": _count_jsonl(metadata_path),
        "unique_images": len(image_ids),
        "fit_split": str(fit_path) if fit_path else None,
        "val_split": str(val_path) if val_path else None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
