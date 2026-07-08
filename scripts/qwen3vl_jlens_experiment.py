#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Run Qwen3-VL multimodal Jacobian-lens experiments.

Examples:

    python scripts/qwen3vl_jlens_experiment.py make-splits
    python scripts/qwen3vl_jlens_experiment.py smoke-forward --n 5
    python scripts/qwen3vl_jlens_experiment.py smoke-hooks --n 1
    python scripts/qwen3vl_jlens_experiment.py vanilla-cat
    python scripts/qwen3vl_jlens_experiment.py fit --metadata .../splits/fit_900.jsonl --layer-stride 2 --out lens.pt
    python scripts/qwen3vl_jlens_experiment.py validate --metadata .../splits/val_100.jsonl --lens lens.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jlens
from jlens.qwen3vl import (
    Qwen3VLJLensModel,
    Qwen3VLResidualRecorder,
    evaluate_answer_ranks,
    fit_vqa_jacobian_lens,
    load_vqa_metadata,
    parse_layer_list,
    stride_layers,
    token_ranks,
    write_vqa_splits,
)

DEFAULT_MODEL = "/home/dkx/Projects/VLM/qwen3-vl/Qwen3-VL-4B-Instruct-ckpt"
DEFAULT_DATA = "/home/dkx/Datasets/VQAv2_val_1000"
DEFAULT_CAT = "/home/dkx/Datasets/VQAv2_val_1000/cat.png"


def _load_model(args: argparse.Namespace) -> Qwen3VLJLensModel:
    jlens.configure_logging()
    logging.getLogger("jlens").setLevel(args.log_level)
    return Qwen3VLJLensModel.from_pretrained(
        args.model,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn,
    )


def _metadata_path(args: argparse.Namespace) -> Path:
    return Path(args.metadata or Path(args.data_dir) / "metadata.jsonl")


def _layers_for_args(
    model: Qwen3VLJLensModel,
    args: argparse.Namespace,
    *,
    default_stride: int = 1,
) -> list[int]:
    target = model.n_layers - 1 if args.target_layer is None else args.target_layer
    if target < 0:
        target += model.n_layers
    explicit = parse_layer_list(args.layers)
    if explicit is not None:
        return explicit
    return stride_layers(model.n_layers, target_layer=target, stride=args.layer_stride or default_stride)


def cmd_make_splits(args: argparse.Namespace) -> None:
    fit_path, val_path = write_vqa_splits(
        args.data_dir, fit_count=args.fit_count, val_count=args.val_count
    )
    print(json.dumps({"fit": str(fit_path), "val": str(val_path)}, indent=2))


def cmd_smoke_forward(args: argparse.Namespace) -> None:
    model = _load_model(args)
    samples = load_vqa_metadata(_metadata_path(args), limit=args.n)
    rows: list[dict[str, Any]] = []
    for sample in samples:
        encoded = model.encode_sample(sample)
        inputs = model.batch_to_device(encoded.inputs)
        with torch.no_grad():
            out = model.forward_logits(inputs, logits_to_keep=0)
        first_pos = encoded.answer_prediction_positions[0]
        first_tid = encoded.answer_token_ids[0]
        logits = out.logits[0, first_pos].detach().cpu()
        rank = token_ranks(logits, [first_tid])[0]
        pred_id = int(logits.argmax().item())
        rows.append(
            {
                "question_id": sample.question_id,
                "seq_len": encoded.seq_len,
                "image_tokens": int((encoded.inputs["mm_token_type_ids"][0] != 0).sum()),
                "question": sample.question,
                "answer": sample.short_answer,
                "answer_token_id": first_tid,
                "answer_first_token_rank": rank,
                "model_top1": model.tokenizer.decode([pred_id], skip_special_tokens=False),
            }
        )
        print(json.dumps(rows[-1], ensure_ascii=False))
    if torch.cuda.is_available():
        print("max_cuda_memory_allocated", torch.cuda.max_memory_allocated())


def cmd_smoke_hooks(args: argparse.Namespace) -> None:
    model = _load_model(args)
    samples = load_vqa_metadata(_metadata_path(args), limit=args.n)
    layers = _layers_for_args(model, args, default_stride=max(1, model.n_layers // 4))
    for sample in samples:
        encoded = model.encode_sample(sample)
        inputs = model.batch_to_device(encoded.inputs)
        with torch.no_grad(), Qwen3VLResidualRecorder(model.layers, at=layers) as recorder:
            model.forward_model(inputs)
        shapes = {layer: tuple(recorder.activations[layer].shape) for layer in layers}
        print(
            json.dumps(
                {
                    "question_id": sample.question_id,
                    "seq_len": encoded.seq_len,
                    "layers": layers,
                    "activation_shapes": shapes,
                },
                ensure_ascii=False,
            )
        )


def _candidate_ids(tokenizer, word: str) -> list[int]:
    ids: set[int] = set()
    for text in (word, " " + word, word.capitalize(), " " + word.capitalize()):
        encoded = tokenizer(text, add_special_tokens=False).input_ids
        if len(encoded) == 1:
            ids.add(int(encoded[0]))
    return sorted(ids)


def cmd_vanilla_cat(args: argparse.Namespace) -> None:
    model = _load_model(args)
    inputs_cpu = model.encode_generation_prompt(args.image, args.question)
    inputs = model.batch_to_device(inputs_cpu)
    layers = _layers_for_args(model, args, default_stride=args.layer_stride or 3)
    candidates = [c.strip() for c in args.candidates.split(",") if c.strip()]
    candidate_ids = {c: _candidate_ids(model.tokenizer, c) for c in candidates}
    with torch.no_grad(), Qwen3VLResidualRecorder(model.layers, at=layers) as recorder:
        model.forward_model(inputs)
        rows = []
        pos = inputs["input_ids"].shape[-1] - 1
        for layer in layers:
            residual = recorder.activations[layer][0, pos].float()
            logits = model.unembed(residual).detach().cpu()
            row = {"layer": layer}
            for word, ids in candidate_ids.items():
                row[word] = min(token_ranks(logits, ids)) if ids else None
            top_ids = logits.topk(args.top_k).indices.tolist()
            row["top"] = model.tokenizer.batch_decode(
                [[int(t)] for t in top_ids],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            rows.append(row)
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))


def cmd_fit(args: argparse.Namespace) -> None:
    model = _load_model(args)
    samples = load_vqa_metadata(_metadata_path(args), limit=args.limit)
    source_layers = _layers_for_args(model, args, default_stride=args.layer_stride or 1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint) if args.checkpoint else out.with_suffix(".ckpt.pt")
    lens = fit_vqa_jacobian_lens(
        model,
        samples,
        source_layers=source_layers,
        target_layer=args.target_layer,
        dim_batch=args.dim_batch,
        position_scope=args.position_scope,
        skip_first=args.skip_first,
        checkpoint_path=checkpoint,
        checkpoint_every=args.checkpoint_every,
        resume=not args.no_resume,
    )
    lens.save(str(out), dtype=getattr(torch, args.save_dtype))
    print(
        json.dumps(
            {
                "out": str(out),
                "checkpoint": str(checkpoint),
                "n_prompts": lens.n_prompts,
                "source_layers": lens.source_layers,
                "d_model": lens.d_model,
            },
            indent=2,
        )
    )


def cmd_validate(args: argparse.Namespace) -> None:
    model = _load_model(args)
    lens = jlens.JacobianLens.load(args.lens)
    samples = load_vqa_metadata(_metadata_path(args), limit=args.limit)
    layers = parse_layer_list(args.layers) or lens.source_layers
    results = evaluate_answer_ranks(
        model,
        lens,
        samples,
        layers=layers,
        use_jacobian=not args.vanilla,
    )
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for record in results:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    best = [r["best_rank"] for r in results if r["best_rank"] is not None]
    summary = {
        "n": len(results),
        "layers": layers,
        "mean_best_rank": sum(best) / len(best) if best else None,
        "top1_rate": sum(1 for r in best if r == 0) / len(best) if best else None,
        "top10_rate": sum(1 for r in best if r < 10) / len(best) if best else None,
        "out": args.out,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data-dir", default=DEFAULT_DATA)
    parser.add_argument("--metadata")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--attn", default="sdpa")
    parser.add_argument("--log-level", default="INFO")

    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("make-splits")
    p.add_argument("--fit-count", type=int, default=900)
    p.add_argument("--val-count", type=int, default=100)
    p.set_defaults(func=cmd_make_splits)

    p = sub.add_parser("smoke-forward")
    p.add_argument("--n", type=int, default=5)
    p.set_defaults(func=cmd_smoke_forward)

    p = sub.add_parser("smoke-hooks")
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--layers")
    p.add_argument("--layer-stride", type=int)
    p.add_argument("--target-layer", type=int)
    p.set_defaults(func=cmd_smoke_hooks)

    p = sub.add_parser("vanilla-cat")
    p.add_argument("--image", default=DEFAULT_CAT)
    p.add_argument("--question", default="What animal is shown in the image?")
    p.add_argument("--candidates", default="cat,dog,car")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--layers")
    p.add_argument("--layer-stride", type=int, default=3)
    p.add_argument("--target-layer", type=int)
    p.set_defaults(func=cmd_vanilla_cat)

    p = sub.add_parser("fit")
    p.add_argument("--out", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--limit", type=int)
    p.add_argument("--layers")
    p.add_argument("--layer-stride", type=int, default=1)
    p.add_argument("--target-layer", type=int)
    p.add_argument("--dim-batch", type=int, default=8)
    p.add_argument(
        "--position-scope",
        choices=["all_nonfinal", "text_nonfinal", "answer_prediction"],
        default="all_nonfinal",
    )
    p.add_argument("--skip-first", type=int, default=0)
    p.add_argument("--checkpoint-every", type=int, default=1)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--save-dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.set_defaults(func=cmd_fit)

    p = sub.add_parser("validate")
    p.add_argument("--lens", required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--layers")
    p.add_argument("--out")
    p.add_argument("--vanilla", action="store_true")
    p.set_defaults(func=cmd_validate)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
