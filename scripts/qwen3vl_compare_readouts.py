#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Compare vanilla logit-lens and multimodal J-lens readouts on held-out VQA.

For each held-out sample, the script teacher-forces the sequence

    <image>
    User: Question: {question}
    Assistant: {short_answer}

and evaluates the first answer token at the position immediately before it.
For every selected layer it records:

    vanilla: logits_l = lm_head(norm(h_l))
    J-lens:  logits_l = lm_head(norm(J_l @ h_l))

Ranks are reported as 1-based ranks, so MRR is exactly ``1 / rank``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jlens
from jlens.qwen3vl import (
    Qwen3VLJLensModel,
    Qwen3VLResidualRecorder,
    load_vqa_metadata,
    parse_layer_list,
)

DEFAULT_MODEL = "/home/dkx/Projects/VLM/qwen3-vl/Qwen3-VL-4B-Instruct-ckpt"
DEFAULT_METADATA = "/home/dkx/Datasets/VQAv2_val_1000/splits/val_100.jsonl"
DEFAULT_LENS = "/home/dkx/Projects/jacobian-lens/runs/qwen3vl_jlens_fit100_stride2/lens.pt"
DEFAULT_OUT = "/home/dkx/Projects/jacobian-lens/runs/qwen3vl_readout_compare_val5"


def load_lens_or_checkpoint(path: str | Path) -> jlens.JacobianLens:
    """Load a final ``lens.pt`` or a resumable ``lens.ckpt.pt`` checkpoint."""

    path = Path(path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if "J" in state:
        return jlens.JacobianLens(
            jacobians=state["J"],
            n_prompts=int(state["n_prompts"]),
            d_model=int(state["d_model"]),
        )
    if "jacobian_sum" not in state:
        raise ValueError(f"{path} is neither a JacobianLens file nor a fit checkpoint")
    n_done = int(state["n_done"])
    if n_done <= 0:
        raise ValueError(f"{path} has n_done={n_done}; no fitted samples yet")
    jacobians = {
        int(layer): tensor / n_done for layer, tensor in state["jacobian_sum"].items()
    }
    first = next(iter(jacobians.values()))
    return jlens.JacobianLens(
        jacobians=jacobians,
        n_prompts=n_done,
        d_model=int(first.shape[0]),
    )


def first_token_variants(model: Qwen3VLJLensModel, sample) -> tuple[list[int], list[str]]:
    """Unique first-token variants from VQAv2 annotator answers.

    The readout position predicts only the first assistant-answer token, so
    multi-token answers contribute their first token. The multiple-choice answer
    is always included even if it is not present in the ten annotator strings.
    """

    variants = {sample.short_answer.strip()}
    variants.update(answer.strip() for answer in sample.answers if answer.strip())
    token_ids: list[int] = []
    labels: list[str] = []
    seen: set[int] = set()
    for answer in sorted(variants):
        ids = model.tokenizer(answer, add_special_tokens=False).input_ids
        if not ids:
            continue
        token_id = int(ids[0])
        if token_id in seen:
            continue
        seen.add(token_id)
        token_ids.append(token_id)
        labels.append(answer)
    return token_ids, labels


def metric_row(logits: torch.Tensor, target_id: int, variant_ids: list[int]) -> dict[str, Any]:
    """Metrics for one ``[vocab]`` logit row.

    ``rank`` is for the standard answer first token. ``variant_group_rank`` is
    the best rank among the first tokens of all VQAv2 answer variants.
    """

    logits = logits.float()
    log_probs = torch.log_softmax(logits, dim=-1)
    target_logit = logits[target_id]
    rank = int((logits > target_logit).sum().item()) + 1
    logprob = float(log_probs[target_id].item())
    top1_id = int(logits.argmax().item())
    if not variant_ids:
        variant_ids = [target_id]
    variant_tensor = torch.tensor(variant_ids, dtype=torch.long)
    variant_logits = logits[variant_tensor]
    best_variant_logit = variant_logits.max()
    variant_group_rank = int((logits > best_variant_logit).sum().item()) + 1
    variant_group_logprob = float(torch.logsumexp(log_probs[variant_tensor], dim=0).item())
    return {
        "rank": rank,
        "log10_rank": math.log10(rank),
        "logprob": logprob,
        "top20_hit": rank <= 20,
        "mrr": 1.0 / rank,
        "top1_id": top1_id,
        "variant_group_rank": variant_group_rank,
        "variant_group_log10_rank": math.log10(variant_group_rank),
        "variant_group_logprob": variant_group_logprob,
        "variant_group_top20_hit": variant_group_rank <= 20,
        "variant_group_mrr": 1.0 / variant_group_rank,
    }


def evaluate(
    model: Qwen3VLJLensModel,
    lens: jlens.JacobianLens,
    samples,
    *,
    layers: list[int],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    missing = sorted(set(layers) - set(lens.source_layers))
    if missing:
        raise ValueError(f"layers {missing} are not in lens.source_layers={lens.source_layers}")

    n_samples = len(samples)
    for sample_index, sample in enumerate(samples):
        print(
            f"[eval] sample {sample_index + 1}/{n_samples} qid={sample.question_id}",
            file=sys.stderr,
            flush=True,
        )
        encoded = model.encode_sample(sample)
        if not encoded.answer_token_ids:
            continue
        pred_pos = encoded.answer_prediction_positions[0]
        target_id = int(encoded.answer_token_ids[0])
        if pred_pos != encoded.answer_start - 1:
            raise AssertionError(
                f"prediction position {pred_pos} does not precede answer_start "
                f"{encoded.answer_start}"
            )
        actual_target_id = int(encoded.inputs["input_ids"][0, encoded.answer_start])
        if actual_target_id != target_id:
            raise AssertionError(
                f"target_id {target_id} does not match input_ids[answer_start] "
                f"{actual_target_id}"
            )
        variant_ids, variant_labels = first_token_variants(model, sample)
        if target_id not in variant_ids:
            variant_ids.append(target_id)
            variant_labels.append(sample.short_answer)
        inputs = model.batch_to_device(encoded.inputs)
        with torch.no_grad(), Qwen3VLResidualRecorder(model.layers, at=layers) as recorder:
            model.forward_model(inputs)

        target_token = model.tokenizer.decode(
            [target_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        variant_tokens = [
            model.tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            for token_id in variant_ids
        ]
        for layer in layers:
            residual = recorder.activations[layer][0, pred_pos].float()
            vanilla_logits = model.unembed(residual).detach().cpu()
            jlens_logits = model.unembed(lens.transport(residual, layer)).detach().cpu()
            for method, logits in (("vanilla", vanilla_logits), ("jlens", jlens_logits)):
                metrics = metric_row(logits, target_id, variant_ids)
                top1_token = model.tokenizer.decode(
                    [metrics["top1_id"]],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                records.append(
                    {
                        "sample_index": sample_index,
                        "question_id": sample.question_id,
                        "image_id": sample.image_id,
                        "question": sample.question,
                        "answer": sample.short_answer,
                        "answer_type": sample.answer_type or "unknown",
                        "question_type": sample.question_type,
                        "answer_token_id": target_id,
                        "answer_token": target_token,
                        "variant_first_token_ids": variant_ids,
                        "variant_answers": variant_labels,
                        "variant_first_tokens": variant_tokens,
                        "answer_start": int(encoded.answer_start),
                        "prediction_position": pred_pos,
                        "layer": int(layer),
                        "method": method,
                        "rank": metrics["rank"],
                        "log10_rank": metrics["log10_rank"],
                        "logprob": metrics["logprob"],
                        "top20_hit": metrics["top20_hit"],
                        "mrr": metrics["mrr"],
                        "variant_group_rank": metrics["variant_group_rank"],
                        "variant_group_log10_rank": metrics["variant_group_log10_rank"],
                        "variant_group_logprob": metrics["variant_group_logprob"],
                        "variant_group_top20_hit": metrics["variant_group_top20_hit"],
                        "variant_group_mrr": metrics["variant_group_mrr"],
                        "top1_id": metrics["top1_id"],
                        "top1_token": top1_token,
                    }
                )
    return records


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(int(record["layer"]), record["method"])].append(record)

    rows: list[dict[str, Any]] = []
    for (layer, method), vals in sorted(grouped.items()):
        rows.append(
            {
                "layer": layer,
                "method": method,
                "n": len(vals),
                "mean_log10_rank": mean(v["log10_rank"] for v in vals),
                "median_rank": median(v["rank"] for v in vals),
                "top20_rate": mean(1.0 if v["top20_hit"] else 0.0 for v in vals),
                "mean_mrr": mean(v["mrr"] for v in vals),
                "variant_group_mean_log10_rank": mean(
                    v["variant_group_log10_rank"] for v in vals
                ),
                "variant_group_median_rank": median(v["variant_group_rank"] for v in vals),
                "variant_group_top20_rate": mean(
                    1.0 if v["variant_group_top20_hit"] else 0.0 for v in vals
                ),
                "variant_group_mean_mrr": mean(v["variant_group_mrr"] for v in vals),
            }
        )
    return rows


def summarize_by_answer_type(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[
            (
                record.get("answer_type") or "unknown",
                int(record["layer"]),
                record["method"],
            )
        ].append(record)

    rows: list[dict[str, Any]] = []
    for (answer_type, layer, method), vals in sorted(grouped.items()):
        rows.append(
            {
                "answer_type": answer_type,
                "layer": layer,
                "method": method,
                "n": len(vals),
                "mean_log10_rank": mean(v["log10_rank"] for v in vals),
                "median_rank": median(v["rank"] for v in vals),
                "top20_rate": mean(1.0 if v["top20_hit"] else 0.0 for v in vals),
                "mean_mrr": mean(v["mrr"] for v in vals),
                "variant_group_mean_log10_rank": mean(
                    v["variant_group_log10_rank"] for v in vals
                ),
                "variant_group_median_rank": median(v["variant_group_rank"] for v in vals),
                "variant_group_top20_rate": mean(
                    1.0 if v["variant_group_top20_hit"] else 0.0 for v in vals
                ),
                "variant_group_mean_mrr": mean(v["variant_group_mrr"] for v in vals),
            }
        )
    return rows


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    records: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    answer_type_summary: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "per_sample_layer.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    (out_dir / "summary_by_layer.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "summary_by_answer_type.json").write_text(
        json.dumps(answer_type_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_table(out_dir / "summary_by_layer.csv", summary)
    write_table(out_dir / "summary_by_answer_type.csv", answer_type_summary)


def plot_summary(
    summary: list[dict[str, Any]],
    answer_type_summary: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary:
        by_method[row["method"]].append(row)
    for rows in by_method.values():
        rows.sort(key=lambda r: r["layer"])

    def line_plot(
        metric: str,
        ylabel: str,
        filename: str,
        *,
        log_y: bool = False,
        source: dict[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for method, rows in sorted((source or by_method).items()):
            ax.plot(
                [r["layer"] for r in rows],
                [r[metric] for r in rows],
                marker="o",
                label=method,
            )
        ax.set_xlabel("Layer")
        ax.set_ylabel(ylabel)
        if log_y:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=160)
        plt.close(fig)

    line_plot("mean_log10_rank", "Mean log10 answer rank", "mean_log10_rank_by_layer.png")
    line_plot("median_rank", "Median answer rank (1 = best)", "median_rank_by_layer.png", log_y=True)
    line_plot("top20_rate", "Top-20 hit rate", "top20_hit_rate_by_layer.png")
    line_plot("mean_mrr", "MRR", "mrr_by_layer.png")
    line_plot(
        "variant_group_mean_log10_rank",
        "Mean log10 answer-variant group rank",
        "variant_group_mean_log10_rank_by_layer.png",
    )
    line_plot(
        "variant_group_median_rank",
        "Median answer-variant group rank",
        "variant_group_median_rank_by_layer.png",
        log_y=True,
    )

    by_answer_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in answer_type_summary:
        key = f"{row['method']} {row['answer_type']}"
        by_answer_type[key].append(row)
    for rows in by_answer_type.values():
        rows.sort(key=lambda r: r["layer"])

    def answer_type_plot(metric: str, ylabel: str, filename: str, *, log_y: bool = False) -> None:
        fig, ax = plt.subplots(figsize=(9, 5))
        for label, rows in sorted(by_answer_type.items()):
            ax.plot(
                [r["layer"] for r in rows],
                [r[metric] for r in rows],
                marker="o",
                label=label,
            )
        ax.set_xlabel("Layer")
        ax.set_ylabel(ylabel)
        if log_y:
            ax.set_yscale("log")
        if "rate" in metric or metric.endswith("mrr"):
            ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=160)
        plt.close(fig)

    answer_type_plot(
        "mean_log10_rank",
        "Mean log10 answer rank",
        "answer_type_mean_log10_rank_by_layer.png",
    )
    answer_type_plot(
        "median_rank",
        "Median answer rank",
        "answer_type_median_rank_by_layer.png",
        log_y=True,
    )
    answer_type_plot(
        "top20_rate",
        "Top-20 hit rate",
        "answer_type_top20_hit_rate_by_layer.png",
    )
    answer_type_plot("mean_mrr", "MRR", "answer_type_mrr_by_layer.png")
    answer_type_plot(
        "variant_group_mean_log10_rank",
        "Mean log10 answer-variant group rank",
        "answer_type_variant_group_mean_log10_rank_by_layer.png",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--metadata", default=DEFAULT_METADATA)
    parser.add_argument("--lens", default=DEFAULT_LENS)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--layers", help="Comma-separated subset. Defaults to lens.source_layers.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--attn", default="sdpa")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    lens = load_lens_or_checkpoint(args.lens)
    layers = parse_layer_list(args.layers) or lens.source_layers
    samples = load_vqa_metadata(args.metadata, limit=args.n)

    model = Qwen3VLJLensModel.from_pretrained(
        args.model,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn,
    )
    records = evaluate(model, lens, samples, layers=layers)
    if not records:
        raise RuntimeError("no records produced")
    summary = summarize(records)
    answer_type_summary = summarize_by_answer_type(records)
    write_outputs(records, summary, answer_type_summary, out_dir)
    if not args.no_plots:
        plot_summary(summary, answer_type_summary, out_dir)
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "lens": str(args.lens),
                "lens_n_prompts": lens.n_prompts,
                "metadata": str(args.metadata),
                "n_samples": len(samples),
                "layers": layers,
                "files": [
                    "per_sample_layer.jsonl",
                    "summary_by_layer.json",
                    "summary_by_layer.csv",
                    "summary_by_answer_type.json",
                    "summary_by_answer_type.csv",
                    *(
                        []
                        if args.no_plots
                        else [
                            "mean_log10_rank_by_layer.png",
                            "median_rank_by_layer.png",
                            "top20_hit_rate_by_layer.png",
                            "mrr_by_layer.png",
                            "variant_group_mean_log10_rank_by_layer.png",
                            "variant_group_median_rank_by_layer.png",
                            "answer_type_mean_log10_rank_by_layer.png",
                            "answer_type_median_rank_by_layer.png",
                            "answer_type_top20_hit_rate_by_layer.png",
                            "answer_type_mrr_by_layer.png",
                            "answer_type_variant_group_mean_log10_rank_by_layer.png",
                        ]
                    ),
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
