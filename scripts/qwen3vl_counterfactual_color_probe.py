#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Counterfactual red/blue car color probe for Qwen3-VL J-lens.

The experiment keeps the object category fixed (car) and swaps only the visible
color. For each image/prompt pair it teacher-forces the expected one-word color
answer, then reads logits at ``answer_start - 1`` so the scored answer token is
not visible to the residual state being decoded.
"""
# ruff: noqa: E402,I001

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
    VQASample,
    parse_layer_list,
)


DEFAULT_MODEL = "local_models/Qwen3-VL-4B-Instruct"
DEFAULT_LENS = "runs/hf_upload_qwen3vl_jlens_vqav2_100/lens.pt"
DEFAULT_RED_IMAGE = "assets/red_car.png"
DEFAULT_BLUE_IMAGE = "assets/blue_car.png"
DEFAULT_OUT = "runs/qwen3vl_counterfactual_color_probe"
DEFAULT_PROMPTS = (
    "What color is the object?",
    "What color is the car?",
    "Describe the color of the main object.",
)
DEFAULT_TARGETS = ("red", "blue", "car")


def load_lens_or_checkpoint(path: str | Path) -> jlens.JacobianLens:
    """Load either a final ``lens.pt`` or a resumable ``lens.ckpt.pt``."""

    path = Path(path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if "J" in state:
        return jlens.JacobianLens(
            jacobians=state["J"],
            n_prompts=int(state["n_prompts"]),
            d_model=int(state["d_model"]),
        )
    if "jacobian_sum" not in state:
        raise ValueError(f"{path} is neither a lens file nor a fit checkpoint")
    n_done = int(state["n_done"])
    if n_done <= 0:
        raise ValueError(f"{path} has n_done={n_done}; no usable Jacobian")
    jacobians = {
        int(layer): tensor / n_done for layer, tensor in state["jacobian_sum"].items()
    }
    first = next(iter(jacobians.values()))
    return jlens.JacobianLens(
        jacobians=jacobians,
        n_prompts=n_done,
        d_model=int(first.shape[0]),
    )


def parse_csv(text: str | None, fallback: tuple[str, ...]) -> list[str]:
    raw = ",".join(fallback) if text is None else text
    values: list[str] = []
    seen: set[str] = set()
    for chunk in raw.replace("\n", ",").split(","):
        value = chunk.strip()
        if value and value not in seen:
            seen.add(value)
            values.append(value)
    if not values:
        raise ValueError("empty comma-separated argument")
    return values


def token_id_for_single_token(model: Qwen3VLJLensModel, text: str) -> int:
    ids = model.tokenizer(text, add_special_tokens=False).input_ids
    if len(ids) != 1:
        tokens = [
            model.tokenizer.decode(
                [int(token_id)],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            for token_id in ids
        ]
        raise ValueError(
            f"target {text!r} is not one tokenizer token: ids={ids}, tokens={tokens}"
        )
    return int(ids[0])


def decode_one(model: Qwen3VLJLensModel, token_id: int) -> str:
    return model.tokenizer.decode(
        [int(token_id)],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def score_token(
    model: Qwen3VLJLensModel,
    logits: torch.Tensor,
    token_id: int,
) -> dict[str, Any]:
    logits = logits.float().cpu()
    log_probs = torch.log_softmax(logits, dim=-1)
    token_logit = logits[token_id]
    rank = int((logits > token_logit).sum().item()) + 1
    topk = logits.topk(20).indices.tolist()
    top1_id = int(topk[0])
    return {
        "token_id": int(token_id),
        "token": decode_one(model, token_id),
        "rank": rank,
        "log10_rank": math.log10(rank),
        "logprob": float(log_probs[token_id].item()),
        "top1_hit": rank == 1,
        "top5_hit": rank <= 5,
        "top20_hit": rank <= 20,
        "top1_id": top1_id,
        "top1_token": decode_one(model, top1_id),
        "top20_tokens": [decode_one(model, int(t)) for t in topk],
    }


def make_samples(
    *,
    red_image: str | Path,
    blue_image: str | Path,
    prompts: list[str],
) -> list[dict[str, Any]]:
    specs = [
        ("red_car", Path(red_image), "red", "blue"),
        ("blue_car", Path(blue_image), "blue", "red"),
    ]
    rows: list[dict[str, Any]] = []
    sample_index = 0
    for image_label, image_file, expected_color, counter_color in specs:
        for prompt_index, prompt in enumerate(prompts):
            sample = VQASample(
                index=sample_index,
                question_id=10_000 + sample_index,
                image_id=1 if image_label == "red_car" else 2,
                image_file=image_file,
                question=prompt,
                short_answer=expected_color,
                answer_type="color",
                question_type="counterfactual_color",
                answers=(expected_color,),
            )
            rows.append(
                {
                    "sample": sample,
                    "image_label": image_label,
                    "prompt_index": prompt_index,
                    "prompt": prompt,
                    "expected_color": expected_color,
                    "counter_color": counter_color,
                    "object": "car",
                }
            )
            sample_index += 1
    return rows


def evaluate(
    *,
    model: Qwen3VLJLensModel,
    lens: jlens.JacobianLens,
    sample_specs: list[dict[str, Any]],
    targets: list[str],
    layers: list[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target_ids = {target: token_id_for_single_token(model, target) for target in targets}
    target_tokens = {target: decode_one(model, token_id) for target, token_id in target_ids.items()}
    missing = sorted(set(layers) - set(lens.source_layers))
    if missing:
        raise ValueError(f"layers {missing} are not in lens.source_layers={lens.source_layers}")

    final_layer = model.n_layers - 1
    record_at = sorted({*layers, final_layer})
    records: list[dict[str, Any]] = []
    no_leakage_checks: list[dict[str, Any]] = []

    for spec_index, spec in enumerate(sample_specs):
        sample: VQASample = spec["sample"]
        print(
            (
                f"[eval] {spec_index + 1}/{len(sample_specs)} "
                f"{spec['image_label']} prompt={spec['prompt_index']} "
                f"answer={spec['expected_color']!r}"
            ),
            file=sys.stderr,
            flush=True,
        )
        encoded = model.encode_sample(sample)
        if not encoded.answer_token_ids:
            raise ValueError(f"sample {sample.index} has no answer tokens")
        pred_pos = encoded.answer_prediction_positions[0]
        if pred_pos != encoded.answer_start - 1:
            raise AssertionError(
                f"prediction position {pred_pos} does not precede answer_start "
                f"{encoded.answer_start}"
            )
        answer_first_id = int(encoded.answer_token_ids[0])
        expected_id = target_ids[spec["expected_color"]]
        if answer_first_id != expected_id:
            raise AssertionError(
                f"teacher answer token {answer_first_id}={decode_one(model, answer_first_id)!r} "
                f"does not match expected target {spec['expected_color']!r} id={expected_id}"
            )
        actual_id = int(encoded.inputs["input_ids"][0, encoded.answer_start])
        if actual_id != answer_first_id:
            raise AssertionError(
                f"input_ids[answer_start]={actual_id} does not match answer token "
                f"{answer_first_id}"
            )
        no_leakage_checks.append(
            {
                "sample_index": int(sample.index),
                "image_label": spec["image_label"],
                "prompt_index": int(spec["prompt_index"]),
                "answer_start": int(encoded.answer_start),
                "prediction_position": int(pred_pos),
                "scored_token_position": int(encoded.answer_start),
                "answer_first_token_id": answer_first_id,
                "answer_first_token": decode_one(model, answer_first_id),
                "passes": bool(pred_pos == encoded.answer_start - 1),
            }
        )

        inputs = model.batch_to_device(encoded.inputs)
        with torch.no_grad(), Qwen3VLResidualRecorder(model.layers, at=record_at) as recorder:
            model.forward_model(inputs)

        logits_by_method_layer: list[tuple[str, int, torch.Tensor]] = []
        for layer in layers:
            residual = recorder.activations[layer][0, pred_pos].float()
            logits_by_method_layer.append(
                ("vanilla", int(layer), model.unembed(residual).detach().cpu())
            )
            transported = lens.transport(residual, int(layer))
            logits_by_method_layer.append(
                ("jlens", int(layer), model.unembed(transported).detach().cpu())
            )
        final_residual = recorder.activations[final_layer][0, pred_pos].float()
        logits_by_method_layer.append(
            ("model_final", int(final_layer), model.unembed(final_residual).detach().cpu())
        )

        for method, layer, logits in logits_by_method_layer:
            for target in targets:
                scored = score_token(model, logits, target_ids[target])
                records.append(
                    {
                        "sample_index": int(sample.index),
                        "image_label": spec["image_label"],
                        "image_file": str(sample.image_file),
                        "prompt_index": int(spec["prompt_index"]),
                        "prompt": spec["prompt"],
                        "expected_color": spec["expected_color"],
                        "counter_color": spec["counter_color"],
                        "object": spec["object"],
                        "teacher_answer": sample.short_answer,
                        "answer_start": int(encoded.answer_start),
                        "prediction_position": int(pred_pos),
                        "method": method,
                        "layer": int(layer),
                        "target": target,
                        **scored,
                    }
                )

    metadata = {
        "targets": targets,
        "target_ids": target_ids,
        "target_tokens": target_tokens,
        "layers": layers,
        "final_layer": final_layer,
        "lens_n_prompts": lens.n_prompts,
        "no_leakage_checks": no_leakage_checks,
    }
    return records, metadata


def rows_by_key(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    return grouped


def summarize_target_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = rows_by_key(records, ("method", "layer", "image_label", "target"))
    out: list[dict[str, Any]] = []
    for (method, layer, image_label, target), vals in sorted(grouped.items()):
        out.append(
            {
                "method": method,
                "layer": int(layer),
                "image_label": image_label,
                "target": target,
                "n": len(vals),
                "mean_logprob": mean(v["logprob"] for v in vals),
                "mean_log10_rank": mean(v["log10_rank"] for v in vals),
                "median_rank": median(v["rank"] for v in vals),
                "top1_rate": mean(1.0 if v["top1_hit"] else 0.0 for v in vals),
                "top5_rate": mean(1.0 if v["top5_hit"] else 0.0 for v in vals),
                "top20_rate": mean(1.0 if v["top20_hit"] else 0.0 for v in vals),
            }
        )
    return out


def summarize_color_margin(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_case: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in records:
        key = (row["method"], int(row["layer"]), int(row["sample_index"]))
        by_case[key][row["target"]] = row

    case_rows: list[dict[str, Any]] = []
    for (method, layer, sample_index), by_target in sorted(by_case.items()):
        red = by_target.get("red")
        blue = by_target.get("blue")
        car = by_target.get("car")
        if red is None or blue is None:
            continue
        base = red
        expected = by_target[base["expected_color"]]
        counter = by_target[base["counter_color"]]
        case_rows.append(
            {
                "method": method,
                "layer": int(layer),
                "sample_index": int(sample_index),
                "image_label": base["image_label"],
                "prompt_index": int(base["prompt_index"]),
                "prompt": base["prompt"],
                "expected_color": base["expected_color"],
                "counter_color": base["counter_color"],
                "expected_rank": int(expected["rank"]),
                "counter_rank": int(counter["rank"]),
                "car_rank": int(car["rank"]) if car else None,
                "expected_logprob": float(expected["logprob"]),
                "counter_logprob": float(counter["logprob"]),
                "car_logprob": float(car["logprob"]) if car else None,
                "color_margin_logprob": float(expected["logprob"] - counter["logprob"]),
                "expected_beats_counter": bool(expected["rank"] < counter["rank"]),
                "expected_top20": bool(expected["top20_hit"]),
                "counter_top20": bool(counter["top20_hit"]),
                "car_top20": bool(car["top20_hit"]) if car else None,
            }
        )

    grouped = rows_by_key(case_rows, ("method", "layer"))
    summary_rows: list[dict[str, Any]] = []
    for (method, layer), vals in sorted(grouped.items()):
        car_vals = [v["car_logprob"] for v in vals if v["car_logprob"] is not None]
        car_ranks = [v["car_rank"] for v in vals if v["car_rank"] is not None]
        summary_rows.append(
            {
                "method": method,
                "layer": int(layer),
                "n": len(vals),
                "mean_color_margin_logprob": mean(v["color_margin_logprob"] for v in vals),
                "median_color_margin_logprob": median(
                    v["color_margin_logprob"] for v in vals
                ),
                "expected_beats_counter_rate": mean(
                    1.0 if v["expected_beats_counter"] else 0.0 for v in vals
                ),
                "expected_top20_rate": mean(1.0 if v["expected_top20"] else 0.0 for v in vals),
                "counter_top20_rate": mean(1.0 if v["counter_top20"] else 0.0 for v in vals),
                "expected_median_rank": median(v["expected_rank"] for v in vals),
                "counter_median_rank": median(v["counter_rank"] for v in vals),
                "car_median_rank": median(car_ranks) if car_ranks else None,
                "mean_expected_logprob": mean(v["expected_logprob"] for v in vals),
                "mean_counter_logprob": mean(v["counter_logprob"] for v in vals),
                "mean_car_logprob": mean(car_vals) if car_vals else None,
            }
        )
    return case_rows, summary_rows


def summarize_counterfactual_specificity(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    indexed: dict[tuple[str, int, int, str, str], dict[str, Any]] = {}
    for row in records:
        indexed[
            (
                row["method"],
                int(row["layer"]),
                int(row["prompt_index"]),
                row["image_label"],
                row["target"],
            )
        ] = row

    methods_layers_prompts = sorted(
        {
            (row["method"], int(row["layer"]), int(row["prompt_index"]), row["prompt"])
            for row in records
        }
    )
    contrast_rows: list[dict[str, Any]] = []
    for method, layer, prompt_index, prompt in methods_layers_prompts:
        rr = indexed.get((method, layer, prompt_index, "red_car", "red"))
        rb = indexed.get((method, layer, prompt_index, "blue_car", "red"))
        bb = indexed.get((method, layer, prompt_index, "blue_car", "blue"))
        br = indexed.get((method, layer, prompt_index, "red_car", "blue"))
        car_red = indexed.get((method, layer, prompt_index, "red_car", "car"))
        car_blue = indexed.get((method, layer, prompt_index, "blue_car", "car"))
        if None in (rr, rb, bb, br):
            continue
        red_specificity = float(rr["logprob"] - rb["logprob"])
        blue_specificity = float(bb["logprob"] - br["logprob"])
        contrast_rows.append(
            {
                "method": method,
                "layer": int(layer),
                "prompt_index": int(prompt_index),
                "prompt": prompt,
                "red_target_specificity": red_specificity,
                "blue_target_specificity": blue_specificity,
                "mean_color_specificity": (red_specificity + blue_specificity) / 2.0,
                "red_target_rank_on_red_car": int(rr["rank"]),
                "red_target_rank_on_blue_car": int(rb["rank"]),
                "blue_target_rank_on_blue_car": int(bb["rank"]),
                "blue_target_rank_on_red_car": int(br["rank"]),
                "car_logprob_red_car": float(car_red["logprob"]) if car_red else None,
                "car_logprob_blue_car": float(car_blue["logprob"]) if car_blue else None,
                "car_logprob_delta_red_minus_blue": (
                    float(car_red["logprob"] - car_blue["logprob"])
                    if car_red and car_blue
                    else None
                ),
            }
        )

    grouped = rows_by_key(contrast_rows, ("method", "layer"))
    summary_rows: list[dict[str, Any]] = []
    for (method, layer), vals in sorted(grouped.items()):
        summary_rows.append(
            {
                "method": method,
                "layer": int(layer),
                "n_prompts": len(vals),
                "mean_red_target_specificity": mean(
                    v["red_target_specificity"] for v in vals
                ),
                "mean_blue_target_specificity": mean(
                    v["blue_target_specificity"] for v in vals
                ),
                "mean_color_specificity": mean(
                    v["mean_color_specificity"] for v in vals
                ),
                "red_specificity_positive_rate": mean(
                    1.0 if v["red_target_specificity"] > 0 else 0.0 for v in vals
                ),
                "blue_specificity_positive_rate": mean(
                    1.0 if v["blue_target_specificity"] > 0 else 0.0 for v in vals
                ),
            }
        )
    return contrast_rows, summary_rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_outputs(
    *,
    out_dir: Path,
    margin_summary: list[dict[str, Any]],
    target_summary: list[dict[str, Any]],
    specificity_summary: list[dict[str, Any]],
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def method_rows(rows: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
        selected = [row for row in rows if row["method"] == method]
        selected.sort(key=lambda row: int(row["layer"]))
        return selected

    def line_for_method(rows: list[dict[str, Any]], metric: str, label: str) -> None:
        if not rows:
            return
        plt.plot(
            [int(row["layer"]) for row in rows],
            [row[metric] for row in rows],
            marker="o",
            linewidth=1.8,
            label=label,
        )

    plt.figure(figsize=(9, 5))
    for method in ("vanilla", "jlens", "model_final"):
        line_for_method(
            method_rows(margin_summary, method),
            "mean_color_margin_logprob",
            method,
        )
    plt.axhline(0, color="black", linewidth=1, alpha=0.4)
    plt.xlabel("Layer")
    plt.ylabel("Expected color logprob - counter color logprob")
    plt.title("Color margin by layer")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "color_margin_by_layer.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    for method in ("vanilla", "jlens", "model_final"):
        line_for_method(
            method_rows(margin_summary, method),
            "expected_beats_counter_rate",
            method,
        )
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Layer")
    plt.ylabel("Fraction of prompts where expected color rank is better")
    plt.title("Expected color win rate by layer")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "color_win_rate_by_layer.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    for method in ("vanilla", "jlens", "model_final"):
        line_for_method(
            method_rows(specificity_summary, method),
            "mean_color_specificity",
            method,
        )
    plt.axhline(0, color="black", linewidth=1, alpha=0.4)
    plt.xlabel("Layer")
    plt.ylabel("Same-color image specificity")
    plt.title("Counterfactual color specificity by layer")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "counterfactual_specificity_by_layer.png", dpi=180)
    plt.close()

    jlens_target = [
        row
        for row in target_summary
        if row["method"] == "jlens" and row["target"] in {"red", "blue", "car"}
    ]
    line_specs = [
        ("red_car", "red", "red target on red car", "#d62728", "-"),
        ("red_car", "blue", "blue target on red car", "#1f77b4", "--"),
        ("blue_car", "blue", "blue target on blue car", "#1f77b4", "-"),
        ("blue_car", "red", "red target on blue car", "#d62728", "--"),
        ("red_car", "car", "car target on red car", "#555555", ":"),
        ("blue_car", "car", "car target on blue car", "#999999", ":"),
    ]
    for metric, ylabel, filename, log_y in (
        ("mean_logprob", "Mean logprob", "jlens_target_logprob_by_layer.png", False),
        ("median_rank", "Median rank", "jlens_target_rank_by_layer.png", True),
    ):
        plt.figure(figsize=(10, 5.5))
        for image_label, target, label, color, style in line_specs:
            rows = [
                row
                for row in jlens_target
                if row["image_label"] == image_label and row["target"] == target
            ]
            rows.sort(key=lambda row: int(row["layer"]))
            if not rows:
                continue
            plt.plot(
                [int(row["layer"]) for row in rows],
                [row[metric] for row in rows],
                marker="o",
                linewidth=1.6,
                linestyle=style,
                color=color,
                label=label,
            )
        if log_y:
            plt.yscale("log")
        plt.xlabel("Layer")
        plt.ylabel(ylabel)
        plt.title(f"J-lens {ylabel.lower()} for red/blue/car")
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=180)
        plt.close()


def write_outputs(
    *,
    out_dir: Path,
    records: list[dict[str, Any]],
    metadata: dict[str, Any],
    no_plots: bool,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    target_summary = summarize_target_records(records)
    margin_cases, margin_summary = summarize_color_margin(records)
    specificity_cases, specificity_summary = summarize_counterfactual_specificity(records)

    write_jsonl(out_dir / "color_probe_records.jsonl", records)
    write_csv(out_dir / "target_summary_by_layer.csv", target_summary)
    write_csv(out_dir / "color_margin_cases.csv", margin_cases)
    write_csv(out_dir / "color_margin_by_layer.csv", margin_summary)
    write_csv(out_dir / "counterfactual_specificity_cases.csv", specificity_cases)
    write_csv(out_dir / "counterfactual_specificity_by_layer.csv", specificity_summary)
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "target_summary_by_layer": target_summary,
                "color_margin_by_layer": margin_summary,
                "counterfactual_specificity_by_layer": specificity_summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if not no_plots:
        plot_outputs(
            out_dir=out_dir,
            margin_summary=margin_summary,
            target_summary=target_summary,
            specificity_summary=specificity_summary,
        )
    return {
        "records": len(records),
        "target_summary_rows": len(target_summary),
        "margin_case_rows": len(margin_cases),
        "margin_summary_rows": len(margin_summary),
        "specificity_case_rows": len(specificity_cases),
        "specificity_summary_rows": len(specificity_summary),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--lens", default=DEFAULT_LENS)
    parser.add_argument("--red-image", default=DEFAULT_RED_IMAGE)
    parser.add_argument("--blue-image", default=DEFAULT_BLUE_IMAGE)
    parser.add_argument("--prompts", help="Comma-separated prompt list.")
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    parser.add_argument("--layers", help="Comma-separated subset. Defaults to lens layers.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--attn", default="sdpa")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    prompts = parse_csv(args.prompts, DEFAULT_PROMPTS)
    targets = parse_csv(args.targets, DEFAULT_TARGETS)
    for required in ("red", "blue", "car"):
        if required not in targets:
            raise ValueError(f"--targets must include {required!r}")

    lens = load_lens_or_checkpoint(args.lens)
    layers = parse_layer_list(args.layers) or lens.source_layers
    model = Qwen3VLJLensModel.from_pretrained(
        args.model,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn,
    )
    sample_specs = make_samples(
        red_image=args.red_image,
        blue_image=args.blue_image,
        prompts=prompts,
    )
    records, metadata = evaluate(
        model=model,
        lens=lens,
        sample_specs=sample_specs,
        targets=targets,
        layers=layers,
    )
    metadata.update(
        {
            "model": str(args.model),
            "lens": str(args.lens),
            "red_image": str(args.red_image),
            "blue_image": str(args.blue_image),
            "prompts": prompts,
            "out_dir": str(out_dir),
        }
    )
    summary = write_outputs(
        out_dir=out_dir,
        records=records,
        metadata=metadata,
        no_plots=args.no_plots,
    )
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "summary": summary,
                "layers": layers,
                "targets": targets,
                "prompts": prompts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
