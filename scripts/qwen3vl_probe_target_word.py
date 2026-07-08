#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Probe where target phrases appear in vanilla and J-lens readouts.

The readout at sequence position ``p`` predicts token ``p + 1``. Multi-token
targets are scored as contiguous phrases: each tokenizer token is scored at its
own consecutive prediction position, phrase logprob is the sum of component
logprobs, and phrase rank is the worst component-token rank.
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

DEFAULT_MODEL = "/home/dkx/Projects/VLM/qwen3-vl/Qwen3-VL-4B-Instruct-ckpt"
DEFAULT_IMAGE = "/home/dkx/Projects/jacobian-lens/assets/image.png"
DEFAULT_LENS = (
    "/home/dkx/Projects/jacobian-lens/runs/"
    "qwen3vl_jlens_fit100_stride2/lens.ckpt.pt"
)
DEFAULT_OUT = (
    "/home/dkx/Projects/jacobian-lens/runs/"
    "qwen3vl_image1_multi_targets_metaphor_phrase"
)
DEFAULT_TARGETS = "禁止,色禽,鸟,鹦鹉,黄色,成人,淫秽,情色,低俗"


def load_lens_or_checkpoint(path: str | Path) -> jlens.JacobianLens:
    """Load either a final lens file or an in-progress fit checkpoint."""

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
        raise ValueError(f"{path} has n_done={n_done}; no usable Jacobian yet")
    jacobians = {
        int(layer): tensor / n_done for layer, tensor in state["jacobian_sum"].items()
    }
    first = next(iter(jacobians.values()))
    return jlens.JacobianLens(
        jacobians=jacobians,
        n_prompts=n_done,
        d_model=int(first.shape[0]),
    )


def decode_one(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode(
        [int(token_id)],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_targets(text: str | None, fallback: str) -> list[str]:
    """Parse comma/newline/whitespace separated target phrases."""

    raw = fallback if text is None else text
    targets: list[str] = []
    seen: set[str] = set()
    for chunk in raw.replace("\n", ",").replace("，", ",").split(","):
        for part in chunk.split():
            target = part.strip()
            if target and target not in seen:
                seen.add(target)
                targets.append(target)
    if not targets:
        raise ValueError("no targets provided")
    return targets


def build_target_specs(model: Qwen3VLJLensModel, targets: list[str]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for index, target in enumerate(targets):
        token_ids = [
            int(t) for t in model.tokenizer(target, add_special_tokens=False).input_ids
        ]
        if not token_ids:
            raise ValueError(f"target {target!r} produced no tokenizer ids")
        specs.append(
            {
                "target_index": index,
                "target": target,
                "target_token_ids": token_ids,
                "target_tokens": [decode_one(model.tokenizer, t) for t in token_ids],
                "target_first_token_id": token_ids[0],
                "target_first_token": decode_one(model.tokenizer, token_ids[0]),
                "n_target_tokens": len(token_ids),
            }
        )
    return specs


def generate_answer(
    model: Qwen3VLJLensModel,
    *,
    image: str | Path,
    prompt: str,
    max_new_tokens: int,
) -> tuple[list[int], str]:
    """Greedy-generate the assistant continuation for the image prompt."""

    inputs_cpu = model.encode_generation_prompt(image, prompt)
    inputs = model.batch_to_device(inputs_cpu)
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_cache": True,
    }
    if model.tokenizer.eos_token_id is not None:
        kwargs["pad_token_id"] = model.tokenizer.eos_token_id
    with torch.no_grad():
        generated = model.model.generate(**inputs, **kwargs)
    prompt_len = int(inputs["input_ids"].shape[-1])
    new_ids = [int(t) for t in generated[0, prompt_len:].detach().cpu().tolist()]
    text = model.tokenizer.decode(
        new_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    return new_ids, text


def find_subsequence_starts(sequence: list[int], needle: list[int]) -> list[int]:
    if not needle or len(needle) > len(sequence):
        return []
    width = len(needle)
    return [
        idx
        for idx in range(0, len(sequence) - width + 1)
        if sequence[idx : idx + width] == needle
    ]


def token_phase(index: int, *, answer_start: int, answer_end: int) -> str:
    if index < answer_start:
        return "prompt"
    if index < answer_end:
        return "assistant_answer"
    return "after_answer_special"


def score_target_phrase(
    logits: torch.Tensor,
    *,
    target_ids: list[int],
    top_k: int,
) -> tuple[list[dict[str, Any]], list[list[int]]]:
    """Return phrase-level metrics and first-token top-k ids.

    Exact rank over all token sequences would require enumerating ``vocab_size``
    to the phrase length. This proxy requires every component token in the
    phrase to rank well at its consecutive prediction position.
    """

    logits = logits.float()
    if not target_ids:
        raise ValueError("target_ids must not be empty")
    phrase_len = len(target_ids)
    n_starts = int(logits.shape[0]) - phrase_len + 1
    if n_starts <= 0:
        return [], []

    log_denoms = torch.logsumexp(logits, dim=-1)
    top1_all = logits.argmax(dim=-1)
    first_topk_ids = [
        [int(t) for t in row.tolist()]
        for row in logits[:n_starts].topk(top_k, dim=-1).indices
    ]

    rank_columns: list[torch.Tensor] = []
    logprob_columns: list[torch.Tensor] = []
    top1_columns: list[torch.Tensor] = []
    for offset, target_id in enumerate(target_ids):
        step_logits = logits[offset : offset + n_starts]
        target_logits = step_logits[:, int(target_id)]
        ranks = (step_logits > target_logits[:, None]).sum(dim=-1) + 1
        logprobs = target_logits - log_denoms[offset : offset + n_starts]
        rank_columns.append(ranks.to(torch.int64))
        logprob_columns.append(logprobs)
        top1_columns.append(top1_all[offset : offset + n_starts].to(torch.int64))

    rank_tensor = torch.stack(rank_columns, dim=1)
    logprob_tensor = torch.stack(logprob_columns, dim=1)
    top1_tensor = torch.stack(top1_columns, dim=1)
    phrase_ranks = rank_tensor.max(dim=1).values
    phrase_logprobs = logprob_tensor.sum(dim=1)
    phrase_topk_hits = (rank_tensor <= top_k).all(dim=1)

    rows = []
    for pos in range(n_starts):
        component_ranks = [int(t) for t in rank_tensor[pos].tolist()]
        component_logprobs = [float(t) for t in logprob_tensor[pos].tolist()]
        component_top1_ids = [int(t) for t in top1_tensor[pos].tolist()]
        phrase_rank = int(phrase_ranks[pos].item())
        rows.append(
            {
                "rank": phrase_rank,
                "log10_rank": math.log10(phrase_rank),
                "logprob": float(phrase_logprobs[pos].item()),
                f"top{top_k}_hit": bool(phrase_topk_hits[pos].item()),
                "mrr": 1.0 / phrase_rank,
                "top1_id": component_top1_ids[0],
                "component_ranks": component_ranks,
                "component_logprobs": component_logprobs,
                "component_top1_ids": component_top1_ids,
                "phrase_score_mode": "contiguous_all_tokens",
                "phrase_rank_definition": "max_component_token_rank",
                "phrase_logprob_definition": "sum_component_token_logprobs",
            }
        )
    return rows, first_topk_ids


def build_position_rows(
    *,
    model: Qwen3VLJLensModel,
    token_ids: list[int],
    answer_start: int,
    answer_end: int,
    target_starts_by_name: dict[str, list[int]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pos in range(len(token_ids)):
        phrase_starts = [
            target for target, starts in target_starts_by_name.items() if pos in set(starts)
        ]
        prediction_targets = [
            target
            for target, starts in target_starts_by_name.items()
            if pos in {start - 1 for start in starts if start > 0}
        ]
        rows.append(
            {
                "position": pos,
                "token_id": int(token_ids[pos]),
                "token": decode_one(model.tokenizer, token_ids[pos]),
                "phase": token_phase(pos, answer_start=answer_start, answer_end=answer_end),
                "target_phrase_starts": phrase_starts,
                "target_prediction_position_for": prediction_targets,
            }
        )
    return rows


def evaluate(
    *,
    model: Qwen3VLJLensModel,
    lens: jlens.JacobianLens,
    image: str | Path,
    prompt: str,
    targets: list[str],
    answer_text: str | None,
    max_new_tokens: int,
    layers: list[int],
    top_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    target_specs = build_target_specs(model, targets)

    generated_ids: list[int] = []
    if answer_text is None:
        generated_ids, answer_text = generate_answer(
            model, image=image, prompt=prompt, max_new_tokens=max_new_tokens
        )
    if not answer_text:
        raise ValueError("empty assistant answer; pass --answer-text to force a continuation")

    sample = VQASample(
        index=0,
        question_id=0,
        image_id=0,
        image_file=Path(image),
        question=prompt,
        short_answer=answer_text,
    )
    encoded = model.encode_sample(sample)
    token_ids = [int(t) for t in encoded.inputs["input_ids"][0].tolist()]
    target_starts_by_name = {
        spec["target"]: find_subsequence_starts(token_ids, spec["target_token_ids"])
        for spec in target_specs
    }
    target_prediction_positions_by_name = {
        target: [start - 1 for start in starts if start > 0]
        for target, starts in target_starts_by_name.items()
    }
    target_token_positions_by_name = {
        spec["target"]: {
            pos
            for start in target_starts_by_name[spec["target"]]
            for pos in range(start, start + len(spec["target_token_ids"]))
        }
        for spec in target_specs
    }
    for target, starts in target_starts_by_name.items():
        positions = target_prediction_positions_by_name[target]
        if any(pos >= start for pos, start in zip(positions, starts, strict=False)):
            raise AssertionError(
                f"target {target!r} prediction position is not before occurrence"
            )

    missing = sorted(set(layers) - set(lens.source_layers))
    if missing:
        raise ValueError(f"layers {missing} missing from lens.source_layers={lens.source_layers}")

    position_rows = build_position_rows(
        model=model,
        token_ids=token_ids,
        answer_start=encoded.answer_start,
        answer_end=encoded.answer_end,
        target_starts_by_name=target_starts_by_name,
    )
    readout_positions = list(range(0, max(0, encoded.seq_len - 1)))
    inputs = model.batch_to_device(encoded.inputs)
    with torch.no_grad(), Qwen3VLResidualRecorder(model.layers, at=layers) as recorder:
        model.forward_model(inputs)

    records: list[dict[str, Any]] = []
    for layer in layers:
        residuals = recorder.activations[layer][0, readout_positions].float()
        method_logits = {
            "native": model.unembed(residuals).detach().cpu(),
            "jlens": model.unembed(lens.transport(residuals, layer)).detach().cpu(),
        }
        for method, logits in method_logits.items():
            for spec in target_specs:
                metric_rows, topk_ids = score_target_phrase(
                    logits, target_ids=spec["target_token_ids"], top_k=top_k
                )
                actual_positions = set(
                    target_prediction_positions_by_name[spec["target"]]
                )
                actual_token_positions = target_token_positions_by_name[spec["target"]]
                for idx, pos in enumerate(readout_positions[: len(metric_rows)]):
                    next_pos = pos + 1
                    phrase_readout_positions = [
                        int(pos + offset)
                        for offset in range(spec["n_target_tokens"])
                    ]
                    phrase_next_positions = [
                        int(next_pos + offset)
                        for offset in range(spec["n_target_tokens"])
                    ]
                    top_tokens = [decode_one(model.tokenizer, t) for t in topk_ids[idx]]
                    row = {
                        "method": method,
                        "layer": int(layer),
                        "target_index": spec["target_index"],
                        "target": spec["target"],
                        "target_token_ids": spec["target_token_ids"],
                        "target_tokens": spec["target_tokens"],
                        "target_first_token_id": spec["target_first_token_id"],
                        "target_first_token": spec["target_first_token"],
                        "n_target_tokens": spec["n_target_tokens"],
                        "position": int(pos),
                        "next_position": int(next_pos),
                        "phrase_readout_positions": phrase_readout_positions,
                        "phrase_next_positions": phrase_next_positions,
                        "position_token_id": int(token_ids[pos]),
                        "position_token": decode_one(model.tokenizer, token_ids[pos]),
                        "next_token_id": int(token_ids[next_pos]),
                        "next_token": decode_one(model.tokenizer, token_ids[next_pos]),
                        "phrase_next_token_ids": [
                            int(token_ids[p]) for p in phrase_next_positions
                        ],
                        "phrase_next_tokens": [
                            decode_one(model.tokenizer, token_ids[p])
                            for p in phrase_next_positions
                        ],
                        "position_phase": token_phase(
                            pos,
                            answer_start=encoded.answer_start,
                            answer_end=encoded.answer_end,
                        ),
                        "next_token_phase": token_phase(
                            next_pos,
                            answer_start=encoded.answer_start,
                            answer_end=encoded.answer_end,
                        ),
                        "phrase_readout_phases": [
                            token_phase(
                                p,
                                answer_start=encoded.answer_start,
                                answer_end=encoded.answer_end,
                            )
                            for p in phrase_readout_positions
                        ],
                        "phrase_next_token_phases": [
                            token_phase(
                                p,
                                answer_start=encoded.answer_start,
                                answer_end=encoded.answer_end,
                            )
                            for p in phrase_next_positions
                        ],
                        "is_actual_target_prediction_position": pos in actual_positions,
                        "is_actual_target_token_position": pos in actual_token_positions,
                        "is_leaky_target_window": any(
                            p in actual_token_positions for p in phrase_readout_positions
                        ),
                        "top_tokens": top_tokens,
                    }
                    row.update(metric_rows[idx])
                    row["top1_token"] = decode_one(model.tokenizer, row["top1_id"])
                    records.append(row)

    metadata = {
        "image": str(image),
        "prompt": prompt,
        "targets": targets,
        "target_specs": target_specs,
        "answer_text": answer_text,
        "generated_token_ids": generated_ids,
        "seq_len": encoded.seq_len,
        "answer_start": encoded.answer_start,
        "answer_end": encoded.answer_end,
        "actual_target_phrase_starts": target_starts_by_name,
        "actual_target_prediction_positions": target_prediction_positions_by_name,
        "actual_target_token_positions": {
            target: sorted(positions)
            for target, positions in target_token_positions_by_name.items()
        },
        "readout_position_semantics": "position p predicts token p+1",
        "phrase_scoring": {
            "unit": "full tokenizer phrase",
            "rank": "max component-token rank across consecutive prediction positions",
            "logprob": "sum of component-token logprobs",
            f"top{top_k}_hit": "true only if every component token is top-k",
        },
        "summary_excludes_actual_target_token_positions": True,
        "summary_excludes_leaky_phrase_windows": True,
        "off_by_one_leakage_safe": all(
            pos == start - 1
            for target, starts in target_starts_by_name.items()
            for pos, start in zip(
                target_prediction_positions_by_name[target],
                starts,
                strict=False,
            )
        ),
        "scoring_note": (
            "Each target phrase is scored as its full tokenizer sequence. "
            "Component token j is scored at readout position p+j, which "
            "predicts token p+j+1."
        ),
        "layers": layers,
        "top_k": top_k,
    }
    return records, position_rows, metadata


def summarize(records: list[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[
            (
                int(row["target_index"]),
                row["target"],
                row["method"],
                int(row["layer"]),
            )
        ].append(row)

    summary: list[dict[str, Any]] = []
    hit_key = f"top{top_k}_hit"
    for (target_index, target, method, layer), rows in sorted(grouped.items()):
        nonleaky_rows = [row for row in rows if not row["is_leaky_target_window"]]
        if not nonleaky_rows:
            raise ValueError(
                f"no non-leaky positions for target={target!r} method={method} layer={layer}"
            )
        best = min(nonleaky_rows, key=lambda r: (r["rank"], r["position"]))
        actual_rows = [r for r in rows if r["is_actual_target_prediction_position"]]
        best_actual = min(actual_rows, key=lambda r: r["rank"]) if actual_rows else None
        summary.append(
            {
                "target_index": target_index,
                "target": target,
                "target_tokens": best["target_tokens"],
                "target_first_token": best["target_first_token"],
                "target_first_token_id": best["target_first_token_id"],
                "n_target_tokens": best["n_target_tokens"],
                "score_unit": "full_tokenizer_phrase",
                "rank_definition": best["phrase_rank_definition"],
                "logprob_definition": best["phrase_logprob_definition"],
                "method": method,
                "layer": layer,
                "min_rank": best["rank"],
                "min_log10_rank": best["log10_rank"],
                "min_logprob": best["logprob"],
                "min_component_ranks": best["component_ranks"],
                "min_component_logprobs": best["component_logprobs"],
                "min_rank_position": best["position"],
                "min_rank_next_position": best["next_position"],
                "min_rank_next_token": best["next_token"],
                "min_rank_phrase_next_tokens": best["phrase_next_tokens"],
                "min_rank_next_token_phase": best["next_token_phase"],
                f"n_top{top_k}_hits": sum(1 for r in nonleaky_rows if r[hit_key]),
                "n_leaky_target_token_positions": sum(
                    1 for r in rows if r["is_actual_target_token_position"]
                ),
                "n_leaky_phrase_windows": sum(
                    1 for r in rows if r["is_leaky_target_window"]
                ),
                "actual_target_best_rank": None if best_actual is None else best_actual["rank"],
                "actual_target_best_layer_position": None
                if best_actual is None
                else best_actual["position"],
                "actual_target_best_logprob": None
                if best_actual is None
                else best_actual["logprob"],
                "actual_target_best_component_ranks": None
                if best_actual is None
                else best_actual["component_ranks"],
            }
        )
    return summary


def plot_outputs(
    *,
    records: list[dict[str, Any]],
    metadata: dict[str, Any],
    out_dir: Path,
    summary: list[dict[str, Any]],
    top_k: int,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    matplotlib.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "SimHei",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False

    positions = sorted({int(r["position"]) for r in records})
    layers = sorted({int(r["layer"]) for r in records})
    targets = [spec["target"] for spec in metadata["target_specs"]]
    pos_index = {pos: idx for idx, pos in enumerate(positions)}
    layer_index = {layer: idx for idx, layer in enumerate(layers)}
    target_index = {target: idx for idx, target in enumerate(targets)}

    for method in sorted({r["method"] for r in records}):
        matrix = np.full((len(targets), len(layers)), np.nan, dtype=float)
        hit_matrix = np.zeros((len(targets), len(layers)), dtype=float)
        for row in summary:
            if row["method"] != method:
                continue
            ti = target_index[row["target"]]
            li = layer_index[int(row["layer"])]
            matrix[ti, li] = float(row["min_log10_rank"])
            hit_matrix[ti, li] = float(row[f"n_top{top_k}_hits"])

        fig, ax = plt.subplots(figsize=(11, 4.8))
        im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis_r")
        ax.set_title(f"{method}: best phrase log10 rank over positions")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Target")
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, rotation=45)
        ax.set_yticks(range(len(targets)))
        ax.set_yticklabels(targets)
        fig.colorbar(
            im,
            ax=ax,
            label="min log10(max component rank), lower is better",
        )
        fig.tight_layout()
        fig.savefig(out_dir / f"{method}_target_layer_min_log10_rank.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 4.8))
        im = ax.imshow(hit_matrix, aspect="auto", interpolation="nearest", cmap="magma")
        ax.set_title(f"{method}: number of top-{top_k} positions")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Target")
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, rotation=45)
        ax.set_yticks(range(len(targets)))
        ax.set_yticklabels(targets)
        fig.colorbar(im, ax=ax, label=f"count of top-{top_k} readout positions")
        fig.tight_layout()
        fig.savefig(out_dir / f"{method}_target_layer_top{top_k}_counts.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        for target in targets:
            rows = [
                row
                for row in summary
                if row["method"] == method and row["target"] == target
            ]
            rows.sort(key=lambda r: int(r["layer"]))
            ax.plot(
                [int(r["layer"]) for r in rows],
                [int(r["min_rank"]) for r in rows],
                marker="o",
                label=target,
            )
        ax.set_title(f"{method}: target phrase-rank evolution")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Best max-component rank over positions")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"{method}_target_rank_evolution.png", dpi=180)
        plt.close(fig)

    per_target_dir = out_dir / "per_target_position_heatmaps"
    per_target_dir.mkdir(parents=True, exist_ok=True)
    for method in sorted({r["method"] for r in records}):
        for spec in metadata["target_specs"]:
            target = spec["target"]
            matrix = np.full((len(layers), len(positions)), np.nan, dtype=float)
            for row in records:
                if row["method"] != method or row["target"] != target:
                    continue
                if row["is_leaky_target_window"]:
                    continue
                matrix[
                    layer_index[int(row["layer"])],
                    pos_index[int(row["position"])],
                ] = row["log10_rank"]
            fig, ax = plt.subplots(figsize=(12, 5))
            im = ax.imshow(
                matrix,
                aspect="auto",
                interpolation="nearest",
                cmap="viridis_r",
            )
            ax.set_title(f"{method}: {target} phrase log10 rank")
            ax.set_xlabel("Readout position p (predicts p+1)")
            ax.set_ylabel("Layer")
            ax.set_yticks(range(len(layers)))
            ax.set_yticklabels(layers)
            answer_start = int(metadata["answer_start"])
            ax.axvline(answer_start - 1, color="white", linestyle="--", linewidth=1.0)
            for pos in metadata["actual_target_prediction_positions"][target]:
                ax.axvline(pos, color="red", linestyle="-", linewidth=1.0)
            fig.colorbar(
                im,
                ax=ax,
                label="log10(max component rank), lower is better",
            )
            fig.tight_layout()
            fig.savefig(
                per_target_dir
                / f"{spec['target_index']:02d}_{method}_{target}_log10_rank_heatmap.png",
                dpi=160,
            )
            plt.close(fig)

    for method in sorted({r["method"] for r in records}):
        matrix = np.full((len(layers), len(positions)), np.nan, dtype=float)
        for row in records:
            if row["method"] != method:
                continue
            if row["is_leaky_target_window"]:
                continue
            current = matrix[layer_index[int(row["layer"])], pos_index[int(row["position"])]]
            value = row["log10_rank"]
            if np.isnan(current) or value < current:
                matrix[layer_index[int(row["layer"])], pos_index[int(row["position"])]] = value
        fig, ax = plt.subplots(figsize=(12, 5))
        im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis_r")
        ax.set_title(f"{method}: best log10 rank across targets")
        ax.set_xlabel("Readout position p (predicts p+1)")
        ax.set_ylabel("Layer")
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(layers)
        answer_start = int(metadata["answer_start"])
        ax.axvline(answer_start - 1, color="white", linestyle="--", linewidth=1.0)
        fig.colorbar(
            im,
            ax=ax,
            label="log10(max component rank), lower is better",
        )
        fig.tight_layout()
        fig.savefig(out_dir / f"{method}_best_any_target_log10_rank_heatmap.png", dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for method in sorted({r["method"] for r in records}):
        y = []
        for layer in layers:
            vals = [
                r["rank"]
                for r in records
                if r["method"] == method and int(r["layer"]) == layer
            ]
            y.append(min(vals))
        ax.plot(layers, y, marker="o", label=method)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Best target rank over positions and targets")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "best_any_target_rank_by_layer.png", dpi=160)
    plt.close(fig)


def write_outputs(
    *,
    records: list[dict[str, Any]],
    position_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    out_dir: Path,
    top_k: int,
    no_plots: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(records, top_k=top_k)
    hit_key = f"top{top_k}_hit"
    hits = [
        row
        for row in records
        if row[hit_key] and not row["is_leaky_target_window"]
    ]

    (out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "summary_by_layer.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(out_dir / "target_tokenization.csv", metadata["target_specs"])
    write_csv(out_dir / "summary_by_layer.csv", summary)
    write_csv(out_dir / "position_tokens.csv", position_rows)
    write_csv(out_dir / "target_readout_records.csv", records)
    write_csv(out_dir / f"target_hits_top{top_k}.csv", hits)
    with (out_dir / "target_readout_records.jsonl").open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if not no_plots:
        plot_outputs(
            records=records,
            metadata=metadata,
            out_dir=out_dir,
            summary=summary,
            top_k=top_k,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--prompt", default="思考图片的背后隐喻。")
    parser.add_argument("--target", default="色情")
    parser.add_argument(
        "--targets",
        help=(
            "Comma/newline/whitespace separated target phrases. "
            "When omitted, --target is used."
        ),
    )
    parser.add_argument("--lens", default=DEFAULT_LENS)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--answer-text", help="Optional fixed assistant continuation.")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--layers", help="Comma-separated subset. Defaults to lens layers.")
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
    targets = parse_targets(args.targets, args.target)
    model = Qwen3VLJLensModel.from_pretrained(
        args.model,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn,
    )
    records, position_rows, metadata = evaluate(
        model=model,
        lens=lens,
        image=args.image,
        prompt=args.prompt,
        targets=targets,
        answer_text=args.answer_text,
        max_new_tokens=args.max_new_tokens,
        layers=layers,
        top_k=args.top_k,
    )
    metadata.update(
        {
            "lens": str(args.lens),
            "lens_n_prompts": lens.n_prompts,
            "model": str(args.model),
            "out_dir": str(out_dir),
        }
    )
    write_outputs(
        records=records,
        position_rows=position_rows,
        metadata=metadata,
        out_dir=out_dir,
        top_k=args.top_k,
        no_plots=args.no_plots,
    )
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "prompt": args.prompt,
                "targets": targets,
                "target_specs": metadata["target_specs"],
                "answer_text": metadata["answer_text"],
                "actual_target_phrase_starts": metadata["actual_target_phrase_starts"],
                "actual_target_prediction_positions": metadata[
                    "actual_target_prediction_positions"
                ],
                "actual_target_token_positions": metadata["actual_target_token_positions"],
                "phrase_scoring": metadata["phrase_scoring"],
                "summary_excludes_actual_target_token_positions": metadata[
                    "summary_excludes_actual_target_token_positions"
                ],
                "summary_excludes_leaky_phrase_windows": metadata[
                    "summary_excludes_leaky_phrase_windows"
                ],
                "off_by_one_leakage_safe": metadata["off_by_one_leakage_safe"],
                "lens_n_prompts": lens.n_prompts,
                "files": [
                    "run_metadata.json",
                    "target_tokenization.csv",
                    "position_tokens.csv",
                    "target_readout_records.csv",
                    "target_readout_records.jsonl",
                    "summary_by_layer.csv",
                    "summary_by_layer.json",
                    f"target_hits_top{args.top_k}.csv",
                    *(
                        []
                        if args.no_plots
                        else [
                            "native_target_layer_min_log10_rank.png",
                            "jlens_target_layer_min_log10_rank.png",
                            f"native_target_layer_top{args.top_k}_counts.png",
                            f"jlens_target_layer_top{args.top_k}_counts.png",
                            "native_target_rank_evolution.png",
                            "jlens_target_rank_evolution.png",
                            "native_best_any_target_log10_rank_heatmap.png",
                            "jlens_best_any_target_log10_rank_heatmap.png",
                            "best_any_target_rank_by_layer.png",
                            "per_target_position_heatmaps/",
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
