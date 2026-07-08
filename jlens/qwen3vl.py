# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL multimodal Jacobian-lens helpers.

The standard :mod:`jlens.hf` adapter calls the text module with ``input_ids``.
That is correct for text-only decoder LMs, but it bypasses Qwen-VL's image
embedding injection. This module keeps the public lens object unchanged while
adding a full multimodal path for Qwen3-VL:

``processor.apply_chat_template`` -> ``Qwen3VLForConditionalGeneration.model``
-> hooks on ``model.language_model.layers`` -> ``lm_head(norm(J_l @ h_l))``.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from jlens.lens import JacobianLens

logger = logging.getLogger(__name__)

PositionScope = Literal["all_nonfinal", "text_nonfinal", "answer_prediction"]


@dataclass(frozen=True)
class VQASample:
    """One VQAv2 image/question/short-answer example."""

    index: int
    question_id: int
    image_id: int
    image_file: Path
    question: str
    short_answer: str
    answer_type: str | None = None
    question_type: str | None = None
    answers: tuple[str, ...] = ()


@dataclass
class EncodedVQASample:
    """Tokenized teacher-forcing sample plus answer-token boundaries."""

    inputs: dict[str, torch.Tensor]
    answer_start: int
    answer_end: int
    answer_token_ids: list[int]

    @property
    def seq_len(self) -> int:
        return int(self.inputs["input_ids"].shape[-1])

    @property
    def answer_prediction_positions(self) -> list[int]:
        return list(range(max(0, self.answer_start - 1), max(0, self.answer_end - 1)))


def load_vqa_metadata(path: str | Path, *, limit: int | None = None) -> list[VQASample]:
    """Load the local ``metadata.jsonl`` produced for the VQAv2 subset."""

    path = Path(path)
    root = path.parent
    samples: list[VQASample] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            image_file = Path(record["image_file"])
            if not image_file.is_absolute():
                image_file = root / image_file
                if not image_file.exists() and root.name == "splits":
                    image_file = root.parent / record["image_file"]
            answers = tuple(a["answer"] for a in record.get("answers", ()))
            samples.append(
                VQASample(
                    index=int(record.get("index", len(samples))),
                    question_id=int(record["question_id"]),
                    image_id=int(record["image_id"]),
                    image_file=image_file,
                    question=record["question"],
                    short_answer=record["multiple_choice_answer"],
                    answer_type=record.get("answer_type"),
                    question_type=record.get("question_type"),
                    answers=answers,
                )
            )
            if limit is not None and len(samples) >= limit:
                break
    return samples


def write_vqa_splits(
    data_dir: str | Path,
    *,
    fit_count: int = 900,
    val_count: int = 100,
) -> tuple[Path, Path]:
    """Write deterministic first-900 / next-100 JSONL split files.

    The source subset was already sampled to 1000 examples. Keeping split order
    deterministic makes experiment artifacts easy to reproduce.
    """

    data_dir = Path(data_dir)
    lines = [
        line
        for line in (data_dir / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    required = fit_count + val_count
    if len(lines) < required:
        raise ValueError(f"need at least {required} records, found {len(lines)}")
    split_dir = data_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    fit_path = split_dir / f"fit_{fit_count}.jsonl"
    val_path = split_dir / f"val_{val_count}.jsonl"
    fit_path.write_text("\n".join(lines[:fit_count]) + "\n", encoding="utf-8")
    val_path.write_text(
        "\n".join(lines[fit_count : fit_count + val_count]) + "\n",
        encoding="utf-8",
    )
    return fit_path, val_path


class Qwen3VLJLensModel:
    """Full-multimodal Qwen3-VL model surface used by the Jacobian lens."""

    def __init__(self, model: Any, processor: Any) -> None:
        self.model = model
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.language_model = model.model.language_model
        self.layers = self.language_model.layers
        self.n_layers = int(model.config.text_config.num_hidden_layers)
        self.d_model = int(model.config.text_config.hidden_size)
        self._norm = self.language_model.norm
        self._lm_head = model.lm_head
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        *,
        device: str = "cuda",
        dtype: str | torch.dtype = "auto",
        attn_implementation: str | None = "sdpa",
    ) -> Qwen3VLJLensModel:
        """Load local Qwen3-VL weights and processor."""

        from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

        kwargs: dict[str, Any] = {
            "local_files_only": True,
            "dtype": dtype,
        }
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        model = Qwen3VLForConditionalGeneration.from_pretrained(str(model_path), **kwargs)
        if device:
            model.to(device)
        processor = Qwen3VLProcessor.from_pretrained(
            str(model_path), local_files_only=True
        )
        return cls(model, processor)

    @property
    def input_device(self) -> torch.device:
        return self._lm_head.weight.device

    @property
    def compute_dtype(self) -> torch.dtype:
        return self._lm_head.weight.dtype

    def _messages(self, sample: VQASample, *, include_answer: bool) -> list[dict]:
        image = self._load_image(sample.image_file)
        messages: list[dict] = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"Question: {sample.question}"},
                ],
            }
        ]
        if include_answer:
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": sample.short_answer}],
                }
            )
        return messages

    @staticmethod
    def _load_image(path: str | Path) -> Any:
        """Load local images with PIL to avoid backend-specific path decoding."""

        from PIL import Image

        with Image.open(path) as image:
            return image.convert("RGB").copy()

    def encode_sample(self, sample: VQASample) -> EncodedVQASample:
        """Build ``<image> user question assistant answer`` teacher-forcing input."""

        full = self.processor.apply_chat_template(
            self._messages(sample, include_answer=True),
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        prefix = self.processor.apply_chat_template(
            self._messages(sample, include_answer=False),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v for k, v in full.items() if torch.is_tensor(v)}
        full_ids = inputs["input_ids"][0].tolist()
        prefix_len = int(prefix["input_ids"].shape[-1])
        answer_ids = self.tokenizer(
            sample.short_answer, add_special_tokens=False
        ).input_ids
        tail = full_ids[prefix_len:]
        if answer_ids and tail[: len(answer_ids)] == answer_ids:
            answer_token_ids = list(answer_ids)
        else:
            # Some tokenizers normalize chat-template text differently from a
            # standalone answer. Fall back to the full span before <|im_end|>.
            eos_id = self.tokenizer.eos_token_id
            try:
                eos_rel = tail.index(eos_id)
            except ValueError:
                eos_rel = len(tail)
            answer_token_ids = tail[:eos_rel]
        return EncodedVQASample(
            inputs=inputs,
            answer_start=prefix_len,
            answer_end=prefix_len + len(answer_token_ids),
            answer_token_ids=answer_token_ids,
        )

    def encode_generation_prompt(self, image_file: str | Path, question: str) -> dict[str, torch.Tensor]:
        """Tokenize an image/question prompt ending at the assistant turn."""

        image = self._load_image(image_file)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"Question: {question}"},
                ],
            }
        ]
        encoded = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        return {k: v for k, v in encoded.items() if torch.is_tensor(v)}

    def batch_to_device(
        self, inputs: dict[str, torch.Tensor], *, repeat: int = 1
    ) -> dict[str, torch.Tensor]:
        """Move processor tensors to the model device, optionally repeating sample 0."""

        out: dict[str, torch.Tensor] = {}
        for key, value in inputs.items():
            tensor = value
            if repeat != 1:
                if key in {"input_ids", "attention_mask", "mm_token_type_ids"}:
                    tensor = tensor.expand(repeat, -1).clone()
                elif key in {"image_grid_thw", "video_grid_thw"}:
                    tensor = tensor.repeat(repeat, 1)
                elif key in {"pixel_values", "pixel_values_videos"}:
                    tensor = tensor.repeat(repeat, *([1] * (tensor.ndim - 1)))
            tensor = tensor.to(self.input_device)
            if tensor.is_floating_point():
                tensor = tensor.to(self.compute_dtype)
            out[key] = tensor
        return out

    def forward_model(self, inputs: dict[str, torch.Tensor]) -> Any:
        """Run Qwen3-VL's multimodal model body, including image injection."""

        return self.model.model(**inputs, use_cache=False)

    def forward_logits(self, inputs: dict[str, torch.Tensor], *, logits_to_keep: int = 0) -> Any:
        """Run the full conditional-generation module."""

        return self.model(**inputs, use_cache=False, logits_to_keep=logits_to_keep)

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        """Final-layer basis -> vocabulary logits."""

        residual = residual.to(self.input_device, dtype=self.compute_dtype)
        return self._lm_head(self._norm(residual))


class Qwen3VLResidualRecorder:
    """Capture actual post-layer residual states in Qwen3-VL's text stack.

    Qwen3-VL injects deepstack visual features after some decoder-layer modules,
    outside the modules themselves. A normal forward hook on ``layers[l]`` would
    miss that post-block injection. For layer ``l < n_layers - 1`` this recorder
    captures the input to ``layers[l + 1]``; for the final layer it captures the
    output of the final block, before the final norm.
    """

    def __init__(
        self,
        layers: Sequence[Any],
        *,
        at: Iterable[int],
        start_graph_at: int | None = None,
    ) -> None:
        self._layers = layers
        self._n_layers = len(layers)
        self._indices = sorted(set(at))
        self._start_graph_at = start_graph_at
        if start_graph_at is not None and start_graph_at not in self._indices:
            self._indices = sorted({*self._indices, start_graph_at})
        self.activations: dict[int, torch.Tensor] = {}
        self._handles: list[Any] = []

    def _record(self, index: int, tensor: torch.Tensor) -> None:
        if index == self._start_graph_at:
            tensor.requires_grad_(True)
        self.activations[index] = tensor

    def __enter__(self) -> Qwen3VLResidualRecorder:
        try:
            for index in self._indices:
                if not 0 <= index < self._n_layers:
                    raise ValueError(f"layer {index} out of range for {self._n_layers} layers")
                if index < self._n_layers - 1:
                    next_layer = self._layers[index + 1]

                    def pre_hook(module, inputs, idx=index):
                        self._record(idx, inputs[0])

                    self._handles.append(next_layer.register_forward_pre_hook(pre_hook))
                else:
                    layer = self._layers[index]

                    def hook(module, inputs, output, idx=index):
                        tensor = output if torch.is_tensor(output) else output[0]
                        self._record(idx, tensor)

                    self._handles.append(layer.register_forward_hook(hook))
        except Exception:
            for handle in self._handles:
                handle.remove()
            self._handles = []
            raise
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


def _check_layer_indices(
    source_layers: Sequence[int] | None, target_layer: int | None, n_layers: int
) -> tuple[list[int], int]:
    target = n_layers - 1 if target_layer is None else target_layer
    if target < 0:
        target += n_layers
    if not 0 <= target < n_layers:
        raise ValueError(f"target_layer={target_layer} out of range for {n_layers} layers")
    if source_layers is None:
        return list(range(target)), target
    sources = sorted({layer + n_layers if layer < 0 else layer for layer in source_layers})
    if not sources or sources[0] < 0 or sources[-1] >= n_layers:
        raise ValueError(f"source_layers {sorted(source_layers)} out of range for {n_layers} layers")
    if sources[-1] >= target:
        raise ValueError(
            f"source_layers must all be < target_layer={target}; got max={sources[-1]}"
        )
    return sources, target


def position_mask(
    encoded: EncodedVQASample,
    *,
    scope: PositionScope = "all_nonfinal",
    skip_first: int = 0,
) -> torch.Tensor:
    """Positions included in the Jacobian estimator."""

    seq_len = encoded.seq_len
    if skip_first < 0:
        raise ValueError(f"skip_first must be >= 0, got {skip_first}")
    mask = torch.zeros(seq_len, dtype=torch.bool)
    if scope == "answer_prediction":
        for pos in encoded.answer_prediction_positions:
            if 0 <= pos < seq_len - 1:
                mask[pos] = True
    else:
        mask[skip_first : seq_len - 1] = True
        if scope == "text_nonfinal":
            mm = encoded.inputs.get("mm_token_type_ids")
            if mm is None:
                raise ValueError("scope='text_nonfinal' needs mm_token_type_ids")
            mask &= mm[0].cpu().eq(0)
    if mask.sum() == 0:
        raise ValueError(f"no valid positions for scope={scope!r}, seq_len={seq_len}")
    return mask


def jacobian_for_vqa_sample(
    model: Qwen3VLJLensModel,
    sample: VQASample,
    source_layers: Sequence[int],
    *,
    target_layer: int | None = None,
    dim_batch: int = 8,
    position_scope: PositionScope = "all_nonfinal",
    skip_first: int = 0,
) -> tuple[dict[int, torch.Tensor], int, int]:
    """Compute per-layer ``J_l`` for one multimodal VQA teacher-forcing sample."""

    source_layers, target_layer = _check_layer_indices(
        source_layers, target_layer, model.n_layers
    )
    encoded = model.encode_sample(sample)
    mask = position_mask(encoded, scope=position_scope, skip_first=skip_first)
    valid_positions_cpu = mask.nonzero(as_tuple=True)[0]
    n_valid_positions = int(valid_positions_cpu.numel())
    d_model = model.d_model
    jacobians = {
        layer: torch.zeros(d_model, d_model, dtype=torch.float32)
        for layer in source_layers
    }
    n_passes = math.ceil(d_model / dim_batch)
    batch_inputs = model.batch_to_device(encoded.inputs, repeat=dim_batch)

    with (
        Qwen3VLResidualRecorder(
            model.layers,
            at=[*source_layers, target_layer],
            start_graph_at=min(source_layers),
        ) as recorder,
        torch.enable_grad(),
    ):
        model.forward_model(batch_inputs)
        target_activation = recorder.activations[target_layer]
        source_activations = [recorder.activations[layer] for layer in source_layers]
        valid_positions = valid_positions_cpu.to(target_activation.device)
        batch_indices = torch.arange(dim_batch, device=target_activation.device)
        cotangent = torch.zeros_like(target_activation)

        for pass_idx, dim_start in enumerate(range(0, d_model, dim_batch)):
            n_dims = min(dim_batch, d_model - dim_start)
            cotangent.zero_()
            cotangent[
                batch_indices[:n_dims, None],
                valid_positions[None, :],
                dim_start + batch_indices[:n_dims, None],
            ] = 1.0
            grads = torch.autograd.grad(
                outputs=target_activation,
                inputs=source_activations,
                grad_outputs=cotangent,
                retain_graph=(pass_idx < n_passes - 1),
            )
            for layer, grad in zip(source_layers, grads, strict=True):
                rows = grad[:n_dims, valid_positions.to(grad.device), :].float().mean(dim=1)
                jacobians[layer][dim_start : dim_start + n_dims] = rows.cpu()
            del grads
            if pass_idx % 50 == 0 or pass_idx == n_passes - 1:
                logger.debug(
                    "    sample %s pass %d/%d dims %d-%d",
                    sample.question_id,
                    pass_idx + 1,
                    n_passes,
                    dim_start,
                    dim_start + n_dims,
                )

    return jacobians, encoded.seq_len, n_valid_positions


def _atomic_save(obj: object, path: str | Path) -> None:
    path = Path(path)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def fit_vqa_jacobian_lens(
    model: Qwen3VLJLensModel,
    samples: Sequence[VQASample],
    *,
    source_layers: Sequence[int] | None = None,
    target_layer: int | None = None,
    dim_batch: int = 8,
    position_scope: PositionScope = "all_nonfinal",
    skip_first: int = 0,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int | None = 1,
    resume: bool = True,
) -> JacobianLens:
    """Fit a multimodal Jacobian lens on VQA teacher-forcing samples."""

    source_layers, target_layer = _check_layer_indices(
        source_layers, target_layer, model.n_layers
    )
    d_model = model.d_model
    checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
    logger.info(
        "fit_vqa: n_layers=%d d_model=%d source_layers=%d target=L%d samples=%d scope=%s",
        model.n_layers,
        d_model,
        len(source_layers),
        target_layer,
        len(samples),
        position_scope,
    )

    if resume and checkpoint_path is not None and checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        for key, expected in (
            ("source_layers", source_layers),
            ("target_layer", target_layer),
            ("position_scope", position_scope),
            ("skip_first", skip_first),
        ):
            if key in state and state[key] != expected:
                raise ValueError(
                    f"checkpoint {checkpoint_path} has {key}={state[key]!r}, "
                    f"not {expected!r}; pass resume=False to discard it"
                )
        jacobian_sum = state["jacobian_sum"]
        n_done = int(state["n_done"])
        next_idx = int(state["next_idx"])
        logger.info("  resuming checkpoint: next_idx=%d n_done=%d", next_idx, n_done)
    else:
        jacobian_sum = {
            layer: torch.zeros(d_model, d_model, dtype=torch.float32)
            for layer in source_layers
        }
        n_done = 0
        next_idx = 0

    def write_checkpoint() -> None:
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_save(
                {
                    "jacobian_sum": jacobian_sum,
                    "n_done": n_done,
                    "next_idx": next_idx,
                    "source_layers": source_layers,
                    "target_layer": target_layer,
                    "position_scope": position_scope,
                    "skip_first": skip_first,
                    "sample_question_ids": [s.question_id for s in samples],
                },
                checkpoint_path,
            )

    for sample_idx, sample in enumerate(samples):
        if sample_idx < next_idx:
            continue
        start = time.perf_counter()
        try:
            per_sample, seq_len, n_valid = jacobian_for_vqa_sample(
                model,
                sample,
                source_layers,
                target_layer=target_layer,
                dim_batch=dim_batch,
                position_scope=position_scope,
                skip_first=skip_first,
            )
        except Exception as exc:
            logger.warning("  skipping sample %d qid=%s: %s", sample_idx, sample.question_id, exc)
            next_idx = sample_idx + 1
            continue
        for layer in source_layers:
            jacobian_sum[layer] += per_sample[layer]
        n_done += 1
        next_idx = sample_idx + 1
        logger.info(
            "  sample %d/%d qid=%s seq_len=%d n_valid=%d %.1fs",
            sample_idx + 1,
            len(samples),
            sample.question_id,
            seq_len,
            n_valid,
            time.perf_counter() - start,
        )
        if checkpoint_every is not None and next_idx % checkpoint_every == 0:
            write_checkpoint()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_checkpoint()
    if n_done == 0:
        raise ValueError("no samples were successfully fitted")
    jacobian_mean = {layer: jacobian_sum[layer] / n_done for layer in source_layers}
    return JacobianLens(jacobians=jacobian_mean, n_prompts=n_done, d_model=d_model)


def token_ranks(logits: torch.Tensor, target_ids: Sequence[int]) -> list[int]:
    """Full-vocab 0-based ranks for ``target_ids`` in a single logit row."""

    sorted_idx = logits.float().argsort(dim=-1, descending=True)
    full_rank = torch.empty_like(sorted_idx)
    full_rank.scatter_(0, sorted_idx, torch.arange(logits.shape[-1], device=logits.device))
    return [int(full_rank[int(t)].item()) for t in target_ids]


def evaluate_answer_ranks(
    model: Qwen3VLJLensModel,
    lens: JacobianLens,
    samples: Sequence[VQASample],
    *,
    layers: Sequence[int] | None = None,
    use_jacobian: bool = True,
) -> list[dict[str, Any]]:
    """Rank held-out answer tokens at their teacher-forced prediction positions."""

    if layers is None:
        layers = lens.source_layers
    final_layer = model.n_layers - 1
    record_at = sorted(set(layers) | {final_layer})
    results: list[dict[str, Any]] = []

    for sample in samples:
        encoded = model.encode_sample(sample)
        positions = encoded.answer_prediction_positions
        if not positions:
            continue
        inputs = model.batch_to_device(encoded.inputs)
        with torch.no_grad(), Qwen3VLResidualRecorder(model.layers, at=record_at) as recorder:
            model.forward_model(inputs)
            activations = {i: recorder.activations[i].detach() for i in record_at}
        layer_ranks: dict[int, list[int]] = {}
        for layer in layers:
            residual = activations[layer][0, positions].float()
            if use_jacobian:
                residual = lens.transport(residual, layer)
            logits = model.unembed(residual).detach().cpu()
            layer_ranks[int(layer)] = [
                token_ranks(logits[i], [tid])[0]
                for i, tid in enumerate(encoded.answer_token_ids[: len(positions)])
            ]
        results.append(
            {
                "question_id": sample.question_id,
                "image_id": sample.image_id,
                "question": sample.question,
                "answer": sample.short_answer,
                "answer_token_ids": encoded.answer_token_ids,
                "prediction_positions": positions,
                "layer_ranks": layer_ranks,
                "best_rank": min((min(v) for v in layer_ranks.values() if v), default=None),
            }
        )
    return results


def parse_layer_list(text: str | None) -> list[int] | None:
    if text is None or text == "":
        return None
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def stride_layers(n_layers: int, *, target_layer: int, stride: int) -> list[int]:
    layers = list(range(0, target_layer, stride))
    if target_layer - 1 not in layers:
        layers.append(target_layer - 1)
    return sorted(set(layers))


__all__ = [
    "EncodedVQASample",
    "PositionScope",
    "Qwen3VLJLensModel",
    "Qwen3VLResidualRecorder",
    "VQASample",
    "evaluate_answer_ranks",
    "fit_vqa_jacobian_lens",
    "jacobian_for_vqa_sample",
    "load_vqa_metadata",
    "parse_layer_list",
    "position_mask",
    "stride_layers",
    "token_ranks",
    "write_vqa_splits",
]
