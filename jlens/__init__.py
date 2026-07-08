# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Jacobian lens: fit and apply the average input-output Jacobian as a readout
of decoder-transformer residuals."""

from jlens._logging import configure_logging
from jlens.fitting import fit, jacobian_for_prompt
from jlens.hf import HFLensModel, Layout, from_hf
from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens
from jlens.protocol import LensModel
from jlens.qwen3vl import (
    Qwen3VLJLensModel,
    Qwen3VLResidualRecorder,
    VQASample,
    evaluate_answer_ranks,
    fit_vqa_jacobian_lens,
    jacobian_for_vqa_sample,
    load_vqa_metadata,
    write_vqa_splits,
)

__all__ = [
    "ActivationRecorder",
    "HFLensModel",
    "JacobianLens",
    "Layout",
    "LensModel",
    "configure_logging",
    "evaluate_answer_ranks",
    "fit",
    "fit_vqa_jacobian_lens",
    "from_hf",
    "jacobian_for_prompt",
    "jacobian_for_vqa_sample",
    "load_vqa_metadata",
    "Qwen3VLJLensModel",
    "Qwen3VLResidualRecorder",
    "VQASample",
    "write_vqa_splits",
]
