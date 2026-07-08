# Qwen3-VL Jacobian-Lens Scripts

These scripts are local experiment entry points for the Qwen3-VL multimodal
Jacobian-lens extension.

- `download_vqav2_val_subset.py`: download a small official VQAv2 val2014
  subset, fetch only the referenced COCO images, and write local JSONL metadata.
- `qwen3vl_jlens_experiment.py`: split preparation and fitting for
  VQAv2-style image/question/answer data.
- `qwen3vl_compare_readouts.py`: held-out VQA comparison between native
  logit-lens readout and J-lens readout.
- `qwen3vl_probe_target_word.py`: single-image target-word probing across
  layers and positions.

Large artifacts, checkpoints, logs, plots, and metric tables should be written
under `runs/`, which is intentionally ignored by Git.
