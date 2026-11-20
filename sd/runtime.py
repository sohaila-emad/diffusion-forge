"""Reusable runtime helpers for DiffusionForge.

This module deliberately reuses the existing model_loader and pipeline code.
It does not replace or modify the neural-network implementation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from transformers import CLIPTokenizer

import model_loader
import pipeline


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT_REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DEFAULT_CHECKPOINT_NAME = "v1-5-pruned-emaonly.ckpt"


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve auto/cuda/cpu to a torch device."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False. "
            "In Colab, select Runtime > Change runtime type > T4 GPU, then reconnect."
        )
    return device


def resolve_checkpoint(checkpoint: str | None = None) -> Path:
    """Use a supplied checkpoint path or download the project's SD 1.5 checkpoint."""
    if checkpoint:
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        return path

    return Path(
        hf_hub_download(
            repo_id=DEFAULT_CHECKPOINT_REPO,
            filename=DEFAULT_CHECKPOINT_NAME,
        )
    )


def load_runtime(
    checkpoint: str | None = None,
    requested_device: str = "auto",
) -> tuple[dict[str, Any], CLIPTokenizer, torch.device]:
    """Load tokenizer and model components using the existing implementation.

    Models are initially loaded on CPU, matching the user's working Colab
    notebook. The pipeline moves components to the selected device as needed
    and returns them to CPU through idle_device.
    """
    device = resolve_device(requested_device)

    vocab_path = REPO_ROOT / "data" / "vocab.json"
    merges_path = REPO_ROOT / "data" / "merges.txt"
    if not vocab_path.is_file() or not merges_path.is_file():
        raise FileNotFoundError(
            "Tokenizer files are missing. Expected data/vocab.json and data/merges.txt."
        )

    tokenizer = CLIPTokenizer(
        vocab=str(vocab_path),
        merges=str(merges_path),
    )

    checkpoint_path = resolve_checkpoint(checkpoint)
    models = model_loader.preload_models_from_standard_weights(
        str(checkpoint_path),
        "cpu",
    )
    return models, tokenizer, device


def generate_image(
    *,
    prompt: str,
    input_image=None,
    negative_prompt: str = "",
    strength: float = 0.9,
    cfg_scale: float = 8.0,
    steps: int = 50,
    seed: int | None = 42,
    checkpoint: str | None = None,
    device: str = "auto",
):
    """Run the existing DiffusionForge pipeline and return an RGB uint8 array."""
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty.")
    if steps < 1:
        raise ValueError("Inference steps must be at least 1.")
    if not 0.0 < strength <= 1.0:
        raise ValueError("Strength must be greater than 0 and at most 1.")

    models, tokenizer, resolved_device = load_runtime(
        checkpoint=checkpoint,
        requested_device=device,
    )

    if resolved_device.type == "cpu":
        print(
            "WARNING: CUDA is unavailable. DiffusionForge may be extremely slow "
            "or run out of system memory on CPU. A Colab GPU runtime is recommended."
        )
    else:
        print(f"Using GPU: {torch.cuda.get_device_name(resolved_device)}")

    result = pipeline.generate(
        prompt=prompt,
        uncond_prompt=negative_prompt,
        input_image=input_image,
        strength=strength,
        do_cfg=True,
        cfg_scale=cfg_scale,
        sampler_name="ddpm",
        n_inference_steps=steps,
        models=models,
        seed=seed,
        device=str(resolved_device),
        idle_device="cpu",
        tokenizer=tokenizer,
    )
    return result
