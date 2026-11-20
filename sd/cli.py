"""Command-line interface for running DiffusionForge in Google Colab."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# The original model files use imports such as `from ddpm import ...`.
# Add this directory so those existing imports continue to work unchanged.
SD_DIR = Path(__file__).resolve().parent
REPO_ROOT = SD_DIR.parent
if str(SD_DIR) not in sys.path:
    sys.path.insert(0, str(SD_DIR))

import torch
from PIL import Image

from runtime import (
    DEFAULT_CHECKPOINT_NAME,
    DEFAULT_CHECKPOINT_REPO,
    generate_image,
    resolve_checkpoint,
    resolve_device,
)


def command_doctor(_args: argparse.Namespace) -> int:
    print("DiffusionForge environment check")
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA runtime reported by PyTorch: {torch.version.cuda}")
    else:
        print("GPU: not detected")
        print("For this project, use a Colab GPU runtime for practical inference.")

    required = [
        REPO_ROOT / "data" / "vocab.json",
        REPO_ROOT / "data" / "merges.txt",
    ]
    for path in required:
        print(f"{'OK' if path.is_file() else 'MISSING'}: {path.relative_to(REPO_ROOT)}")

    return 0 if all(p.is_file() for p in required) else 1


def command_setup(args: argparse.Namespace) -> int:
    path = resolve_checkpoint(args.checkpoint)
    print(f"Checkpoint ready: {path}")
    return 0


def run_generation(args: argparse.Namespace, input_image=None) -> int:
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = generate_image(
        prompt=args.prompt,
        input_image=input_image,
        negative_prompt=args.negative_prompt,
        strength=args.strength,
        cfg_scale=args.cfg_scale,
        steps=args.steps,
        seed=args.seed,
        checkpoint=args.checkpoint,
        device=args.device,
    )
    Image.fromarray(result).save(output_path)
    print(f"Image saved to: {output_path.resolve()}")
    return 0


def command_txt2img(args: argparse.Namespace) -> int:
    return run_generation(args)


def command_img2img(args: argparse.Namespace) -> int:
    image_path = Path(args.input).expanduser()
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")
    with Image.open(image_path) as image:
        input_image = image.convert("RGB")
    return run_generation(args, input_image=input_image)


def add_generation_arguments(parser: argparse.ArgumentParser, *, img2img: bool) -> None:
    if img2img:
        parser.add_argument("--input", required=True, help="Path to the source image.")
    parser.add_argument("--prompt", required=True, help="Text prompt.")
    parser.add_argument("--negative-prompt", default="", help="Negative prompt (optional).")
    parser.add_argument("--output", default="outputs/diffusionforge.png", help="Output PNG path.")
    parser.add_argument("--checkpoint", default=None, help="Local .ckpt path; downloads the default if omitted.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strength", type=float, default=0.9,
                        help="Image-to-image strength; must be in (0, 1].")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="diffusionforge",
        description="Run the custom DiffusionForge Stable Diffusion 1.5 pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Check Python, PyTorch, CUDA and tokenizer files.")
    doctor.set_defaults(func=command_doctor)

    setup = subparsers.add_parser("setup", help="Download or verify the pretrained checkpoint.")
    setup.add_argument("--checkpoint", default=None, help="Use an existing local checkpoint.")
    setup.set_defaults(func=command_setup)

    txt = subparsers.add_parser("txt2img", help="Generate an image from a text prompt.")
    add_generation_arguments(txt, img2img=False)
    txt.set_defaults(func=command_txt2img)

    img = subparsers.add_parser("img2img", help="Transform an existing image using a prompt.")
    add_generation_arguments(img, img2img=True)
    img.set_defaults(func=command_img2img)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except Exception as exc:
        print(f"DiffusionForge error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
