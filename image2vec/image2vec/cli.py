"""CLI: `image2vec IMAGE -o out.svg` or `python -m image2vec`."""

from __future__ import annotations

import argparse
import sys

from image2vec.loop import convert_image
from image2vec.prompts import STYLES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Iteratively convert a raster image to SVG with a Pydantic AI agent on OpenRouter.",
    )
    parser.add_argument("image", help="Input PNG/JPEG/WebP image.")
    parser.add_argument("-o", "--output", help="Copy the final SVG here.")
    parser.add_argument("--workspace", help="Directory for iteration files. Default: runs/<timestamp>.")
    parser.add_argument(
        "--style",
        default="flat",
        choices=sorted(STYLES),
        help="Vector look. Default: flat.",
    )
    parser.add_argument("--max-iters", type=int, default=5, dest="max_iterations")
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.75,
        help="Also require critic score >= this (0-1) before stopping.",
    )
    parser.add_argument("--model", help="OpenRouter model slug. Default: OPENROUTER_CHAT_MODEL or openai/gpt-4o.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = convert_image(
            args.image,
            output=args.output,
            workspace_dir=args.workspace,
            style=args.style,
            max_iterations=args.max_iterations,
            min_score=args.min_score,
            model=args.model,
        )
    except (OSError, RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"output {result.output_svg}")
    print(f"render {result.output_png}")
    print(f"passed {result.passed}  iters {result.iterations}  {result.last_metrics.summary()}")
    print(f"critic {result.last_critique.score:.2f}  {result.last_critique.summary}")
    return 0 if result.passed else 1
