# image2vec

Iteratively convert a raster image to SVG using a **Pydantic AI** agent on **OpenRouter**.

Each outer iteration:

1. The generator model writes an SVG (it can inspect the workspace and call vision tools).
2. The SVG is rasterized and compared to the source with OpenCV (SSIM, PSNR, histogram, Canny edge IoU).
3. A critic model looks at source vs render plus those metrics.
4. The loop stops when the critic passes (and the score meets `--min-score`).

## Setup

```bash
cd image2vec
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Rasterizing SVG needs cairo. On macOS: `brew install cairo`. If cairocffi still cannot find it, set `CAIRO_LIB` to the dylib path (for Homebrew: `/opt/homebrew/lib/libcairo.2.dylib`).

Copy `.env.example` to `.env` and set:

- `OPENROUTER_API_KEY` (required)
- `OPENROUTER_CHAT_MODEL` (optional, default `openai/gpt-4o` — must be vision-capable)

## Usage

```bash
python -m image2vec path/to/photo.png -o out.svg --style icon --max-iters 6
```

Styles: `flat`, `icon`, `line-art`, `geometric`, `poster`, `painterly`.

Iteration files land in `--workspace` or `runs/<timestamp>/`: `source.png`, `edges.png`, `iter_01.svg`, renders, metrics, and `final.svg`.

## Layout

| File | Role |
| --- | --- |
| `prompts.py` | System prompts: capability, styles, iterative method |
| `agents.py` | Generator + critic pydantic-ai agents, OpenRouter, FileSystem |
| `tools.py` | `read_image`, Canny, palette, contour trace, `write_svg`, render, similarity |
| `loop.py` | Generate → render → critique → stop |
| `vision.py` / `render.py` | OpenCV metrics and SVG rasterization |

The generator is given a sandboxed `FileSystem` rooted at the run workspace (`pydantic-ai-harness`) so it can read/write SVG text, plus tools that return image previews the FileSystem tools will not dump as binary.
