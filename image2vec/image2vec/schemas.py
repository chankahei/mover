"""Structured outputs and reports for the conversion loop."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SimilarityReport(BaseModel):
    """OpenCV / SSIM comparison of the source image against a rendered SVG."""

    width: int
    height: int
    mse: float = Field(description="Mean squared error in 0-255 RGB space.")
    psnr: float = Field(description="Peak signal-to-noise ratio in dB. Higher is closer.")
    ssim: float = Field(ge=0, le=1, description="Structural similarity. 1 is identical.")
    histogram_correlation: float = Field(
        description="HSV histogram correlation in [-1, 1]. 1 is identical."
    )
    edge_iou: float = Field(
        ge=0, le=1, description="Intersection-over-union of Canny edge maps."
    )

    def summary(self) -> str:
        return (
            f"SSIM={self.ssim:.3f}  PSNR={self.psnr:.1f}dB  "
            f"hist={self.histogram_correlation:.3f}  edge_IoU={self.edge_iou:.3f}  "
            f"MSE={self.mse:.1f}  size={self.width}x{self.height}"
        )


class VectorDraft(BaseModel):
    """Generator output for one iteration. The SVG itself lives on disk."""

    svg_path: str = Field(
        description="Workspace-relative path of the SVG written this iteration, e.g. current.svg."
    )
    focus: str = Field(description="What this iteration tried to get right.")
    notes: str = Field(description="Short note for the critic or the next iteration.")


class CritiqueIssue(BaseModel):
    area: str = Field(description="Region or element, e.g. 'left eye' or 'background'.")
    problem: str
    fix: str = Field(description="Concrete SVG change that would address the problem.")


class Critique(BaseModel):
    """Vision-model QA of source vs rendered SVG."""

    passed: bool = Field(description="True only when the render is good enough to stop.")
    score: float = Field(ge=0, le=1, description="Overall fidelity in the requested style.")
    summary: str
    issues: list[CritiqueIssue] = Field(default_factory=list)


class IterationRecord(BaseModel):
    index: int
    svg_path: str
    png_path: str
    metrics: SimilarityReport
    critique: Critique
    draft: VectorDraft


class ConversionResult(BaseModel):
    passed: bool
    iterations: int
    output_svg: str
    output_png: str
    workspace: str
    last_metrics: SimilarityReport
    last_critique: Critique
    history: list[IterationRecord] = Field(default_factory=list)
