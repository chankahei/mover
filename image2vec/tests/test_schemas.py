from image2vec.schemas import Critique, CritiqueIssue, SimilarityReport, VectorDraft


def test_vector_draft_roundtrip() -> None:
    draft = VectorDraft(svg_path="current.svg", focus="silhouette", notes="blocked in")
    assert draft.svg_path == "current.svg"


def test_critique_pass_threshold() -> None:
    critique = Critique(
        passed=True,
        score=0.88,
        summary="Close enough for icon style.",
        issues=[],
    )
    assert critique.passed
    dumped = critique.model_dump()
    assert dumped["score"] == 0.88


def test_similarity_summary() -> None:
    report = SimilarityReport(
        width=64,
        height=64,
        mse=10.0,
        psnr=30.0,
        ssim=0.9,
        histogram_correlation=0.8,
        edge_iou=0.5,
    )
    text = report.summary()
    assert "SSIM=0.900" in text
    assert "64x64" in text


def test_critique_issue_shape() -> None:
    issue = CritiqueIssue(area="background", problem="too dark", fix="fill #f5f5f5")
    assert "f5f5f5" in issue.fix
