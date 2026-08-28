from image2vec.prompts import STYLES, generator_system_prompt, generator_user_text


def test_system_prompt_covers_loop_and_styles() -> None:
    prompt = generator_system_prompt()
    assert "vectorization" in prompt.lower() or "SVG" in prompt
    assert "write_svg" in prompt
    assert "flat" in prompt
    for name in STYLES:
        assert name in prompt


def test_later_iteration_includes_critique() -> None:
    from image2vec.schemas import Critique, SimilarityReport

    text = generator_user_text(
        width=64,
        height=32,
        style="icon",
        iteration=2,
        max_iterations=5,
        metrics=SimilarityReport(
            width=64,
            height=32,
            mse=1,
            psnr=40,
            ssim=0.5,
            histogram_correlation=0.4,
            edge_iou=0.2,
        ),
        critique=Critique(passed=False, score=0.4, summary="missing ear", issues=[]),
    )
    assert "64x32" in text or "64" in text
    assert "missing ear" in text
    assert "iteration 2" in text.lower() or "2 of 5" in text
