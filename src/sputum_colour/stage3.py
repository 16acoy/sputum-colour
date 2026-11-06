"""Aggregate per-photo evaluation metrics for the pipeline."""

import tempfile
from pathlib import Path

import pandas as pd


def _average_stage_metrics(intermediate_results_dir: Path, stage: int) -> pd.Series:
    """Average each metric across the per-photo CSVs for one pipeline stage."""
    csv_paths = sorted(
        intermediate_results_dir.glob(f"eval_results_stage{stage}_*.csv")
    )
    if not csv_paths:
        raise FileNotFoundError(
            f"No stage{stage} evaluation CSVs found in {intermediate_results_dir}"
        )

    frames = []
    expected_columns = None
    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path)
        if frame.empty or frame.columns.empty:
            raise ValueError(f"Evaluation CSV has no metrics: {csv_path}")
        if expected_columns is None:
            expected_columns = frame.columns
        elif not frame.columns.equals(expected_columns):
            raise ValueError(f"Metric columns do not match in {csv_path}")
        frames.append(frame.apply(pd.to_numeric, errors="raise"))

    metrics = pd.concat(frames, ignore_index=True)
    averages = metrics.mean()
    averages.index = [f"stage{stage}_{column}_mean" for column in averages.index]
    return averages


def main(results_dir):
    """Write the pipeline's averaged evaluation metrics and clear intermediates."""
    if results_dir is None:
        results_path = Path(__file__).resolve().parents[2] / "results"
    else:
        results_path = Path(results_dir)

    intermediate_results_dir = results_path / "intermediate_results"
    if not intermediate_results_dir.is_dir():
        raise FileNotFoundError(
            f"Intermediate results directory does not exist: {intermediate_results_dir}"
        )

    stage1_averages = _average_stage_metrics(intermediate_results_dir, stage=1)
    stage2_averages = _average_stage_metrics(intermediate_results_dir, stage=2)
    final_metrics = pd.DataFrame(
        [{**stage1_averages.to_dict(), **stage2_averages.to_dict()}]
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=results_path,
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)

    try:
        final_metrics.to_csv(temporary_path, index=False)
        temporary_path.replace(results_path / "eval_results.csv")
    finally:
        temporary_path.unlink(missing_ok=True)

    # Remove intermediate files if desired
    # for path in intermediate_results_dir.iterdir():
    # if path.is_file():
    # path.unlink()

    return final_metrics


if __name__ == "__main__":
    main()
