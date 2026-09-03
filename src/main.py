"""Run FEM solutions and convergence analysis."""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray

import convergence
import fem

OUTPUT_DIRECTORY: Path = Path("output")


def plot_solution(
    coordinates: NDArray[np.float64],
    solution: NDArray[np.float64],
    name: str,
) -> None:
    """Save one FEM solution as an interactive surface plot."""
    grid_size: int = len(coordinates) - 1
    figure = go.Figure(
        data=go.Surface(
            x=coordinates,
            y=coordinates,
            z=solution,
            colorscale="Viridis",
            colorbar={"title": "u(x, y)"},
            hovertemplate="x=%{x:.4f}<br>y=%{y:.4f}<br>u=%{z:.6f}<extra></extra>",
        )
    )
    figure.update_layout(
        title=f"FEM solution, {grid_size} x {grid_size} grid",
        scene={
            "xaxis_title": "x",
            "yaxis_title": "y",
            "zaxis_title": "u(x, y)",
            "aspectmode": "cube",
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 50},
    )
    figure.write_html(OUTPUT_DIRECTORY / name, include_plotlyjs=True)


def format_solution_summary(
    grid_size: int,
    solution: NDArray[np.float64],
    fem_residual: float,
) -> str:
    """Format the numerical summary for one FEM solution."""
    return (
        f"{grid_size} x {grid_size}: min={solution.min():.6f}, "
        f"max={solution.max():.6f}, residual={fem_residual:.3e}"
    )


def main() -> None:
    """Solve all grids and save plots and the final text summary."""
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)

    grid_sizes = [10, 20, 40, 80]
    results = [fem.solve(grid_size) for grid_size in grid_sizes]
    summary_lines = []
    for grid_size, (coordinates, solution, residual) in zip(
        grid_sizes, results, strict=True
    ):
        summary_lines.append(format_solution_summary(grid_size, solution, residual))

    for grid_size in grid_sizes[:-1]:
        (OUTPUT_DIRECTORY / f"solution_{grid_size}x{grid_size}.html").unlink(
            missing_ok=True
        )
    reference_coordinates, reference, _ = results[-1]
    plot_solution(reference_coordinates, reference, "solution_80x80.html")

    errors_and_differences = [
        convergence.calculate_error(solution, reference)
        for _, solution, _ in results[:-1]
    ]
    errors = [item[0] for item in errors_and_differences]
    interpolated_solutions = [
        convergence.interpolate_to_reference(solution, 80)
        for _, solution, _ in results[:-1]
    ]
    summary_lines.extend(
        ["", convergence.format_convergence_table(grid_sizes[:-1], errors)]
    )
    (OUTPUT_DIRECTORY / "summary.txt").write_text(
        "\n".join(summary_lines), encoding="utf-8"
    )
    convergence.plot_convergence(
        grid_sizes[:-1],
        errors,
        str(OUTPUT_DIRECTORY / "convergence.png"),
    )
    for grid_size, interpolated_solution in zip(
        grid_sizes[:-1], interpolated_solutions, strict=True
    ):
        convergence.plot_reference_comparison(
            interpolated_solution,
            reference,
            grid_size,
            str(OUTPUT_DIRECTORY / f"comparison_{grid_size}x{grid_size}.png"),
        )


if __name__ == "__main__":
    main()
