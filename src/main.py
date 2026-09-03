from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray

import fem
import finite_volume

OUTPUT_DIRECTORY: Path = Path("output")


class Comparison(NamedTuple):
    fem_centers: NDArray[np.float64]
    difference: NDArray[np.float64]
    relative_l2: float
    maximum_difference: float


def plot_solution(
    coordinates: NDArray[np.float64],
    solution: NDArray[np.float64],
    name: str,
) -> None:
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


def fem_at_cell_centers(solution: NDArray[np.float64]) -> NDArray[np.float64]:
    return 0.5 * (solution[:-1, :-1] + solution[1:, 1:])


def get_comparison(
    fem_solution: NDArray[np.float64],
    finite_volume_solution: NDArray[np.float64],
) -> Comparison:
    fem_centers: NDArray[np.float64] = fem_at_cell_centers(fem_solution)
    difference: NDArray[np.float64] = fem_centers - finite_volume_solution
    relative_l2: float = float(
        np.linalg.norm(difference) / np.linalg.norm(finite_volume_solution)
    )
    maximum_difference: float = float(np.max(np.abs(difference)))
    return Comparison(fem_centers, difference, relative_l2, maximum_difference)


def plot_method_comparison(
    coordinates: NDArray[np.float64],
    finite_volume_solution: NDArray[np.float64],
    comparison: Comparison,
    name: str,
) -> None:
    middle: int = len(coordinates) // 2
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(coordinates, comparison.fem_centers[middle, :], label="FEM")
    axes[0].plot(coordinates, finite_volume_solution[middle, :], "--", label="FVM")
    axes[0].set_title(f"Horizontal cut y = {coordinates[middle]:.3f}")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("u")
    axes[0].grid()
    axes[0].legend()

    axes[1].plot(coordinates, comparison.fem_centers[:, middle], label="FEM")
    axes[1].plot(coordinates, finite_volume_solution[:, middle], "--", label="FVM")
    axes[1].set_title(f"Vertical cut x = {coordinates[middle]:.3f}")
    axes[1].set_xlabel("y")
    axes[1].set_ylabel("u")
    axes[1].grid()
    axes[1].legend()

    image = axes[2].imshow(
        comparison.difference,
        origin="lower",
        extent=(0.0, 1.0, 0.0, 1.0),
        cmap="coolwarm",
    )
    axes[2].set_title("FEM - FVM at cell centers")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    figure.colorbar(image, ax=axes[2], label="difference")

    figure.tight_layout()
    figure.savefig(OUTPUT_DIRECTORY / name, dpi=160)
    plt.close(figure)


def print_summary(
    grid_size: int,
    solution: NDArray[np.float64],
    fem_residual: float,
    comparison: Comparison,
    fvm_residual: float,
) -> None:
    print(
        f"{grid_size} x {grid_size}: min={solution.min():.6f}, "
        f"max={solution.max():.6f}, residual={fem_residual:.3e}"
    )
    print(
        f"FEM vs FVM {grid_size} x {grid_size}: "
        f"relative L2={comparison.relative_l2:.3e}, "
        f"max difference={comparison.maximum_difference:.3e}, "
        f"FVM residual={fvm_residual:.3e}"
    )


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)

    coordinates_10, solution_10, residual_10 = fem.solve(10)
    coordinates_100, solution_100, residual_100 = fem.solve(100)
    fvm_coordinates_10, fvm_solution_10, fvm_residual_10 = finite_volume.solve(10)
    fvm_coordinates_100, fvm_solution_100, fvm_residual_100 = finite_volume.solve(100)

    plot_solution(coordinates_10, solution_10, "solution_10x10.html")
    plot_solution(coordinates_100, solution_100, "solution_100x100.html")

    comparison_10 = get_comparison(solution_10, fvm_solution_10)
    plot_method_comparison(
        fvm_coordinates_10,
        fvm_solution_10,
        comparison_10,
        "comparison_10x10.png",
    )

    comparison_100 = get_comparison(solution_100, fvm_solution_100)
    plot_method_comparison(
        fvm_coordinates_100,
        fvm_solution_100,
        comparison_100,
        "comparison_100x100.png",
    )

    print_summary(
        10,
        solution_10,
        residual_10,
        comparison_10,
        fvm_residual_10,
    )
    print_summary(
        100,
        solution_100,
        residual_100,
        comparison_100,
        fvm_residual_100,
    )
    print(f"Plots saved to {OUTPUT_DIRECTORY.resolve()}")


if __name__ == "__main__":
    main()
