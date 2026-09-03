from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
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
    coordinates: NDArray[np.float64], solution: NDArray[np.float64]
) -> None:
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    figure = plt.figure(figsize=(8, 6))
    axes = figure.add_subplot(111, projection="3d")
    surface = axes.plot_surface(x_grid, y_grid, solution, cmap="viridis")
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_zlabel("u(x, y)")
    axes.set_title(
        f"FEM solution, {len(coordinates) - 1} x {len(coordinates) - 1} grid"
    )
    figure.colorbar(surface, ax=axes, shrink=0.7)
    figure.tight_layout()


def save_plot(name: str) -> None:
    figure = plt.gcf()
    figure.savefig(OUTPUT_DIRECTORY / name, dpi=160)
    plt.close(figure)


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

    plot_solution(coordinates_10, solution_10)
    save_plot("solution_10x10.png")
    plot_solution(coordinates_100, solution_100)
    save_plot("solution_100x100.png")

    comparison_10 = get_comparison(solution_10, fvm_solution_10)
    plot_method_comparison(
        fvm_coordinates_10,
        fvm_solution_10,
        comparison_10,
    )
    save_plot("comparison_10x10.png")

    comparison_100 = get_comparison(solution_100, fvm_solution_100)
    plot_method_comparison(
        fvm_coordinates_100,
        fvm_solution_100,
        comparison_100,
    )
    save_plot("comparison_100x100.png")

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
