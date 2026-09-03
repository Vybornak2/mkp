from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

import finite_volume
import fem


def save_solution_plot(
    coordinates: NDArray[np.float64], solution: NDArray[np.float64], output_path: Path
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
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def fem_at_cell_centers(solution: NDArray[np.float64]) -> NDArray[np.float64]:
    return 0.5 * (solution[:-1, :-1] + solution[1:, 1:])


def save_method_comparison(
    coordinates: NDArray[np.float64],
    fem_solution: NDArray[np.float64],
    finite_volume_solution: NDArray[np.float64],
    output_path: Path,
) -> tuple[float, float]:
    fem_centers: NDArray[np.float64] = fem_at_cell_centers(fem_solution)
    difference: NDArray[np.float64] = fem_centers - finite_volume_solution
    relative_l2: float = float(
        np.linalg.norm(difference) / np.linalg.norm(finite_volume_solution)
    )
    maximum_difference: float = float(np.max(np.abs(difference)))

    middle: int = len(coordinates) // 2
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(coordinates, fem_centers[middle, :], label="FEM")
    axes[0].plot(coordinates, finite_volume_solution[middle, :], "--", label="FVM")
    axes[0].set_title(f"Horizontal cut y = {coordinates[middle]:.3f}")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("u")
    axes[0].grid()
    axes[0].legend()

    axes[1].plot(coordinates, fem_centers[:, middle], label="FEM")
    axes[1].plot(coordinates, finite_volume_solution[:, middle], "--", label="FVM")
    axes[1].set_title(f"Vertical cut x = {coordinates[middle]:.3f}")
    axes[1].set_xlabel("y")
    axes[1].set_ylabel("u")
    axes[1].grid()
    axes[1].legend()

    image = axes[2].imshow(
        difference,
        origin="lower",
        extent=(0.0, 1.0, 0.0, 1.0),
        cmap="coolwarm",
    )
    axes[2].set_title("FEM - FVM at cell centers")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    figure.colorbar(image, ax=axes[2], label="difference")

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return relative_l2, maximum_difference


def main() -> None:
    output_directory: Path = Path("output")
    output_directory.mkdir(exist_ok=True)

    coordinates_10, solution_10, residual_10 = fem.solve(10)
    coordinates_100, solution_100, residual_100 = fem.solve(100)
    fvm_coordinates_10, fvm_solution_10, fvm_residual_10 = finite_volume.solve(10)
    fvm_coordinates_100, fvm_solution_100, fvm_residual_100 = finite_volume.solve(100)

    save_solution_plot(
        coordinates_10, solution_10, output_directory / "solution_10x10.png"
    )
    save_solution_plot(
        coordinates_100, solution_100, output_directory / "solution_100x100.png"
    )
    relative_l2_10, maximum_difference_10 = save_method_comparison(
        fvm_coordinates_10,
        solution_10,
        fvm_solution_10,
        output_directory / "comparison_10x10.png",
    )
    relative_l2_100, maximum_difference_100 = save_method_comparison(
        fvm_coordinates_100,
        solution_100,
        fvm_solution_100,
        output_directory / "comparison_100x100.png",
    )

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(coordinates_10, solution_10[5, :], "o-", label="10 x 10")
    axes[0].plot(coordinates_100, solution_100[50, :], label="100 x 100")
    axes[0].set_title("Horizontal cut y = 0.5")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("u(x, 0.5)")
    axes[0].grid()
    axes[0].legend()

    axes[1].plot(coordinates_10, solution_10[:, 5], "o-", label="10 x 10")
    axes[1].plot(coordinates_100, solution_100[:, 50], label="100 x 100")
    axes[1].set_title("Vertical cut x = 0.5")
    axes[1].set_xlabel("y")
    axes[1].set_ylabel("u(0.5, y)")
    axes[1].grid()
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_directory / "cuts.png", dpi=160)
    plt.close(figure)

    print(
        f"10 x 10:  min={solution_10.min():.6f}, "
        f"max={solution_10.max():.6f}, residual={residual_10:.3e}"
    )
    print(
        f"100 x 100: min={solution_100.min():.6f}, "
        f"max={solution_100.max():.6f}, residual={residual_100:.3e}"
    )
    print(
        f"FEM vs FVM 10 x 10: relative L2={relative_l2_10:.3e}, "
        f"max difference={maximum_difference_10:.3e}, "
        f"FVM residual={fvm_residual_10:.3e}"
    )
    print(
        f"FEM vs FVM 100 x 100: relative L2={relative_l2_100:.3e}, "
        f"max difference={maximum_difference_100:.3e}, "
        f"FVM residual={fvm_residual_100:.3e}"
    )
    print(f"Plots saved to {output_directory.resolve()}")


if __name__ == "__main__":
    main()
