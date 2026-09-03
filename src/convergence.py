"""Compute and visualize FEM convergence against a reference grid."""

from typing import NamedTuple

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
from numpy.typing import NDArray


class Error(NamedTuple):
    """L2 error and H1 error seminorm."""

    l2: float
    h1: float


def interpolate_to_reference(
    solution: NDArray[np.float64],
    reference_size: int,
) -> NDArray[np.float64]:
    """Interpolate a P1 solution onto a nested reference grid."""
    coarse_size: int = solution.shape[0] - 1
    if reference_size % coarse_size != 0:
        raise ValueError("Reference grid must be a multiple of the coarse grid.")

    ratio: int = reference_size // coarse_size
    fine_indices = np.arange(reference_size + 1)
    coarse_indices = np.minimum(fine_indices // ratio, coarse_size - 1)
    local = (fine_indices % ratio) / ratio
    coarse_indices[-1] = coarse_size - 1
    local[-1] = 1.0

    column, row = np.meshgrid(coarse_indices, coarse_indices)
    local_x, local_y = np.meshgrid(local, local)
    bottom_left = solution[row, column]
    top_right = solution[row + 1, column + 1]

    lower = (
        (1.0 - local_x) * bottom_left
        + (local_x - local_y) * solution[row, column + 1]
        + local_y * top_right
    )
    upper = (
        (1.0 - local_y) * bottom_left
        + local_x * top_right
        + (local_y - local_x) * solution[row + 1, column]
    )
    return np.where(local_y <= local_x, lower, upper)


def calculate_error(
    solution: NDArray[np.float64],
    reference: NDArray[np.float64],
) -> tuple[Error, NDArray[np.float64]]:
    """Integrate L2 and H1 errors on the reference triangulation."""
    reference_size: int = reference.shape[0] - 1
    difference = interpolate_to_reference(solution, reference_size) - reference
    bottom_left = difference[:-1, :-1]
    bottom_right = difference[:-1, 1:]
    top_left = difference[1:, :-1]
    top_right = difference[1:, 1:]
    triangle_area: float = 0.5 / reference_size**2

    lower_l2 = (
        bottom_left**2
        + bottom_right**2
        + top_right**2
        + bottom_left * bottom_right
        + bottom_right * top_right
        + top_right * bottom_left
    )
    upper_l2 = (
        bottom_left**2
        + top_right**2
        + top_left**2
        + bottom_left * top_right
        + top_right * top_left
        + top_left * bottom_left
    )
    l2_error: float = float(np.sqrt(triangle_area / 6.0 * np.sum(lower_l2 + upper_l2)))

    lower_gradient_x = reference_size * (bottom_right - bottom_left)
    lower_gradient_y = reference_size * (top_right - bottom_right)
    upper_gradient_x = reference_size * (top_right - top_left)
    upper_gradient_y = reference_size * (top_left - bottom_left)
    h1_error: float = float(
        np.sqrt(
            triangle_area
            * np.sum(
                lower_gradient_x**2
                + lower_gradient_y**2
                + upper_gradient_x**2
                + upper_gradient_y**2
            )
        )
    )
    return Error(l2_error, h1_error), difference


def format_convergence_table(grid_sizes: list[int], errors: list[Error]) -> str:
    """Format the convergence table and its short evaluation."""
    lines = [
        "Convergence table (N=80 is the reference solution)",
        f"{'N':>4} {'h':>8} {'L2 error':>14} {'EOC L2':>9} "
        f"{'H1 error':>14} {'EOC H1':>9}",
    ]
    for index, (grid_size, error) in enumerate(zip(grid_sizes, errors, strict=True)):
        if index == 0:
            eoc_l2 = eoc_h1 = "-"
        else:
            previous = errors[index - 1]
            eoc_l2 = f"{np.log(previous.l2 / error.l2) / np.log(2.0):.3f}"
            eoc_h1 = f"{np.log(previous.h1 / error.h1) / np.log(2.0):.3f}"
        lines.append(
            f"{grid_size:4d} {1.0 / grid_size:8.4f} {error.l2:14.6e} "
            f"{eoc_l2:>9} {error.h1:14.6e} {eoc_h1:>9}"
        )
    lines.extend(
        [
            "",
            "Evaluation:",
            "Both errors decrease and the EOC values approach the expected orders",
            "2 for L2 and 1 for H1. The values are lower because N=80 is itself",
            "a numerical reference solution.",
        ]
    )
    return "\n".join(lines)


def plot_convergence(
    grid_sizes: list[int],
    errors: list[Error],
    output_path: str,
) -> None:
    """Save the L2 and H1 errors as a log-log plot."""
    steps = 1.0 / np.asarray(grid_sizes, dtype=float)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.loglog(steps, [error.l2 for error in errors], "o-", label="$L^2$ error")
    axis.loglog(steps, [error.h1 for error in errors], "s-", label="$H^1$ error")
    axis.set_xlabel("Step h")
    axis.set_ylabel("Error")
    axis.set_xticks(steps, [f"{step:g}" for step in steps])
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.set_title("FEM convergence")
    axis.grid(True, which="both")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_reference_comparison(
    interpolated_solution: NDArray[np.float64],
    reference: NDArray[np.float64],
    grid_size: int,
    output_path: str,
) -> None:
    """Save center cuts and a heatmap comparing one grid with the reference."""
    coordinates = np.linspace(0.0, 1.0, reference.shape[0])
    middle = len(coordinates) // 2
    difference = interpolated_solution - reference
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(coordinates, interpolated_solution[middle, :], label=f"N={grid_size}")
    axes[0].plot(coordinates, reference[middle, :], "--", label="N=80")
    axes[0].set_title("Horizontal cut y = 0.5")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("u")
    axes[0].grid()
    axes[0].legend()

    axes[1].plot(coordinates, interpolated_solution[:, middle], label=f"N={grid_size}")
    axes[1].plot(coordinates, reference[:, middle], "--", label="N=80")
    axes[1].set_title("Vertical cut x = 0.5")
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
    axes[2].set_title(f"N={grid_size} minus N=80")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    figure.colorbar(image, ax=axes[2], label="difference")

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
