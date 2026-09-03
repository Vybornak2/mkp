from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


def solve(
    elements_per_side: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    n: int = elements_per_side
    h: float = 1.0 / n
    node_count: int = (n + 1) * (n + 1)

    coordinates: NDArray[np.float64] = np.linspace(0.0, 1.0, n + 1)
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    x: NDArray[np.float64] = x_grid.ravel()
    y: NDArray[np.float64] = y_grid.ravel()

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    load: NDArray[np.float64] = np.zeros(node_count)

    # Local lower stiffness matrix for the triangular element
    lower_stiffness: NDArray[np.float64] = 0.5 * np.array(
        [
            [1.0, -1.0, 0.0],
            [-1.0, 2.0, -1.0],
            [0.0, -1.0, 1.0],
        ]
    )
    # Local upper stiffness matrix for the triangular element
    upper_stiffness: NDArray[np.float64] = 0.5 * np.array(
        [
            [1.0, 0.0, -1.0],
            [0.0, 1.0, -1.0],
            [-1.0, -1.0, 2.0],
        ]
    )

    # Loop over each square element in the grid
    for row in range(n):
        for column in range(n):
            # Node indices for the current square element
            bottom_left: int = row * (n + 1) + column
            bottom_right: int = bottom_left + 1
            top_left: int = bottom_left + n + 1
            top_right: int = top_left + 1

            triangles: tuple[tuple[list[int], NDArray[np.float64]], ...] = (
                ([bottom_left, bottom_right, top_right], lower_stiffness),
                ([bottom_left, top_right, top_left], upper_stiffness),
            )

            for nodes, local_stiffness in triangles:
                for local_row in range(3):
                    node: int = nodes[local_row]
                    source: float = 1.0 - np.sin(np.pi * x[node]) * np.sin(
                        np.pi * y[node]
                    )
                    load[node] += source * h * h / 6.0

                    for local_column in range(3):
                        rows.append(node)
                        columns.append(nodes[local_column])
                        values.append(local_stiffness[local_row, local_column])

    robin_edges: list[tuple[int, int]] = []

    for column in range(n):
        robin_edges.append((column, column + 1))
        robin_edges.append((n * (n + 1) + column, n * (n + 1) + column + 1))

    for row in range(n):
        robin_edges.append((row * (n + 1) + n, (row + 1) * (n + 1) + n))

    for row in range(n // 2, n):
        robin_edges.append((row * (n + 1), (row + 1) * (n + 1)))

    robin_stiffness: NDArray[np.float64] = h / 48.0 * np.array([[2.0, 1.0], [1.0, 2.0]])

    for first_node, second_node in robin_edges:
        edge_nodes: tuple[int, int] = (first_node, second_node)

        for local_row in range(2):
            node = edge_nodes[local_row]
            load[node] -= 7.0 * h / 16.0

            for local_column in range(2):
                rows.append(node)
                columns.append(edge_nodes[local_column])
                values.append(robin_stiffness[local_row, local_column])

    stiffness = coo_matrix(  # type: ignore
        (values, (rows, columns)),  # type: ignore
        shape=(node_count, node_count),
    ).tocsr()

    free_nodes: NDArray[np.bool_] = np.ones(node_count, dtype=bool)
    for row in range(n // 2 + 1):
        free_nodes[row * (n + 1)] = False

    reduced_stiffness = stiffness[free_nodes][:, free_nodes]
    reduced_load: NDArray[np.float64] = load[free_nodes]
    solution: NDArray[np.float64] = np.zeros(node_count)
    solution[free_nodes] = spsolve(reduced_stiffness, reduced_load)

    residual: float = float(
        np.linalg.norm(reduced_stiffness @ solution[free_nodes] - reduced_load)
    )
    return coordinates, solution.reshape((n + 1, n + 1)), residual


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


def main() -> None:
    output_directory: Path = Path("output")
    output_directory.mkdir(exist_ok=True)

    coordinates_10, solution_10, residual_10 = solve(10)
    coordinates_100, solution_100, residual_100 = solve(100)

    save_solution_plot(
        coordinates_10, solution_10, output_directory / "solution_10x10.png"
    )
    save_solution_plot(
        coordinates_100, solution_100, output_directory / "solution_100x100.png"
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
    print(f"Plots saved to {output_directory.resolve()}")


if __name__ == "__main__":
    main()
