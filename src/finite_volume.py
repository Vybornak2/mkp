import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


def solve(
    cells_per_side: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    n: int = cells_per_side
    h: float = 1.0 / n
    cell_count: int = n * n
    coordinates: NDArray[np.float64] = (np.arange(n, dtype=float) + 0.5) * h

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    load: NDArray[np.float64] = np.zeros(cell_count)

    robin_alpha: float = 1.0 / 8.0
    robin_beta: float = 7.0 / 8.0
    robin_diagonal: float = 2.0 * robin_alpha * h / (2.0 + robin_alpha * h)
    robin_load_value: float = -2.0 * robin_beta * h / (2.0 + robin_alpha * h)

    def add_matrix_entry(row: int, column: int, value: float) -> None:
        rows.append(row)
        columns.append(column)
        values.append(value)

    for row in range(n):
        for column in range(n):
            cell: int = row * n + column
            x: float = coordinates[column]
            y: float = coordinates[row]
            diagonal: float = 0.0
            load[cell] = (1.0 - np.sin(np.pi * x) * np.sin(np.pi * y)) * h * h

            for neighbour in (
                cell - 1 if column > 0 else None,
                cell + 1 if column < n - 1 else None,
                cell - n if row > 0 else None,
                cell + n if row < n - 1 else None,
            ):
                if neighbour is not None:
                    diagonal += 1.0
                    add_matrix_entry(cell, neighbour, -1.0)

            boundary_faces: int = (
                int(column == 0)
                + int(column == n - 1)
                + int(row == 0)
                + int(row == n - 1)
            )
            robin_faces: int = boundary_faces

            if column == 0 and y <= 0.5:
                diagonal += 2.0
                robin_faces -= 1

            diagonal += robin_faces * robin_diagonal
            load[cell] += robin_faces * robin_load_value
            add_matrix_entry(cell, cell, diagonal)

    matrix = coo_matrix(  # type: ignore
        (values, (rows, columns)),  # type: ignore
        shape=(cell_count, cell_count),
    ).tocsr()
    solution: NDArray[np.float64] = spsolve(matrix, load)
    residual: float = float(np.linalg.norm(matrix @ solution - load))
    return coordinates, solution.reshape((n, n)), residual
