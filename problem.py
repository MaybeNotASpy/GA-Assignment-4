import numpy as np
from dataclasses import dataclass
from enums import (
    PROBLEM_TYPE,
    EDGE_WEIGHT_TYPE,
    EDGE_WEIGHT_FORMAT,
    COORD_DISPLAY,
)
from distances import distance


def generate_adjacency_matrix(
        nodes_coords: list[tuple[int, int]],
        edge_weight_type: EDGE_WEIGHT_TYPE,
        edge_weight_format: EDGE_WEIGHT_FORMAT,
        ) -> np.ndarray:
    res = np.zeros((len(nodes_coords), len(nodes_coords)), dtype=np.int32)
    for i in range(len(nodes_coords)):
        for j in range(len(nodes_coords)):
            if i == j:
                continue
            res[i][j] = distance(
                nodes_coords[i],
                nodes_coords[j],
                edge_weight_type
            )
    return res


@dataclass
class Problem():
    NAME: str = ""
    TYPE: PROBLEM_TYPE = PROBLEM_TYPE.TSP
    COMMENT: str = ""
    DIMENSION: int = 0
    EDGE_WEIGHT_TYPE: EDGE_WEIGHT_TYPE = EDGE_WEIGHT_TYPE.EUC_2D
    ADJACENCY_MATRIX: np.ndarray = np.zeros((0, 0), dtype=np.int32)
    EDGE_WEIGHT_FORMAT: EDGE_WEIGHT_FORMAT = EDGE_WEIGHT_FORMAT.NONE
    DISPLAY_DATA_TYPE: COORD_DISPLAY = COORD_DISPLAY.COORD_DISPLAY
    SOLUTION_COST: int = 0

    def evaluate(self, tour: list[int]) -> tuple[float, float]:
        """Evaluate the fitness of a tour.
        Args:
            tour (list[int]): The tour to evaluate.
        Returns:
            fitness (float): The fitness of the tour.
            cost (float): The cost of the tour."""
        assert self.ADJACENCY_MATRIX is not None
        cost: int = 0
        for i in range(len(tour)):
            curr = tour[i] - 1
            next = tour[(i+1) % self.DIMENSION] - 1
            cost += self.ADJACENCY_MATRIX[curr][next]
        assert self.SOLUTION_COST <= cost
        fitness = self.get_fitness(cost)
        return fitness, cost

    def get_max_fitness(self) -> float:
        return 1. / self.SOLUTION_COST

    def get_min_cost(self) -> float:
        return self.SOLUTION_COST

    def get_fitness(self, cost: float) -> float:
        return 1. / cost
