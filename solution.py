from enums import PROBLEM_TYPE, EDGE_WEIGHT_TYPE
from distances import distance
import numpy as np
import functools


class Solution():
    def __init__(self,
                 NAME: str = "",
                 TYPE: PROBLEM_TYPE = PROBLEM_TYPE.TOUR,
                 COMMENT: str = "",
                 DIMENSION: int = 0,
                 solution: list[int] = [],
                 EDGE_WEIGHT_TYPE: EDGE_WEIGHT_TYPE = EDGE_WEIGHT_TYPE.EUC_2D):
        self._solution = solution
        self._name = NAME
        self._problem_type = TYPE
        self._comment = COMMENT
        self._dimension = DIMENSION
        self._edge_weight_type = EDGE_WEIGHT_TYPE
        self._adjacency_matrix = None
    
    def get_distance(self) -> int:
        """Compute the distance of the solution."""
        assert self._adjacency_matrix is not None
        cost: int = 0
        for i in range(len(self._solution)):
            curr = self._solution[i] - 1
            next = self._solution[(i+1) % self._dimension] - 1
            cost += self._adjacency_matrix[curr][next]
        return cost

    def load_adjacency_matrix(self, adjacency_matrix: np.ndarray) -> None:
        """Load the adjacency matrix."""
        self._adjacency_matrix = adjacency_matrix

    def __str__(self) -> str:
        return f"{self._name} \
            {self._problem_type} \
            {self._comment} \
            {self._dimension} \
            {self._solution}"

    def __repr__(self) -> str:
        return str(self)
