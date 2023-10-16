import unittest

from enums import (
    PROBLEM_TYPE,
    EDGE_WEIGHT_TYPE,
    EDGE_WEIGHT_FORMAT,
    COORD_DISPLAY,
)
from problem import (
    Problem,
    generate_adjacency_matrix
)
from helper import nint
from solution import Solution
import numpy as np
import os


def get_tsp_data(filename: str) -> Problem:
    """Load the TSP data."""
    tsp_data = {}
    nodes: list[tuple[int, int]] = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line == "EOF":
                break
            if line.startswith("NAME"):
                tsp_data["NAME"] = line.split(":")[1].strip()
            elif line.startswith("TYPE"):
                tsp_data["TYPE"] = PROBLEM_TYPE[line.split(":")[1].strip()]
                assert tsp_data["TYPE"] != PROBLEM_TYPE.TOUR
            elif line.startswith("COMMENT"):
                tsp_data["COMMENT"] = line.split(":")[1].strip()
            elif line.startswith("DIMENSION"):
                tsp_data["DIMENSION"] = int(line.split(":")[1].strip())
                # Pre-allocate the adjacency matrix
                tsp_data["ADJACENCY_MATRIX"] = np.zeros(
                    (tsp_data["DIMENSION"],
                     tsp_data["DIMENSION"]
                     ), dtype=np.int32)
                # Pre-allocate the node list
                nodes = [None] * tsp_data["DIMENSION"]
            elif line.startswith("EDGE_WEIGHT_TYPE"):
                tsp_data["EDGE_WEIGHT_TYPE"] = EDGE_WEIGHT_TYPE[
                    line.split(":")[1].strip()
                ]
            elif line.startswith("EDGE_WEIGHT_FORMAT"):
                tsp_data["EDGE_WEIGHT_FORMAT"] = EDGE_WEIGHT_FORMAT[
                    line.split(":")[1].strip()
                ]
            elif line.startswith("DISPLAY_DATA_TYPE"):
                tsp_data["DISPLAY_DATA_TYPE"] = COORD_DISPLAY[
                    line.split(":")[1].strip()
                ]
            elif line.startswith("NODE_COORD_SECTION"):
                while True:
                    line = f.readline().strip()
                    if line == "EOF":
                        break
                    if line == "":
                        continue
                    if line == "DISPLAY_DATA_SECTION":
                        break
                    # Structure of a node: id x y
                    line = line.strip()
                    node = line.split()
                    x, y = node[1], node[2]
                    x, y = float(x), float(y)
                    nodes[int(node[0]) - 1] = (
                        x,
                        y
                    )
    try:
        tsp_data["EDGE_WEIGHT_TYPE"]
    except KeyError:
        tsp_data["EDGE_WEIGHT_TYPE"] = EDGE_WEIGHT_TYPE.NONE

    try:
        tsp_data["EDGE_WEIGHT_FORMAT"] = nodes
    except KeyError:
        tsp_data["EDGE_WEIGHT_FORMAT"] = EDGE_WEIGHT_FORMAT.NONE

    tsp_data["ADJACENCY_MATRIX"] = generate_adjacency_matrix(
        nodes,
        tsp_data["EDGE_WEIGHT_TYPE"],
        tsp_data["EDGE_WEIGHT_FORMAT"])
    return Problem(**tsp_data)


def get_tour_data(filename: str) -> Solution:
    """Load the tour data."""
    assert filename.endswith(".tour")
    assert os.path.exists(filename)
    tour_data = {}
    tour: list[int] = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line == "EOF":
                break
            if line.startswith("NAME"):
                tour_data["NAME"] = line.split(":")[1].strip()
            elif line.startswith("TYPE"):
                tour_data["TYPE"] = PROBLEM_TYPE[line.split(":")[1].strip()]
            elif line.startswith("COMMENT"):
                tour_data["COMMENT"] = line.split(":")[1].strip()
            elif line.startswith("DIMENSION"):
                tour_data["DIMENSION"] = int(line.split(":")[1].strip())
                # Pre-allocate the tour list
                tour = [None] * tour_data["DIMENSION"]
            elif line.startswith("TOUR_SECTION"):
                i = 0
                while True:
                    line = f.readline().strip()
                    if line == "EOF" or line == "-1":
                        break
                    if line == "":
                        continue
                    tour[i] = int(line)
                    i += 1
    return Solution(**tour_data, solution=tour)
