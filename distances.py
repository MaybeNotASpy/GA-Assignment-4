import numpy as np
from enums import EDGE_WEIGHT_TYPE
from helper import nint
import pytest


TwoD = tuple[int, int] | np.ndarray
ThreeD = tuple[int, int, int] | np.ndarray
pi = 3.141592
RRR = 6378.388  # Earth radius in km


def distance(x: TwoD | ThreeD, y: TwoD | ThreeD,
             edge_type: EDGE_WEIGHT_TYPE) -> int:
    """Compute the distance between two points."""
    assert len(x) == len(y)
    match edge_type:
        case EDGE_WEIGHT_TYPE.EUC_2D:
            return euclidean_distance(x, y)
        case EDGE_WEIGHT_TYPE.EUC_3D:
            return euclidean_distance(x, y)
        case EDGE_WEIGHT_TYPE.MAX_2D:
            return maximum_distance(x, y)
        case EDGE_WEIGHT_TYPE.MAX_3D:
            return maximum_distance(x, y)
        case EDGE_WEIGHT_TYPE.MAN_2D:
            return manhattan_distance(x, y)
        case EDGE_WEIGHT_TYPE.MAN_3D:
            return manhattan_distance(x, y)
        case EDGE_WEIGHT_TYPE.CEIL_2D:
            return ceiling_of_euclidean_distance(x, y)
        case EDGE_WEIGHT_TYPE.GEO:
            return geographical_distance(x, y)
        case EDGE_WEIGHT_TYPE.ATT:
            return pseudo_euclidean_distance(x, y)
        case _:
            raise ValueError(f"Unimplemented edge weight type for {edge_type}")


def euclidean_distance(x: TwoD | ThreeD, y: TwoD | ThreeD) -> int:
    """Compute the Euclidean distance (L2 Norm) between two points."""
    assert len(x) == len(y)
    return nint(np.linalg.norm(np.array(x) - np.array(y)))


def manhattan_distance(x: TwoD | ThreeD, y: TwoD | ThreeD) -> int:
    """Compute the Manhattan distance (L1 Norm) between two points."""
    assert len(x) == len(y)
    return nint(np.sum(np.abs(np.array(x) - np.array(y))))


def maximum_distance(x: TwoD | ThreeD, y: TwoD | ThreeD) -> int:
    """Compute the maximum distance (Linf Norm) between two points."""
    assert len(x) == len(y)
    return nint(np.max(np.abs(np.array(x) - np.array(y))))


def convert_to_lat_long(x: TwoD) -> tuple[float, float]:
    """
    Convert the lat/lon coordinates from degrees/minutes
    to radians.

    args:
        x: tuple or np.ndarray of length 2

    returns:
        lat: float
        lon: float
    """
    assert len(x) == 2
    coord = np.array(x)
    deg = np.trunc(coord)
    min = coord - deg
    lat, lon = pi * (deg + 5.0 * min / 3.0) / 180.0
    return lat, lon


def geographical_distance(x: TwoD, y: TwoD) -> int:
    """Compute the geographical distance between two points."""
    assert len(x) == len(y) and len(x) == 2
    lat_x, lon_x = convert_to_lat_long(x)
    lat_y, lon_y = convert_to_lat_long(y)

    q1 = np.cos(lon_x - lon_y)
    q2 = np.cos(lat_x - lat_y)
    q3 = np.cos(lat_x + lat_y)
    # np.trunc is equivalent to FORTRAN's INT() function
    return np.trunc(
        RRR * np.arccos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0
    ).astype(np.int32)


def pseudo_euclidean_distance(x: TwoD, y: TwoD) -> int:
    """Compute the pseudo-Euclidean distance between two points."""
    assert len(x) == len(y) and len(x) == 2
    xd = x[0] - y[0]
    yd = x[1] - y[1]
    rij = np.sqrt((xd * xd + yd * yd) / 10.0)
    tij = nint(rij)
    if tij < rij:
        return tij + 1
    else:
        return tij


def ceiling_of_euclidean_distance(x: TwoD, y: TwoD) -> int:
    """Compute the ceiling of the Euclidean distance between two points."""
    assert len(x) == len(y) and len(x) == 2
    return nint(np.ceil(np.linalg.norm(np.array(x) - np.array(y))))


if __name__ == "__main__":
    pytest.main(["-vv", "tests/distances.py"])
