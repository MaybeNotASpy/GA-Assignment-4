from enum import Enum


class PROBLEM_TYPE(Enum):
    TSP = 1
    ATSP = 2
    SOP = 3
    HCP = 4
    CVRP = 5
    TOUR = 6


class EDGE_WEIGHT_TYPE(Enum):
    EXPLICIT = 1
    EUC_2D = 2
    EUC_3D = 3
    MAX_2D = 4
    MAX_3D = 5
    MAN_2D = 6
    MAN_3D = 7
    CEIL_2D = 8
    GEO = 9
    ATT = 10
    XRAY1 = 11
    XRAY2 = 12
    SPECIAL = 13


class COORD_DISPLAY(Enum):
    NO_DISPLAY = 1
    TWOD_DISPLAY = 2
    COORD_DISPLAY = 3


class EDGE_WEIGHT_FORMAT(Enum):
    NONE = 0
    FUNCTION = 1
    FULL_MATRIX = 2
    UPPER_ROW = 3
    LOWER_ROW = 4
    UPPER_DIAG_ROW = 5
    LOWER_DIAG_ROW = 6
    UPPER_COL = 7
    LOWER_COL = 8
    UPPER_DIAG_COL = 9
    LOWER_DIAG_COL = 10