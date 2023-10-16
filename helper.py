import numpy as np


def nint(x: float | np.ndarray) -> int | np.ndarray:
    # np.round is equivalent to FORTRAN's NINT() function
    return np.round(x).astype(np.int32)
