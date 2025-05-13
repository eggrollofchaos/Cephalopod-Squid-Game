"""
# Utilty functions
# Commented by wax
"""
import numpy as np
import math


def manhattan_distance(position, target) -> np.int64:
    """
    Computes the Manhattan Distance between two points on a Cartesian plane.
    Manhattan Distance is the distance in horizontal and vertical units, aka Taxicab Distance.
    This is defined as the sum of the absolute differences of the X and Y coordinates.
    """

    return np.abs(target[0] - position[0]) + np.abs(target[1] - position[1])


def euclidean_distance(position, target) -> float:
    """
    Computes the Euclidian Distance between two points on a Cartesian grid.
    Euclidian Distance is the straight-line distance, aka "as the crow flies".
    This is defined as the square root of the sum of the squares of the differences of the X and Y coordiantes.
       i.e. Application of the Pythagorean Theorem.
    """

    return math.sqrt(np.abs(target[0] - position[0]) ** 2 + np.abs(target[1] - position[1]) ** 2)


def grid_distance(position, target) -> np.int64:
    """
    Computes the Grid Distance between two points on a Cartesian grid.
    Grid Distance is the number of steps between two points in any direction, with diagonals counting as length one, aka Chebyshev Distance.
        e.g. This is how a King would move on a chessboard.
    This is calculated by finding the 
    """

    dx = np.abs(target[0] - position[0])
    dy = np.abs(target[1] - position[1])

    # diagonal_dist = min(dx, dy)
    # cardinal_dist = max(dx, dy) - diagonal_dist

    # return diagonal_dist + cardinal_dist

    return max(dx, dy)