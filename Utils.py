import numpy as np
import math

def manhattan_distance(position, target):
        return np.abs(target[0] - position[0]) + np.abs(target[1] - position[1])


def euclidean_distance(position, target):
    return math.sqrt(np.abs(target[0] - position[0]) ** 2 + np.abs(target[1] - position[1]) ** 2)

def grid_distance(position, target):
    dx = np.abs(target[0] - position[0])
    dy = np.abs(target[1] - position[1])

    diagonal_dist = min(dx, dy);
    cardinal_dist = max(dx, dy) - diagonal_dist

    return diagonal_dist + cardinal_dist