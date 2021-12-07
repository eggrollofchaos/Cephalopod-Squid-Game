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

def get_neighbors(grid, pos, radius=1, only_available = False):
    """
    Description
    -----------
    The function returns the neighboring cells of a certain cell in the board, given its x,y coordinates

    Parameters
    -----------
    pos : position (x,y) whose neighbors are desired

    only_available (bool) : if True, the function will return only available neighboring cells. 
                            default = False

    radius : search distance in grid_distance, int, range of [0, 6]
    
    """
    x,y = pos
    
    valid_range = lambda t: range(max(t-radius, 0), min(t+radius+1, grid.dim))

    # find all neighbors
    neighbors = list({(a,b) for a in valid_range(x) for b in valid_range(y)} - {(x,y)})
    
    # select only neighboring cells which aren't occupied by a player or trap
    if only_available:
        return [neighbor for neighbor in neighbors if grid.map[neighbor] == 0]
    
    return neighbors