# wax
import numpy as np
import random
import time
import sys
import os
import queue as Q
from BaseAI import BaseAI
from Grid import Grid
from Utils import manhattan_distance, grid_distance

DEFAULT_DEPTH_LIMIT = 2

class PlayerAIOppV1(BaseAI):
    """
    Custom AI Opponent Version 1.
    Uses Expectiminimax.
    Only applies n-neighbors heuristics.
    Set DEFAULT_DEPTH_LIMIT = 2.
    """

    def __init__(self, depth_limit = DEFAULT_DEPTH_LIMIT, verbose = 0) -> None:

        super().__init__()
        # self.cape_color = 'blue'
        self.verbose = verbose
        self.pos = None
        self.player_num = None
        self.optimal_trap_position = None
        self.depth_limit = depth_limit
        if self.depth_limit < 1:
            self.depth_limit = DEFAULT_DEPTH_LIMIT
        self.turns = 1              # early game = 1-3, mid = 4-6, late to 7+; generally early game <= grid.dim/2, mid = 2xearly
        # self.use_advanced_heuristics = heur        # none implemented

    def getPosition(self):
        return self.pos

    def getPlayerPosition(self, grid):
        return grid.find(self.getPlayerNum())

    def setPosition(self, new_position):
        self.pos = new_position

    def getPlayerNum(self):
        return self.player_num

    def getOpponentNum(self):
        return 3 - self.getPlayerNum()

    def getOpponentPosition(self, grid: Grid):
        return grid.find(self.getOpponentNum())

    def setPlayerNum(self, num):
        self.player_num = num

    def getMove(self, grid: Grid) -> tuple:
        """
        YOUR CODE GOES HERE

        The function should return a tuple of (x,y) coordinates to which the player moves.

        It should be the result of the ExpectiMinimax algorithm, maximizing over the Opponent's *Trap* actions, 
        taking into account the probabilities of them landing in the positions you believe they'd throw to.

        Note that you are not required to account for the probabilities of it landing in a different cell.

        You may adjust the input variables as you wish (though it is not necessary). Output has to be (x,y) coordinates.
        
        """
        alpha = -np.inf
        beta = np.inf
        max_grid = self.__decision(grid, alpha, beta, self.depth_limit)
        self.turns += 1
        return max_grid.move_position

    def __is_over(self, grid: Grid, turn) -> int:
        """
        Check if game is over, i.e., Player or Opponent has no moves to make
        """
        # check if Player has won
        opponent_neighbors = grid.get_neighbors(self.getOpponentPosition(grid), only_available=True)
        if len(opponent_neighbors) == 0:
            return self.getPlayerNum()

        # check if Opponent has won
        player_neighbors = grid.get_neighbors(self.getPlayerPosition(grid), only_available=True)
        if len(player_neighbors) == 0:
            return self.getOpponentNum()

    def __evaluate(self, grid: Grid, gameover_result) -> int:
        """
        Function for returning high or low utility based on gameover state.
        """
        if gameover_result:
            if gameover_result == self.getPlayerNum():
                return grid, 99999
            else:
                return grid, -99999

    def __get_heuristics(self, grid: Grid, is_me) -> tuple:
        """
        Apply heuristics
        Returns a tuple of Grid object, heuristic int
        """

        return grid, self.__n_neighbors_heur(grid, is_me=True) - self.__n_neighbors_heur(grid, is_me=False)

    def __n_neighbors_heur(self, grid: Grid, is_me=True) -> int:
        """
        Returns the difference in available squares around player vs available squares around opponent.
        """
        if is_me:
            pos = self.getPlayerPosition(grid)
            # other_pos = self.getOpponentPosition(grid)
        else:
            pos = self.getOpponentPosition(grid)
            # other_pos = self.getPlayerPosition(grid)
        # return grid, len(grid.get_neighbors(self.getPlayerPosition(grid), only_available=True)) - len(
        #     grid.get_neighbors(self.getOpponentPosition(grid), only_available=True))
        # return ( len(grid.get_neighbors(pos, only_available=True)) - len(
        #     grid.get_neighbors(other_pos, only_available=True)) )
        return len(grid.get_neighbors(pos, only_available=True))

    def __probability(self, position, trap_position) -> float:
        """
            Calculates probability of a trap landing in an intended square.
        """
        alpha = manhattan_distance(position, trap_position)
        # alpha = grid_distance(position, trap_position)
        p = 1 - 0.05 * (alpha - 1)
        return p

    def __move_children(self, grid: Grid, is_me=True) -> list:
        """
        get neighbors of a player's intended Move
        returns a list of Grid objects
        """
        if is_me:
            player = self.getPlayerNum()
            position = self.getPlayerPosition(grid)
            other_position = self.getOpponentPosition(grid)
        else:
            player = self.getOpponentNum()
            position = self.getOpponentPosition(grid)
            other_position = self.getPlayerPosition(grid)

        children = []
        available_moves = grid.get_neighbors(position, only_available=True)
        for move_position in available_moves:
            move_clone = grid.clone()
            move_clone.move(move_position, player)
            # dynamically creating class attributes at runtime for access in level above in recursive tree
            move_clone.move_position = move_position
            children.append(move_clone)

        return children

    def __get_all_available_traps(self, grid: Grid, position) -> list:
        """
        helper function to get a list of open spots on the board
        default is threshold of 3 squares away from position param
        returns a list of tuples (positions)
        """
        all_available_pos = grid.getAvailableCells()
        available_traps = []
        threshold = 2
        for trap_pos in all_available_pos:
            # if manhattan_distance(position, trap_pos) < threshold:
            if grid_distance(position, trap_pos) < threshold:
                available_traps.append(trap_pos)
        return available_traps

    def __trap_children(self, grid: Grid, is_me=True) -> list:
        """
        a player (either player or opponent) throws Trap
        function expands node of trades from param position
        returns a list of Grid objects
        """
        if is_me:
            player = self.getPlayerNum()
            position = grid.move_position # hypothetical move
            other_position = self.getOpponentPosition(grid)
        else:
            player = self.getOpponentNum()
            position = grid.move_position # hypothetical move
            other_position = self.getPlayerPosition(grid)

        children = []
        available_traps = self.__get_all_available_traps(grid, other_position)
        for trap_position in available_traps:
            if trap_position != position:
                trap_clone = grid.clone()
                trap_clone.trap(trap_position)
                # dynamically creating class attributes at runtime for access at the very top of the search tree
                trap_clone.trap_position = trap_position
                trap_clone.probability = self.__probability(position, trap_position)
                children.append(trap_clone)

        return children

    def __trap_neighbors(self, grid: Grid, trap_position) -> list:
        """
        get neighbors of intended trap_pos
        returns a list of Grid objects
        """
        neighbors = []

        for trap_position in grid.get_neighbors(trap_position, only_available=True):
            trap_clone = grid.clone()
            trap_clone.trap(trap_position)
            # dynamically creating class attributes at runtime
            trap_clone.trap_position = trap_position
            neighbors.append(trap_clone)

        return neighbors

    def __trap_minimize(self, grid: Grid, alpha, beta, depth, depth_limit) -> tuple:
        """
        opponent Min node for throwing Trap
        returns a list of Grid objects
        """
        gameover_result = self.__is_over(grid, self.getPlayerNum())
        if gameover_result:
            # if game ends because move above results in a gameover, then we need to place a valid trap somewhere randomly
            # grid = self.__trap_children(grid, is_me=False)[0]
            return self.__evaluate(grid, gameover_result)
        
        # break if hit depth limit
        if depth >= depth_limit:
            return self.__get_heuristics(grid, is_me=True)

        minTrap, minUtility = None, np.inf
        cache = {}

        for trap in self.__trap_children(grid, is_me=False):
            # initialize with the main trap's probability-weighted utility, then move on to those of the neighbors
            key = tuple(map(tuple, trap.map))
            if key in cache:
                utility = cache[key]
            else:
                _, utility = self.__move_maximize(trap, alpha, beta, depth+1, depth_limit)
                cache[key] = utility
            expected_utility = trap.probability * utility

            neighbors = self.__trap_neighbors(grid, trap.trap_position)
            for neighbor in neighbors:
                key = tuple(map(tuple, neighbor.map))
                if key in cache:
                    utility = cache[key]
                else:
                    _, utility = self.__move_maximize(neighbor, alpha, beta, depth+1, depth_limit)
                    cache[key] = utility
                expected_utility += (1-trap.probability)/len(neighbors) * utility

            if expected_utility < minUtility:
                minUtility = expected_utility
        return minTrap, minUtility

    def __move_minimize(self, grid: Grid, alpha, beta, depth, depth_limit) -> tuple:
        """
        opponent Min node for making Move
        returns a list of Grid objects
        """
        gameover_result = self.__is_over(grid, self.getOpponentNum())
        if gameover_result:
            return self.__evaluate(grid, gameover_result)

        # break if hit depth limit
        if depth >= depth_limit:
            return self.__get_heuristics(grid, is_me=True)

        minChild, minUtility = None, np.inf

        for child in self.__move_children(grid, is_me=False):
            _, utility = self.__trap_minimize(child, alpha, beta, depth+1, depth_limit)

            if utility < minUtility:
                minChild, minUtility = child, utility

            if minUtility <= alpha:
                break

            if minUtility < beta:
                beta = minUtility

        return minChild, minUtility

    def __trap_maximize(self, grid: Grid, alpha, beta, depth, depth_limit) -> tuple:
        """
        player Max node for throwing Trap
        returns a list of Grid objects
        """
        gameover_result = self.__is_over(grid, self.getPlayerNum())
        if gameover_result:
            # if game ends because move above results in a gameover, then we need to place a valid trap somewhere randomly
            # grid = self.__trap_children(grid, is_me=True)[0]
            return self.__evaluate(grid, gameover_result)

        # break if (exceed) hit depth limit
        if depth > depth_limit:
            return self.__get_heuristics(grid, is_me=True)

        maxTrap, maxUtility = None, -np.inf
        cache = {}
        for trap in self.__trap_children(grid, is_me=True):
            # initialize with the main trap's probability-weighted utility, then move on to those of the neighbors
            key = tuple(map(tuple, trap.map))
            if key in cache:
                utility = cache[key]
            else:
                _, utility = self.__move_minimize(trap, alpha, beta, depth+1, depth_limit)
                cache[key] = utility
            expected_utility = trap.probability * utility

            neighbors = self.__trap_neighbors(grid, trap.trap_position)
            for neighbor in neighbors:
                key = tuple(map(tuple, neighbor.map))
                if key in cache:
                    utility = cache[key]
                else:
                    _, utility = self.__move_minimize(neighbor, alpha, beta, depth+1, depth_limit)
                    cache[key] = utility
                expected_utility += (1-trap.probability)/len(neighbors) * utility

            if expected_utility > maxUtility:
                maxTrap, maxUtility = trap, expected_utility
        # returns max trap so maximize can cache it
        return maxTrap, maxUtility

    def __move_maximize(self, grid: Grid, alpha, beta, depth, depth_limit) -> tuple:
        """
        player Min node for making Move
        returns a list of Grid objects
        """
        gameover_result = self.__is_over(grid, self.getPlayerNum())
        if gameover_result:
            return self.__evaluate(grid, gameover_result)

        # break if (exceed) hit depth limit
        if depth > depth_limit:
            return self.__get_heuristics(grid, is_me=True)

        maxMove, maxTrap, maxUtility = None, None, -np.inf

        for move in self.__move_children(grid, is_me=True):
            trap, utility = self.__trap_maximize(move, alpha, beta, depth+1, depth_limit)
            if utility > maxUtility:
                maxMove, maxTrap, maxUtility = move, trap, utility

            if maxUtility >= beta:
                break

            if maxUtility > alpha:
                alpha = maxUtility

        if hasattr(maxTrap, 'trap_position'):
            self.optimal_trap_position = maxTrap.trap_position

        return maxMove, maxUtility

    def __decision(self, grid: Grid, alpha, beta, depth_limit=DEFAULT_DEPTH_LIMIT) -> object:
        """
        Helper function to start the Expectiminimax algo
        returns a Grid object
        """
        start = time.time()
        child, _ = self.__move_maximize(grid, alpha, beta, depth=0, depth_limit=depth_limit)
        end = time.time()
        # print(f'This move took {end-start:.5f} seconds.')
        return child

    def getTrap(self, grid: Grid) -> tuple:
        """
        YOUR CODE GOES HERE

        The function should return a tuple of (x,y) coordinates to which the player *WANTS* to throw the trap.
        
        It should be the result of the ExpectiMinimax algorithm, maximizing over the Opponent's *Move* actions, 
        taking into account the probabilities of it landing in the positions you want. 
        
        Note that you are not required to account for the probabilities of it landing in a different cell.

        You may adjust the input variables as you wish (though it is not necessary). Output has to be (x,y) coordinates.
        
        """
        # use cached optimal trap position that we computed in getMove()
        if self.optimal_trap_position is None:
            return grid.getAvailableCells()[0]
        return self.optimal_trap_position