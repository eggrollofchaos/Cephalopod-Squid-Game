# gc2950
# mhr2145
# wax1
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

class PlayerAIOpp(BaseAI):
    def __init__(self, depth_limit=DEFAULT_DEPTH_LIMIT) -> None:
        self.cape_color = 'blue'
        super().__init__()
        self.pos = None
        self.player_num = None
        self.optimal_trap_pos = None
        self.depth_limit = depth_limit
        if self.depth_limit == 0:
            self.depth_limit = DEFAULT_DEPTH_LIMIT
        self.turns = 1              # early game = 1-3, mid = 4-6, late to 7+; generally early game <= grid.dim/2, mid = 2xearly

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
        depth_limit = self.depth_limit
        max_grid = self.__decision(grid, alpha, beta, depth_limit)
        self.turns += 1
        return max_grid.move_pos

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

    def __get_heuristic(self, grid: Grid, is_me) -> int:
        """
        Apply heuristic
        """
        return self.__get_neighbors_heuristic(grid)

    def __get_neighbors_heuristic(self, grid: Grid) -> int:
        """
        Returns the difference in available square around player vs available square around computer.
        """
        return grid, len(grid.get_neighbors(self.getPlayerPosition(grid), only_available=True)) - len(
            grid.get_neighbors(self.getOpponentPosition(grid), only_available=True))

    def __probability(self, pos, trap_pos) -> float:
        """
        Calculates probability of a trap landing in an intended square.
        """
        alpha = manhattan_distance(pos, trap_pos)
        # alpha = grid_distance(pos, trap_pos)
        p = 1 - 0.05 * (alpha - 1)
        return p

    def __move_children(self, grid: Grid, is_me=True) -> list:
        """
        get neighbors of a player's intended Move
        returns a list of Grid objects
        """
        if is_me:
            player = self.getPlayerNum()
            pos = self.getPlayerPosition(grid)
            other_pos = self.getOpponentPosition(grid)
        else:
            player = self.getOpponentNum()
            pos = self.getOpponentPosition(grid)
            other_pos = self.getPlayerPosition(grid)

        children = []
        available_moves = grid.get_neighbors(pos, only_available=True)
        for move_pos in available_moves:
            move_clone = grid.clone()
            move_clone.move(move_pos, player)
            # dynamically creating class attributes at runtime for access in level above in recursive tree
            move_clone.move_pos = move_pos
            children.append(move_clone)

        return children

    def __get_all_available_traps(self, grid: Grid, pos) -> list:
        """
        helper function to get a list of open spots on the board, threshold of 3 squares away from position param
        returns a list of tuples (positions)
        """
        all_available_pos = grid.getAvailableCells()
        available_traps = []
        threshold = 3
        for trap_pos in all_available_pos:
            if grid_distance(pos, trap_pos) < threshold:
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
            pos = grid.move_pos # hypothetical move
            other_pos = self.getOpponentPosition(grid)
        else:
            player = self.getOpponentNum()
            pos = grid.move_pos # hypothetical move
            other_pos = self.getPlayerPosition(grid)

        children = []
        available_traps = self.__get_all_available_traps(grid, other_pos)
        for trap_pos in available_traps:
            if trap_pos != pos:
                trap_clone = grid.clone()
                trap_clone.trap(trap_pos)
                # dynamically creating class attributes at runtime for access at the very top of the search tree
                trap_clone.trap_pos = trap_pos
                trap_clone.probability = self.__probability(pos, trap_pos)
                children.append(trap_clone)

        return children

    def __trap_neighbors(self, grid: Grid, trap_pos) -> list:
        """
        get neighbors of intended trap_pos
        returns a list of Grid objects
        """
        neighbors = []

        for trap_pos in grid.get_neighbors(trap_pos, only_available=True):
            trap_clone = grid.clone()
            trap_clone.trap(trap_pos)
            # dynamically creating class attributes at runtime
            trap_clone.trap_pos = trap_pos
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
        if depth >= depth_limit:
            return self.__get_heuristic(grid, is_me=True)

        # input()
        minTrap, minUtility = None, np.inf

        for trap in self.__trap_children(grid, is_me=False):
            # initialize with the main trap's probability-weighted utility, then move on to those of the neighbors
            _, utility = self.__move_maximize(trap, alpha, beta, depth+1, depth_limit)
            expected_utility = trap.probability * utility

            neighbors = self.__trap_neighbors(grid, trap.trap_pos)
            for neighbor in neighbors:
                _, utility = self.__move_maximize(neighbor, alpha, beta, depth+1, depth_limit)
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
            return self.__get_heuristic(grid, is_me=True)

        # input()
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

        if depth >= depth_limit:
            return self.__get_heuristic(grid, is_me=True)

        maxTrap, maxUtility = None, -np.inf
        cache = {}
        for trap in self.__trap_children(grid, is_me=True):
            # initialize with the main trap's probability-weighted utility, then move on to those of the neighbors
            _, utility = self.__move_minimize(trap, alpha, beta, depth+1, depth_limit)
            expected_utility = trap.probability * utility

            neighbors = self.__trap_neighbors(grid, trap.trap_pos)
            for neighbor in neighbors:
                _, utility = self.__move_minimize(neighbor, alpha, beta, depth+1, depth_limit)
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

        # break if hit depth limit
        if depth == depth_limit:
            return self.__get_heuristic(grid, is_me=True)

        maxMove, maxTrap, maxUtility = None, None, -np.inf

        for move in self.__move_children(grid, is_me=True):
            trap, utility = self.__trap_maximize(move, alpha, beta, depth+1, depth_limit)

            if utility > maxUtility:
                maxMove, maxTrap, maxUtility = move, trap, utility

            if maxUtility >= beta:
                break

            if maxUtility > alpha:
                alpha = maxUtility

        if hasattr(maxTrap, 'trap_pos'):
            self.optimal_trap_pos = maxTrap.trap_pos

        return maxMove, maxUtility

    def __decision(self, grid: Grid, alpha, beta, depth_limit=4) -> object:
        """
        Helper function to start the Expectiminimax algo
        returns a Grid object
        """
        start = time.time()
        child, _ = self.__move_maximize(grid, alpha, beta, depth=1, depth_limit=depth_limit)
        end = time.time()
        print(f'This move took {end-start:.5f} seconds.')
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
        if self.optimal_trap_pos is None:
            return grid.getAvailableCells()[0]
        return self.optimal_trap_pos
