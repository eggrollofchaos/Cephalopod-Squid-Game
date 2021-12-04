# mhr2145
# gc2950
# wax1
import numpy as np
import random
import time
import sys
import os 
from BaseAI import BaseAI
from Grid import Grid
from Utils import manhattan_distance, grid_distance

class PlayerAI(BaseAI):
    def __init__(self) -> None:
        self.cape_color = 'blue'
        super().__init__()
        self.pos = None
        self.player_num = None
        self.optimal_trap_position = None

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
        max_grid = self.__decision(grid, alpha, beta)
        return max_grid.move_position

    def __is_over(self, grid: Grid, turn):
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
        if gameover_result:
            if gameover_result == self.getPlayerNum():
                return grid, 99999
            else:
                return grid, -99999

    def IS_heuristic(self, grid):
        return grid, len(grid.get_neighbors(self.getPlayerPosition(grid), only_available=True)) - len(
            grid.get_neighbors(self.getOpponentPosition(grid), only_available=True))

    def __probability(self, position, trap_position):
        # alpha = manhattan_distance(position, trap_position)
        alpha = grid_distance(position, trap_position)
        p = 1 - 0.05 * (alpha - 1)
        return p

    def __move_children(self, grid: Grid, is_me=True) -> None:
        """
        a player (either us or opponent) moves
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

    def get_all_available_traps(self, grid):
        map_ = grid.getMap()
        open_spots = (map_ == 0).nonzero()
        available_traps = []
        for x, y in zip(list(open_spots[0]), list(open_spots[1])):
            available_traps.append((x, y))
        return available_traps

    def __trap_children(self, grid, is_me=True):
        if is_me:
            player = self.getPlayerNum()
            position = grid.move_position # hypothetical move
            other_position = self.getOpponentPosition(grid)
        else:
            player = self.getOpponentNum()
            position = grid.move_position # hypothetical move
            other_position = self.getPlayerPosition(grid)

        children = []
        available_traps = self.get_all_available_traps(grid)
        for trap_position in available_traps:
            if trap_position != position:
                trap_clone = grid.clone()
                trap_clone.trap(trap_position)
                # dynamically creating class attributes at runtime for access at the very top of the search tree
                trap_clone.trap_position = trap_position
                trap_clone.probability = self.__probability(position, trap_position)
                children.append(trap_clone)

        return children

    def __trap_neighbors(self, grid, trap_position):
        neighbors = []

        for trap_position in grid.get_neighbors(trap_position, only_available=True):
            trap_clone = grid.clone()
            trap_clone.trap(trap_position)
            # dynamically creating class attributes at runtime
            trap_clone.trap_position = trap_position
            neighbors.append(trap_clone)

        return neighbors

    def __trap_minimize(self, grid, alpha, beta, depth, depth_limit):
        gameover_result = self.__is_over(grid, self.getPlayerNum())
        if gameover_result:
            # if game ends because move above results in a gameover, then we need to place a valid trap somewhere randomly
            grid = self.__trap_children(grid, is_me=False)[0]
            return self.__evaluate(grid, gameover_result)

        minTrap, minUtility = None, np.inf

        for trap in self.__trap_children(grid, is_me=False):
            # initialize with the main trap's probability-weighted utility, then move on to those of the neighbors
            _, utility = self.__move_maximize(trap, alpha, beta, depth+1, depth_limit)
            expected_utility = trap.probability * utility

            neighbors = self.__trap_neighbors(grid, trap.trap_position)
            for neighbor in neighbors:
                _, utility = self.__move_maximize(neighbor, alpha, beta, depth+1, depth_limit)
                expected_utility += (1-trap.probability)/len(neighbors) * utility

            if expected_utility < minUtility:
                minUtility = expected_utility

        return minTrap, minUtility

    def __move_minimize(self, grid: Grid, alpha, beta, depth_limit, depth=1) -> tuple:
        gameover_result = self.__is_over(grid, self.getOpponentNum())
        if gameover_result:
            return self.__evaluate(grid, gameover_result)

        # break if hit depth limit
        if depth == depth_limit:
            return self.IS_heuristic(grid)

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

    def __trap_maximize(self, grid, alpha, beta, depth, depth_limit):
        gameover_result = self.__is_over(grid, self.getPlayerNum())
        if gameover_result:
            # if game ends because move above results in a gameover, then we need to place a valid trap somewhere randomly
            grid = self.__trap_children(grid, is_me=True)[0]
            return self.__evaluate(grid, gameover_result)

        maxTrap, maxUtility = None, -np.inf

        for trap in self.__trap_children(grid, is_me=True):
            # initialize with the main trap's probability-weighted utility, then move on to those of the neighbors
            _, utility = self.__move_minimize(trap, alpha, beta, depth_limit, depth+1)
            expected_utility = trap.probability * utility

            neighbors = self.__trap_neighbors(grid, trap.trap_position)
            for neighbor in neighbors:
                _, utility = self.__move_minimize(neighbor, alpha, beta, depth+1, depth_limit)
                expected_utility += (1-trap.probability)/len(neighbors) * utility

            if expected_utility > maxUtility:
                maxTrap, maxUtility = trap, expected_utility

        # returns max trap so maximize can cache it
        return maxTrap, maxUtility

    def __move_maximize(self, grid: Grid, alpha, beta, depth, depth_limit) -> tuple:
        gameover_result = self.__is_over(grid, self.getPlayerNum())
        if gameover_result:
            return self.__evaluate(grid, gameover_result)

        # break if hit depth limit
        if depth == depth_limit:
            return self.IS_heuristic(grid)

        maxMove, maxTrap, maxUtility = None, None, -np.inf

        for move in self.__move_children(grid, is_me=True):
            trap, utility = self.__trap_maximize(move, alpha, beta, depth+1, depth_limit)

            if utility > maxUtility:
                maxMove, maxTrap, maxUtility = move, trap, utility

            if maxUtility >= beta:
                break

            if maxUtility > alpha:
                alpha = maxUtility

        self.optimal_trap_position = maxTrap.trap_position

        return maxMove, maxUtility

    def __decision(self, grid: Grid, alpha, beta, depth_limit=3) -> object:
        start = time.time()
        child, _ = self.__move_maximize(grid, alpha, beta, 1, depth_limit)
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
        return self.optimal_trap_position
