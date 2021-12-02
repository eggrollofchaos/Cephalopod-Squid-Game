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
from Utils import manhattan_distance

class PlayerAI(BaseAI):
    def __init__(self) -> None:
        self.cape_color = 'blue'
        super().__init__()
        self.pos = None
        self.player_num = None
        self.over = False
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
        """Check if game is over, i.e., Player or Opponent has no moves to make"""
        # check if Player has won
        # find available neighbors of player 1
        opponent_neighbors = grid.get_neighbors(self.getOpponentPosition(grid), only_available=True)
        # if none - win
        if len(opponent_neighbors) == 0:
            self.over = True
            return 1

        # check if Opponent has won
        player_neighbors = grid.get_neighbors(self.getPlayerPosition(grid), only_available=True)

        if len(player_neighbors) == 0:
            self.over = True
            return 2
        
        elif self.over:
            return turn

        else: 
            return 0

    def __evaluate(self, grid: Grid, gameover_result) -> int:
        if gameover_result:
            if gameover_result == self.getPlayerNum():
                return grid, 1
            else:
                return grid, -1

    def __probability(self, position, trap_position):
        alpha = manhattan_distance(position, trap_position)
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
            move_clone.move_position = move_position
            children.append(move_clone)

        return children

    def __trap_children(self, grid, is_me=True):
        if is_me:
            player = self.getPlayerNum()
            position = self.getPlayerPosition(grid)
            other_position = self.getOpponentPosition(grid)
        else:
            player = self.getOpponentNum()
            position = self.getOpponentPosition(grid)
            other_position = self.getPlayerPosition(grid)

        children = []
        available_traps = grid.get_neighbors(other_position, only_available=True)
        for trap_position in available_traps:
            if trap_position != position:
                trap_clone = grid.clone()
                trap_clone.trap(trap_position)
                # dynamically creating class attributes at runtime for access at the very top of the search tree
                trap_clone.trap_position = trap_position
                trap_clone.probability = self.__probability(position, trap_position)
                children.append(trap_clone)

        return children

    def __move_minimize(self, grid: Grid, alpha, beta) -> tuple:
        gameover_result = self.__is_over(grid, self.getOpponentNum())
        if gameover_result:
            return self.__evaluate(grid, gameover_result)

        minChild, minUtility = None, np.inf

        for child in self.__move_children(grid, is_me=False):
            utility = self.__random_trap(child, alpha, beta, parent_type="minimize")

            if utility < minUtility:
                minChild, minUtility = child, utility

            if minUtility <= alpha:
                break

            if minUtility < beta:
                beta = minUtility

        return minChild, minUtility

    def __random_trap(self, grid: Grid, alpha, beta, parent_type="maximize"):
        """
        Returns expected value of trap
        """
        maxChild, maxUtility = None, -np.inf

        expected_utility = 0
        if parent_type == "maximize":
            is_me = True
        else:
            is_me = False
        for child in self.__trap_children(grid, is_me=is_me):
            if parent_type == "maximize":
                _, utility = self.__move_minimize(child, alpha, beta)
            else:
                _, utility = self.__move_maximize(child, alpha, beta)

            expected_utility += child.probability * utility

            if utility > maxUtility:
                maxChild, maxUtility = child, utility

        self.optimal_trap_position = maxChild.trap_position

        return expected_utility

    def __move_maximize(self, grid: Grid, alpha, beta) -> tuple:
        gameover_result = self.__is_over(grid, self.getPlayerNum())
        if gameover_result:
            return self.__evaluate(grid, gameover_result)

        maxChild, maxUtility = None, -np.inf

        for child in self.__move_children(grid, is_me=True):
            utility = self.__random_trap(child, alpha, beta, parent_type="maximize")

            if utility > maxUtility:
                maxChild, maxUtility = child, utility

            if maxUtility >= beta:
                break

            if maxUtility > alpha:
                alpha = maxUtility

        return maxChild, maxUtility

    def __decision(self, grid: Grid, alpha, beta) -> object:
        start = time.time()
        child, _ = self.__move_maximize(grid, alpha, beta)
        end = time.time()
        print(f'This move took {end-start:.5f} seconds.')
        self.over = False
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
