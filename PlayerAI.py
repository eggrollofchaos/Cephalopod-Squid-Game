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
        # self.alpha = -np.inf
        # self.beta = np.inf

    def getPosition(self):
        return self.pos

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
        self.optimal_trap_position = max_grid.trap_position
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
        player_neighbors = grid.get_neighbors(self.getPosition(), only_available=True)

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

    def __children(self, grid: Grid, is_me=True) -> None:
        """
        every turn,
        1) a player (either us or opponent) first moves
        2) and then throws a trap
        """
        if is_me:
            player = self.getPlayerNum()
            position = self.getPosition()
            other_position = self.getOpponentPosition(grid)
        else:
            player = self.getOpponentNum()
            position = self.getOpponentPosition(grid)
            other_position = self.getPosition()

        children = []
        available_moves = grid.get_neighbors(position, only_available=True)
        for move_position in available_moves:
            move_clone = grid.clone()
            move_clone.move(move_position, player)
            # get neighbors for available traps from the new grid *after* the move (AKA move_clone)
            available_traps = move_clone.get_neighbors(other_position, only_available=True)
            for trap_position in available_traps:
                trap_clone = move_clone.clone()
                trap_clone.trap(trap_position)
                # dynamically creating class attributes at runtime for access at the very top of the search tree
                trap_clone.move_position = move_position
                trap_clone.trap_position = trap_position
                children.append(trap_clone)
        return children

    def __minimize(self, grid: Grid, alpha, beta) -> tuple:
        gameover_result = self.__is_over(grid, self.getOpponentNum())
        if gameover_result:
            return self.__evaluate(grid, gameover_result)

        minChild, minUtility = None, np.inf

        for child in self.__children(grid, is_me=False):
            _, utility = self.__maximize(child, alpha, beta)

            if utility < minUtility:
                minChild, minUtility = child, utility

            if minUtility <= alpha:
                break

            if minUtility < beta:
                beta = minUtility

        return minChild, minUtility

    def __maximize(self, grid: Grid, alpha, beta) -> tuple:
        gameover_result = self.__is_over(grid, self.getPlayerNum())
        if gameover_result:
            return self.__evaluate(grid, gameover_result)

        maxChild, maxUtility = None, -np.inf

        for child in self.__children(grid, is_me=True):
            _, utility = self.__minimize(child, alpha, beta)

            if utility > maxUtility:
                maxChild, maxUtility = child, utility

            if maxUtility >= beta:
                break

            if maxUtility > alpha:
                alpha = maxUtility

        return maxChild, maxUtility

    def __decision(self, grid: Grid, alpha, beta) -> object:
        start = time.time()
        child, _ = self.__maximize(grid, alpha, beta)
        end = time.time()
        print(f'This move took {end-start:.3f} seconds.')
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
