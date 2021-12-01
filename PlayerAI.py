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
    
    def getPosition(self):
        return self.pos

    def setPosition(self, new_position):
        self.pos = new_position 

    def getPlayerNum(self):
        return self.player_num

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
        grid.print_grid()
        player_pos = self.getPosition()
        available_moves = grid.get_neighbors(player_pos, only_available = True)
        # search in available moves

        # find opponent
        player_num = self.getPlayerNum()
        opp_num = 3 - player_num
        opp_coord = grid.find(opp_num)

        # compute Manhattan distance from opponent to all of player's neighbor's
        for move in available_moves:


            # manhattan_distance(player_pos, moves)

        return move


    def evaluate_grid(grid: Grid) -> int:

        gameover_result = Game.is_over(grid)
        if gameover_result:
            if gameover_result == player_num:
                return 1
            else:
                return -1

    def __minimize(grid : Grid) -> tuple:
        if evaluate_grid(grid):
            return 1

        minChild, minUtility = None, np.inf

        for child in grid.children():
            _, utility = self.__maximize(child)

            if utility < minUtility:
                minChild, minUtility = child, utility

        return minChild, minUtility

    def __maximize(grid : Grid) -> tuple:

        if Game.is_over(grid):
            return None, evaluate(grid)

        maxChild, maxUtility = None, -np.inf

        for child in grid.children():
            _, utility = self.__minimize(child)

            if utility < maxUtility:
                maxChild, maxUtility = child, utility

        return maxChild, maxUtility

    def __decision(grid : Grid) -> object:
        child, _ = self.__maximize(grid)

        return child

    def getTrap(self, grid : Grid) -> tuple:
        """ 
        YOUR CODE GOES HERE

        The function should return a tuple of (x,y) coordinates to which the player *WANTS* to throw the trap.
        
        It should be the result of the ExpectiMinimax algorithm, maximizing over the Opponent's *Move* actions, 
        taking into account the probabilities of it landing in the positions you want. 
        
        Note that you are not required to account for the probabilities of it landing in a different cell.

        You may adjust the input variables as you wish (though it is not necessary). Output has to be (x,y) coordinates.
        
        """
        available_cells = grid.getAvailableCells()
        # search in available cells and make a judgment
