"""
HumanOpp Class module.
Contains logic for handling a human opponent's inputs.
Authored and fully commented by WAX.
"""
import os 
from platform import system as os_type
import random
import sys

import numpy as np
from termcolor import cprint

from BaseAI import BaseAI
from Grid import Grid
from Utils import manhattan_distance, grid_distance

is_unix = os_type()
OPPONENT = lambda player: 3 - player                    # TODO: may not be needed?
TO_CLEAN = str.maketrans('', '', ".()")                 # for sanitizing user input

class HumanOpp(BaseAI):
    def __init__(self, initial_position: tuple[int, int] = None, verbose: int = 0) -> None:
        super().__init__()
        print('Running HumanOpp()...') if verbose else None
        self.verbose = verbose
        self.pos = initial_position
        self.player_num = None

    def setPosition(self, new_pos: tuple[int, int]):
        self.pos = new_pos
    
    def getPosition(self):
        return self.pos 

    def setPlayerNum(self, num: int):
        self.player_num = num

    def getMove(self, grid: Grid):
        """
        Request keyboard input from human player as coordinates to Move to.
        Input must be in the form of a tuple, e.g. (1,3).
        """
        
        # find all available moves 
        available_moves = grid.get_neighbors(self.pos, only_available = True)

        # make random move
        # new_pos = random.choice(available_moves) if available_moves else None
        
        # valid_move = False
        
        # get Move coordinates
        # while not valid_move:
                
        while True:
            pos_input = input('Where will you move to? Enter the coordinates (row #, column #): ')
            
            # x, y, *rest = tuple(map(int, pos_input.translate(TO_CLEAN).strip().split(" ")))           # sanitize, take first 2 numbers
            
            try:
                pos_input_list = pos_input.translate(TO_CLEAN).replace(" ", ",").strip().split(",")                       # sanitize
                x, y, *rest = tuple(map(int, [num.strip() for num in pos_input_list if num]))           # sanitize more, take first 2 numbers
            except ValueError:
                cprint('Invalid entry, please try again.', 'yellow') if is_unix else print('Invalid entry, please try again.')
                continue
                
            cprint(f'You\'ve entered ({x},{y}) as your move.', 'yellow') if self.verbose and is_unix else None
            
            try:
                cell_value = int(grid.getCellValue((x,y)))
            except IndexError:
                cprint(f'Invalid square: ({x},{y}) does not exist on this game board.', 'yellow') if is_unix else print(f'Invalid square: ({x},{y}) does not exist on this game board.')
                continue
                
            if (x,y) in available_moves:
                return x,y
            else:
                cprint('That move is invalid.', 'yellow') if is_unix else print('That move is invalid.')
                print(f'Intended square ({x}, {y}) has value of {cell_value} and is a distance {grid_distance(self.pos,(x,y))} from current position.') if self.verbose else None

    def getTrap(self, grid: Grid):

        """
        Request keyboard input from human player as coordinates to which a Trap should be thrown.
        Input must be in the form of a tuple, e.g. (1,3).
        """
        
        # find opponent
        opp_pos = grid.find(3 - self.player_num)
        
        # find all available cells surrounding Opponent
        # available_cells = grid.get_neighbors(opp_pos, only_available=True)
        
        # find all available cells on board
        available_cells = grid.getAvailableCells()
            
        # get Trap coordinates
        while True:
            trap_input = input('Where to throw the trap? Enter the coordinates: ')
            # x, y, *rest = tuple(map(int, trap_input.translate(TO_CLEAN).strip().split(" ")))          # sanitize, take first 2 numbers
            
            try:
                trap_input_list = trap_input.translate(TO_CLEAN).replace(" ", ",").strip().split(",")                     # sanitize
                x, y, *rest = tuple(map(int, [num.strip() for num in trap_input_list if num]))          # sanitize more, take first 2 numbers
            except ValueError:
                cprint('Invalid entry, please try again.', 'yellow') if is_unix else print('Invalid entry, please try again.')
                continue
            
            cprint(f'You\'ve entered ({x},{y}) as your intended trap position.', 'red') if self.verbose and is_unix else None
            
            try:
                cell_value = int(grid.getCellValue((x,y)))
            except IndexError:
                cprint(f'Invalid square: ({x},{y}) does not exist on this game board.', 'yellow') if is_unix else print(f'Invalid square: ({x},{y}) does not exist on this game board.')
                continue
            
            if (x,y) in available_cells:
                return x,y
            else:
                cprint('That trap position is invalid.', 'yellow') if is_unix else print('That trap position is invalid.')
                print(f'Intended square ({x}, {y}) has value of {cell_value} and is a distance {grid_distance(opp_pos,(x,y))} from opponent\'s position.') if self.verbose else None
                
