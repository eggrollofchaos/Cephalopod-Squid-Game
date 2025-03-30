import numpy as np
import random
import sys
import os 
# setting path to parent directory
# sys.path.append(os.getcwd())
from BaseAI import BaseAI
from Grid import Grid
from platform import system as os_type
from termcolor import cprint
from Utils import manhattan_distance, grid_distance


is_unix = os_type()
OPPONENT = lambda player: 3 - player
TO_CLEAN = str.maketrans('', '', ".()")

class HumanOpp(BaseAI):



    def __init__(self, initial_position = None, verbose = 0) -> None:
        super().__init__()
        self.pos = initial_position
        self.player_num = None
        self.verbose = verbose

    def setPosition(self, new_pos: tuple):
        self.pos = new_pos
    
    def getPosition(self):
        return self.pos 

    def setPlayerNum(self, num):
        self.player_num = num

    def getMove(self, grid):
        '''
        Requests keyboard input from human player as coordinates to Move to.
        Input must be in the form of a tuple, e.g. (1,3).
        '''
        
        # find all available moves 
        available_moves = grid.get_neighbors(self.pos, only_available = True)

        # make random move
        # new_pos = random.choice(available_moves) if available_moves else None
        
        # valid_move = False
        
        # get Move coordinates
        # while not valid_move:
                
        while True:
            pos_input = input('Where will you move to? Enter the coordinates: ')
            
            # x, y, *rest = tuple(map(int, pos_input.translate(TO_CLEAN).strip().split(" ")))           # sanitize, take first 2 numbers
            
            try:
                pos_input_list = pos_input.translate(TO_CLEAN).replace(" ", ",").strip().split(",")                       # sanitize
                x, y, *rest = tuple(map(int, [num.strip() for num in pos_input_list if num]))           # sanitize more, take first 2 numbers
            except ValueError:
                cprint('Invalid entry, please try again.', 'yellow') if is_unix else print('Invalid entry, please try again.')
                continue
                
            cprint(f'You\'ve entered ({x},{y}) as your move.', 'green') if self.verbose and is_unix else None
            
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

    def getTrap(self, grid : Grid):

        '''
        Requests keyboard input from human player as coordinates to throw a Trap to. 
        Input must be in the form of a tuple, e.g. (1,3).
        '''
        
        # find opponent
        opp_pos = grid.find(3 - self.player_num)
        
        # find all available cells surrounding Opponent
        # available_cells = grid.get_neighbors(opp_pos, only_available=True)
        
        # find all available cells on board
        available_cells = grid.getAvailableCells()

        # throw to one of the available cells randomly (unless now occupying the last space)
        # if available_cells:
            # trap = random.choice(available_cells)
        # else:
            # trap = opp_pos
            
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
                
