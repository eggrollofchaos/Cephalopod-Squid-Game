
from random import choice as rand_choice
import sys
import os 

sys.path.append(os.getcwd())        # setting path to parent directory
from BaseAI import BaseAI
from Grid import Grid

class EasyAI(BaseAI):
    """
    Easy Computer AI player.
    Move AI is random.
    Trap AI chooses random neighboring square of opponent.
    """

    def __init__(self, initial_position = None, verbose = 0) -> None:
        super().__init__()
        print('Running EasyAI()...') if verbose else None
        self.verbose = verbose
        self.pos = initial_position
        self.player_num = None

    def setPosition(self, new_pos: tuple) -> None:
        self.pos = new_pos
    
    def getPosition(self) -> tuple:
        return self.pos 

    def setPlayerNum(self, num) -> None:
        self.player_num = num

    def getMove(self, grid) -> tuple:
        """ Returns a random, valid move. """
        
        # find all available moves 
        available_moves = grid.get_neighbors(self.pos, only_available = True)

        # choose random move
        new_pos = rand_choice(available_moves) if available_moves else None
        
        return new_pos

    def getTrap(self, grid : Grid) -> tuple:
        """ EasyAI throws randomly to the immediate neighbors of the opponent. """
        
        # find opponent
        opponent = grid.find(3 - self.player_num)
        
        # find all available cells surrounding opponent
        available_neighbors = grid.get_neighbors(opponent, only_available=True)

        # throw to one of the available cells randomly (unless now occupying the last space)
        if available_neighbors:
            trap_pos = rand_choice(available_neighbors)
        else:
            print('EasyAI')
            input(f'No available cells around player {3 - self.player_num}! Press enter to continue.') if self.verbose else None
            trap_pos = opponent

        return trap_pos