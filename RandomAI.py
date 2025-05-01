
from random import choice as rand_choice




from BaseAI import BaseAI
from Grid import Grid

class RandomAI(BaseAI):
    """
    Random AI.
    Debug AI for testing purposes.
    Selects a random move and trap position.
    """

    def __init__(self, initial_position = None, verbose = 0) -> None:
        super().__init__()
        print('Running RandomAI()...') if verbose else None
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
        """ RandomAI (not really an AI) returns a random, valid trap position. """
        
        # find all available cells in the grid
        available_cells = grid.getAvailableCells()

        # choose a random trap position
        trap_pos = rand_choice(available_cells) if available_cells else None

        return trap_pos


