"""
MediumAI module.
Default AI for opponent player.
Fully commented by WAX.
"""

import os
from random import choice as rand_choice
import sys

import numpy as np

sys.path.append(os.getcwd())  # setting path to parent directory
from BaseAI import BaseAI
from Grid import Grid


class MediumAI(BaseAI):
    """
    Medium AI, default AI for opponent player.
    Move AI finds move that maximizes number of available moves for current player.
    Trap AI finds position that maximizes difference between current player's available moves vs opponent's.
    """

    def __init__(
        self, verbose: int = 0, initial_position: tuple[int, int] = None
    ) -> None:
        super().__init__()
        print("Running MediumAI()...") if verbose else None
        self.verbose = verbose
        self.pos = initial_position
        self.player_num = None

    def setPosition(self, new_pos: tuple[int, int]) -> None:
        self.pos = new_pos

    def getPosition(self) -> tuple:
        return self.pos

    def setPlayerNum(self, num: int) -> None:
        self.player_num = num

    def getMove(self, grid: Grid) -> tuple:
        """Find the move that results in the most available cells around current player."""

        # find all available moves
        available_moves = grid.get_neighbors(self.pos, only_available=True)

        # create a copy of the Grid for each potential move
        states = [grid.clone().move(mv, self.player_num) for mv in available_moves]

        # for each potential move, calculate the number of available moves (AM)
        am_scores = np.array([AM(state, self.player_num) for state in states])

        # find move with best AM score
        # will return the first occurrence in the event of a tie
        new_pos = available_moves[np.argmax(am_scores)]

        return new_pos

    def getTrap(self, grid: Grid) -> tuple:
        """Find a trap position that results in the greatest difference between number of available cells around current player vs opponent."""

        # find opponent
        opponent = grid.find(3 - self.player_num)

        # find all available cells surrounding opponent
        available_neighbors = grid.get_neighbors(opponent, only_available=True)

        if available_neighbors:

            # create a copy of the Grid for each potential move
            states = [grid.clone().trap(cell) for cell in available_neighbors]

            # for each potential trap position, calculate the difference in available moves for player vs opponent
            # TODO: rather than pass in player_um = '3-self.player_num' and finding argmin, could pass self.player_num and find argmax
            is_scores = np.array([IS(state, 3 - self.player_num) for state in states])

            # find trap position with greatest difference
            # will return the first occurrence in the event of a tie
            # TODO: see above
            trap = available_neighbors[np.argmin(is_scores)]

        else:

            # edge case - if there are no available cell around opponent,
            # then MediumAI player will win; therefore choosing a random trap position
            # TODO: change the trap position to be somewhere not in the vicinity of current player
            print("MediumAI")
            (
                input(
                    f"No available cells around player {3 - self.player_num}! Press enter to continue."
                )
                if self.verbose
                else None
            )
            trap = rand_choice(grid.getAvailableCells())

        return trap


def AM(grid: Grid, player_num: int) -> int:
    """
    Get number of available moves (AM) for the specified player, in the Grid argument passed.
    Calls get_neighbors with only_available = True.
    """

    available_moves = grid.get_neighbors(grid.find(player_num), only_available=True)

    return len(available_moves)


def IS(grid: Grid, player_num: int) -> int:
    """
    Get the difference between number of available moves for specified player vs opponent, in the Grid argument passed.
    Improved Score (IS) method, a basic heuristics calculation.
    """

    # find all available moves by current player
    player_moves = grid.get_neighbors(grid.find(player_num), only_available=True)

    # find all available moves by opponent
    opp_moves = grid.get_neighbors(grid.find(3 - player_num), only_available=True)

    return len(player_moves) - len(opp_moves)
