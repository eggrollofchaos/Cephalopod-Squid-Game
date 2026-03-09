"""
Game Class module.
2-player turn-based game implenting Adversarial AI search using Expectiminimax and various heuristics.
Can have Player (AI) vs Computer Opponent (AI), or Player (AI) vs Human Opponent.
Originally created by @tomcohen13 as a final coding project called Squid Game,
  for COMS W4701 Artificial Intelligence graduate level course at Columbia University, 2021.
  See: https://github.com/tomcohen13/AI-Squid-Game/
Early contributions by teammates @mhr and @gongchen161.
Enhancements by WAX.
"""

from os import system
from platform import system as os_type
from sys import argv
import time

import numpy as np
from termcolor import cprint

from Grid import Grid
from Displayer import Displayer
from HumanOpp import HumanOpp
from PlayerAI import PlayerAI
from PlayerAIOppV1 import PlayerAIOppV1
from PlayerAIOppV2 import PlayerAIOppV2
from PlayerAIOppV3 import PlayerAIOppV3
from test_players.RandomAI import RandomAI
from test_players.EasyAI import EasyAI
from test_players.MediumAI import MediumAI
from test_players.MinimaxAI import MinimaxAI
from test_players.HardAI import HardAI
from Utils import *

PLAYER_TURN, COMPUTER_TURN = (
    1,
    2,
)  # convention: set Player to Player 1, AI Opponent to Player 2
SIZE = 7  # default dimension of square grid
is_unix = os_type()  # for Unix-based systems, can show colors and emojis

# Time Limit Before Losing
timeLimit = 5
allowance = 0.05
clear = lambda: system("clear") if is_unix else system("cls")  # clear screen


class Game:
    """
    Command Line Switches
    ----------
    -oa : set Opponent AI level of {-2,-1,0,1,9,10,11,12,13}; default = 0 (Medium); level = 9 indicates Human Opponent
          -2 : Random AI                                -- for debugging
          -1 : Easy AI
           0 : Medium AI
           1 : Basic Expectiminimax AI
           9 : Human Opponent                           -- NOT AI
          10 : Hard Expectiminimax AI w/ Heuristics
          11 : CustomV1 AI using Expectiminimax
          12 : CustomV2 AI using Expectiminimax
          13 : CustomV3 AI using Expectiminimax
    -od : set Opponent AI search depth limit; min 1, default = 2, only applicable if Opponent AI level in {1,10,11,12,13}

    Debug Note
    ----------
    Set Player AI depth limit to -2 to have player make moves/traps using Random AI.
    Set Opponent AI level to -2 to have opponent make moves/traps using Random AI.

    """

    def __init__(
        self,
        playerAI=None,
        computerAI=None,
        grid_size=SIZE,
        displayer=None,
        test_mode=False,
        verbose=0,
    ):
        """
        Description
        ----------
        Construct new game given two players, board size and displayer.

        Parameters
        ----------
        playerAI    - Human player AI, of type PlayerAI. default = None
                    - This is the player that the adversarial AI will optimize.
                    - In the code, this player will be Player 1, and plays when 'turn' = 1

        computerAI  - Computer or Human Opponent. default = None
                    - If Computer opponent, there is a choice of Easy, Medium, CustomV1, CustomV2, or CustomV3.
                    - There is also a RandomAI class, but it is only used for testing.

        grid_size   - Dimension of square grid.

        displayer   - Class for handling drawing of the game board aka grid.

        test_mode   - Handles whether to pause 5 seconds between moves.

        verbose     - Verbosity level of output in [0,1,2,3].

        """
        self.grid = Grid(grid_size)
        self.playerAI = playerAI or RandomAI(verbose)
        self.computerAI = computerAI or RandomAI(verbose)
        self.dim = grid_size
        self.over = False
        self.displayer = displayer
        self.test_mode = test_mode  # don't wait 5 seconds between moves
        self.verbose = verbose

        # exit('Testing')

    def initialize_game(self) -> None:
        """Initialization of game variables"""

        p1_index, p2_index = (0, self.dim // 2), (self.dim - 1, self.dim // 2)

        self.grid.setCellValue(p1_index, 1)
        self.playerAI.setPosition(p1_index)
        self.playerAI.setPlayerNum(1)

        self.grid.setCellValue(p2_index, 2)
        self.computerAI.setPosition(p2_index)
        self.computerAI.setPlayerNum(2)

    def is_over(self, turn) -> int:
        """Check if game is over, i.e., Player or Opponent has no moves to make"""

        # check if Player has won
        # find available neighbors of player 1
        opponent_neighbors = self.grid.get_neighbors(
            self.computerAI.getPosition(), only_available=True
        )
        # if none - win
        if len(opponent_neighbors) == 0:
            self.over = True
            return 1

        # check if Opponent has won
        player_neighbors = self.grid.get_neighbors(
            self.playerAI.getPosition(), only_available=True
        )

        if len(player_neighbors) == 0:
            self.over = True
            return 2

        # catch-all
        elif self.over:
            return turn

        else:
            return 0

    def is_valid_move(self, grid: Grid, player, move: tuple) -> bool:
        """Validate move - cell has to be available and immediate neighbor"""

        if grid.getCellValue(move) == 0 and move in grid.get_neighbors(
            player.getPosition()
        ):
            return True

        return False

    def is_valid_trap(self, grid: Grid, trap: tuple) -> bool:
        """Validate trap - cell can't be a player"""
        # TODO - maybe remove this limitation?

        if grid.getCellValue(trap) > 0:
            return False

        return True

    def throw(self, player, grid: Grid, intended_position: tuple) -> tuple:
        """
        Description
        ----------
        Function returns the coordinates in which the trap lands, given an intended location.
        This implements a random chance factor to determine if trap lands correctly or in a neighboring cell.

        Parameters
        ----------

        player : the current player throwing the trap

        grid : current game Grid

        intended position : the (x,y) coordinates to which the player intends to throw the trap to.

        Returns
        -------
        Position (x_0,y_0) in which the trap landed : tuple

        """

        # find neighboring cells
        neighbors = grid.get_neighbors(intended_position)
        # print(f'\nneighbors list =:\n{neighbors}')                        # for debugging

        # include only empty cells or existing traps
        neighbors = [
            neighbor for neighbor in neighbors if grid.getCellValue(neighbor) <= 0
        ]
        # print(f'\nneighbors list after <= 0 list comp =:\n{neighbors}')   # for debugging

        n = len(neighbors)
        probs = np.ones(1 + n)  # intended position + neighbors

        # compute probability of success, p, based on factor of 0.05 and the distance to intended trap position
        # distance is computed as Manhattan Distance, which is the sum of the absolute distances in both X and Y directions
        p = 1 - 0.05 * (manhattan_distance(player.getPosition(), intended_position) - 1)

        probs[0] = p  # intended position gets base probability
        probs[1:] = np.ones(len(neighbors)) * (
            (1 - p) / n
        )  # remaining probability is split among neighbors

        # insert intended position coordinates to neighbors list
        neighbors.insert(0, intended_position)

        # determine where trap lands
        result = np.random.choice(np.arange(n + 1), p=probs)

        return neighbors[result]

    def updateAlarm(self, currTime) -> None:
        """
        A method to handle time limit calculation.
        Only runs if self.test_mode is True, i.e. if -t flag is not passed at init.
        """
        if currTime - self.prevTime > timeLimit + allowance:
            self.over = True
            print("Went over time. The Doll has shot the current player! Game over.")
        else:
            # while time.process_time() - self.prevTime < timeLimit + allowance:
            # pass

            self.prevTime = time.process_time()

    def play(self) -> tuple:
        """
        Handles all game logistics.
        Do not modify.
        """

        total_player_moves = 0
        total_player_traps = 0

        print("")
        # cprint('\n', on_color='on_yellow')
        if is_unix:
            cprint(" " * 49, on_color="on_yellow")
            cprint(
                "                  AI SQUID GAME                  ",
                color="blue",
                on_color="on_yellow",
                attrs=["bold"],
            )
            cprint(" " * 49, on_color="on_yellow")
        else:
            print("")
            print("\n\nAI SQUID GAME\n")
            print("")

        # cprint('  ', on_color='on_yellow')
        # print('')

        self.initialize_game()  # set initial game variables

        self.displayer.display(self.grid)  # display starting game board

        turn = PLAYER_TURN  # Player goes first

        # main game loop
        while not self.over:
            self.prevTime = time.process_time()
            start = self.prevTime
            grid_copy = (
                self.grid.clone()
            )  # convention, always make a copy of current game board

            move = None

            # TODO: Refactor below so that the code is not duplicative for Player vs Opponent

            if turn == 1:  # Player

                total_player_moves += 1

                (
                    input("<Press enter to begin!>")
                    if total_player_moves == 1 and not self.test_mode
                    else None
                )  # waiting for input to continue for debugging

                (
                    cprint(f"Player's Turn {total_player_moves}: ", color="green")
                    if is_unix
                    else print(f"Player's Turn {total_player_moves}:")
                )
                # find best move; should return two coordinates: new player position and position that the trap landed on
                move = self.playerAI.getMove(grid_copy)

                # if move is valid, perform it
                if self.is_valid_move(self.grid, self.playerAI, move):
                    self.grid.move(move, turn)
                    self.playerAI.setPosition(move)
                    print(f"Moving to {move}.")
                else:  # check this condition, does it ever happen?
                    self.over = True
                    print(f"Tried to move to : {move}")
                    print("invalid Player AI move!")

                total_player_traps += 1

                intended_trap = self.playerAI.getTrap(self.grid.clone())
                # input()
                if self.is_valid_trap(self.grid, intended_trap):
                    print(f"Throwing a trap to: {intended_trap}... ", end="")
                    trap = self.throw(self.playerAI, self.grid, intended_trap)
                    if trap == intended_trap:
                        (
                            cprint(
                                f"Trap landed successfully in {trap}",
                                end="",
                                color="green",
                            )
                            if is_unix
                            else print(f"Trap landed successfully in {trap}", end="")
                        )
                    else:
                        (
                            cprint(
                                f"Trap missed, landed in {trap}", end="", color="red"
                            )
                            if is_unix
                            else print(f"Trap missed, landed in {trap}", end="")
                        )

                    if self.grid.getCellValue(trap) == -1:
                        print(", which already had a trap, no effect.", end="")
                    print(".")
                    self.grid.trap(trap)

                else:  # check this condition, does it ever happen?
                    self.over = True
                    print(f"Tried to put trap in {intended_trap}")
                    print("Invalid trap!")

                end = time.process_time()
                total_time = end - start

                if self.verbose:
                    (
                        cprint(
                            f"Player's move + throw took {total_time:.3f} seconds.",
                            color="green",
                        )
                        if is_unix
                        else print(
                            f"Player's move + throw took {total_time:.3f} seconds."
                        )
                    )

                if not self.test_mode:  # for testing, can allow exceeding time
                    if total_time >= timeLimit + allowance:  # i.e. 5.05
                        # raise Exception('Exceeded time limit.')
                        (
                            cprint(
                                "\nExceeded 5 second time limit!\n",
                                on_color="on_yellow",
                            )
                            if is_unix
                            else print("\nExceeded 5 second time limit!\n")
                        )
                        raise RuntimeError(
                            "GAME OVER"
                        )  # TODO: why RunTimeError vs Exception?
                        # self.over = True

            else:  # Opponent

                (
                    cprint(f"Opponent's Turn {total_player_moves}: ", color="magenta")
                    if is_unix
                    else print(f"Opponent's Turn {total_player_moves}: ")
                )

                # make move
                move = self.computerAI.getMove(grid_copy)

                # check if move is valid; perform if it is.
                if self.is_valid_move(self.grid, self.computerAI, move):
                    self.grid.move(move, turn)
                    self.computerAI.setPosition(move)
                    print(f"Moving to {move}.")

                else:
                    self.over = True
                    print("Invalid Opponent move")

                intended_trap = self.computerAI.getTrap(self.grid.clone())

                if self.is_valid_trap(self.grid, intended_trap):
                    print(f"Throwing a trap to: {intended_trap}...", end="")
                    trap = self.throw(self.computerAI, self.grid, intended_trap)
                    self.grid.trap(trap)
                    print(f"Trap landed in {trap}")
                else:
                    self.over = True
                    print(f"Tried to put trap in {intended_trap}")
                    print("Invalid trap!")

                end = time.process_time()
                total_time = end - start

                if self.verbose:
                    (
                        cprint(
                            f"Opponent's move + throw took {total_time:.3f} seconds.",
                            color="magenta",
                        )
                        if is_unix
                        else print(
                            f"Opponent's move + throw took {total_time:.3f} seconds."
                        )
                    )

                # clear()

            if self.is_over(turn):
                # cprint('is_over(turn) evaluated to True. Press Enter.', color='yellow')
                # input()                         # waiting for input to continue for debugging
                self.over = True

            if not self.test_mode:  # wait time in between
                self.updateAlarm(time.process_time())
            turn = 3 - turn

            self.displayer.display(self.grid)

        return self.is_over(turn), total_player_moves, total_player_traps


def main():
    depth_limit = 4  # default depth limit for Player AI
    test_mode = False
    verbose = 0
    heur = False
    # heur = 'graphcut'
    opp_ai_default = True
    opp_ai_int = 0
    opp_ai_level = "MediumAI()"  # for eval
    opp_depth_limit = 2  # only applicable for AI level higher than Easy/Medium AI
    playerAIdebug = False  # for debugging with random AI
    computerAIdebug = False  # for debugging with random AI

    if len(argv) > 1:
        if "-t" in argv:  # enable to skip the 5 seconds wait in between moves
            test_mode = True
        if "-v" in argv:  # for more game info
            verbose = 1
        if "-vv" in argv:  # for extra information
            verbose = 2
        if "-vvv" in argv:  # for detailed trace logging
            verbose = 3
        if "-h" in argv:  # graphcut heuristics
            heur = "graphcut"
        if "-h2" in argv:  # geodesics heuristics (not implemented)
            heur = "geodesics"
        if "-d" in argv:  # Player search depth limit, set to -2 for RandomAI
            try:
                dl_flag_index = argv.index("-d")
                depth_limit = int(argv[dl_flag_index + 1])
                if depth_limit == -2:  # debugging with RandomAI()
                    playerAIdebug = True
            except:
                pass
        if (
            "-oa" in argv
        ):  # Opponent AI level; set to 9 for Human Opponent, set to -2 for RandomAI
            try:
                od_flag_index = argv.index("-oa")
                opp_ai_int = int(argv[od_flag_index + 1])
                opp_ai_default = False
                if opp_ai_int == -2:  # debugging with RandomAI()
                    computerAIdebug = True
            except:
                pass
        if (
            opp_ai_int > 0 and "-od" in argv
        ):  # applicable if higher than Easy/Medium AI and not Human Opponent
            try:
                opp_dl_flag_index = argv.index("-od")
                opp_depth_limit = int(argv[opp_dl_flag_index + 1])
            except:
                pass

    # clear()

    # DEBUG ONLY
    # playerAIdebug = True                                    # use random moves / throws via RandomAI.py, for testing only
    # computerAIdebug = True                                  # use random moves / throws via RandomAI.py, for testing only

    if depth_limit != -2:  # skip if debugging with RandomAI()
        depth_limit = max(
            depth_limit, 1
        )  # Expectiminimax needs at least a search depth of 1, but really 2

    match heur:
        case False:
            heur_str = "standard"
        case "graphcut":
            heur_str = "standard + graph cut advanced"

    # introduce game, initialize Player and display to stdout
    if verbose:
        print("\n\nRunning AI Squid Game with ", end="")
        cprint("verbose", color="blue", end="") if is_unix else print("verbose", end="")
        print(f" level = {verbose}.\n")
        if playerAIdebug:  # for random moves
            # print('\n')
            (
                cprint(
                    "NOTE: Debug mode -- Player is using no AI, all moves/throws are random.",
                    color="green",
                )
                if is_unix
                else print(
                    "NOTE: Debug mode -- Player is using no AI, all moves/throws are random."
                )
            )
        else:
            print(f"Player is using Expectiminimax search algo with ", end="")
            (
                cprint(f"depth limit of {depth_limit}", color="green", end="")
                if is_unix
                else print(f"{depth_limit}", end="")
            )
            print(" and ", end="")
            (
                cprint(f"{heur_str} heuristics.", color="cyan")
                if is_unix
                else print(f"and {heur_str} heuristics.")
            )
    if playerAIdebug:
        playerAI = RandomAI(verbose)
    else:
        playerAI = PlayerAI(
            depth_limit, heur, verbose, SIZE
        )  # PlayerAI is the primary Expectiminimax adversarial search AI for this game

    # logic for handling what AI level Opponent will use, or if it will be a Human player
    opp_pre_string = ""
    opp_string = ""
    if computerAIdebug:
        opp_ai_level = "RandomAI(verbose)"  # for eval
        opp_pre_string = "NOTE: Debug mode -- "
        opp_string = "no AI, all moves/throws are random.\n"
        # print('Here')
    elif opp_ai_default:
        opp_ai_level = "MediumAI(verbose)"  # for eval
        opp_string = "Medium AI (default) as none was specified.\n"
    else:
        match opp_ai_int:
            case -1:
                opp_ai_level = "EasyAI(verbose)"  # for eval
                opp_string = "Easy AI.\n"
            case 0:
                opp_ai_level = "MediumAI(verbose)"  # for eval
                opp_string = "Medium AI.\n"
            case 1:
                opp_ai_level = "MinimaxAI(opp_depth_limit, verbose, SIZE)"  # for eval
                opp_string = "Minimax AI (with Expectiminimax search algo)"
            case 10:
                opp_ai_level = "HardAI(opp_depth_limit, verbose, SIZE)"  # for eval
                opp_string = "Hard AI (with Expectiminimax seach algo + heuristics)"
            case 11:
                opp_ai_level = (
                    "PlayerAIOppV1(opp_depth_limit, verbose, SIZE)"  # for eval
                )
                opp_string = "Custom AI version 1"
            case 12:
                opp_ai_level = (
                    "PlayerAIOppV2(opp_depth_limit, verbose, SIZE)"  # for eval
                )
                opp_string = "Custom AI version 2"
            case 13:
                opp_ai_level = (
                    "PlayerAIOppV3(opp_depth_limit, heur, verbose, SIZE)"  # for eval
                )
                opp_string = "Custom AI version 3"
            case 9:
                if is_unix:
                    opp_string = "will be a human player. 👴👵\n"
                else:
                    opp_string = "will be a human player.\n"
            case _:
                opp_ai_level = "MediumAI()"  # for eval
                opp_ai_int = 0  # setting as default
                opp_string = (
                    f"Medium AI (default) as input of '{opp_ai_int}' not defined.\n"
                )

    # output Opponent selection
    if verbose and is_unix:
        cprint(
            f"\n{opp_pre_string}Opponent is using {opp_string}", color="magenta", end=""
        )
        if opp_ai_int > 0 and opp_ai_int != 9:
            cprint(f", with depth limit of {opp_depth_limit}.", color="magenta")
    elif verbose:
        print(f"\n{opp_pre_string}Opponent is using {opp_string}", end="")
        if opp_ai_int > 0 and opp_ai_int != 9:
            print(f", with depth limit of {opp_depth_limit}.")

    # initialize Opponent
    if opp_ai_int == 9:
        computerAI = HumanOpp(verbose=verbose)
    else:
        computerAI = eval(opp_ai_level)

    # depth_limit = 0                                   # for testing

    # initialize Displayer, Game, start gameplay
    displayer = Displayer(grid_size=SIZE)
    game = Game(
        playerAI=playerAI,
        computerAI=computerAI,
        grid_size=SIZE,
        displayer=displayer,
        test_mode=test_mode,
        verbose=verbose,
    )
    result, moves, traps = game.play()

    # game has ended
    exit_code = int(
        str(result) + str(moves)
    )  # combine player number and number of moves in exit_code for capture
    if result == 1:
        (
            cprint(
                "\n\nPlayer 1 wins!\n",
                color="green",
                on_color="on_black",
                attrs=["bold", "blink"],
            )
            if is_unix
            else print("\n\nPlayer 1 wins!\n")
        )
    elif result == 2:
        (
            cprint(
                "\n\nPlayer 1 loses!\n",
                color="magenta",
                on_color="on_black",
                attrs=["bold", "blink"],
            )
            if is_unix
            else print("\n\nPlayer 1 loses!\n")
        )

    # print summary
    if verbose:
        print("\n\nSUMMARY\n")
        if playerAIdebug:
            print("Player used no AI, with all random moves and throws...")
        else:
            print(f"Player used Expectiminimax with ", end="")
            (
                cprint(f"depth limit of {depth_limit}", color="green", end="")
                if is_unix
                else print(f"{depth_limit}", end="")
            )
            print(" and ", end="")
            (
                cprint(f"{heur_str} heuristics...", color="cyan")
                if is_unix
                else print(f"and {heur_str} heuristics...")
            )

        if result == 1:
            (
                cprint("and defeated...", color="green")
                if is_unix
                else print("and defeated...")
            )
        elif result == 2:
            (
                cprint("and lost to...", color="magenta")
                if is_unix
                else print("and lost to...")
            )

        if computerAIdebug:
            print(
                "Computer opponent who used no AI, with all random moves and throws.\n"
            )
        else:
            if is_unix:
                cprint(
                    f"Computer opponent, who used {opp_string}", color="magenta", end=""
                )
                if opp_ai_int > 0:
                    cprint(
                        f", with depth limit of {opp_depth_limit}.\n", color="magenta"
                    )
            else:
                print(f"Computer opponent, who used {opp_string}", end="")
                if opp_ai_int > 0:
                    print(f", with depth limit of {opp_depth_limit}.\n")

        print(f"Total rounds: {moves}\n\n")

    # TODO
    # Add other statistics:
    # total game time, best seen utility, average move length, throw accuracy

    exit(exit_code)


if __name__ == "__main__":
    main()
