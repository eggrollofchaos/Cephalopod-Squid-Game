import time
import numpy as np
from sys import argv
from Grid import Grid
from ComputerAI import ComputerAI
from Displayer import Displayer
from HumanOpp import HumanOpp
from platform import system as os_type
from PlayerAI import PlayerAI
from PlayerAIOppV1 import PlayerAIOppV1
from PlayerAIOppV2 import PlayerAIOppV2
from PlayerAIOppV3 import PlayerAIOppV3
from test_players.EasyAI import EasyAI
from test_players.MediumAI import MediumAI
from Utils import *
from termcolor import cprint
from os import system

PLAYER_TURN, COMPUTER_TURN = 1,2
is_unix = os_type()

# Time Limit Before Losing
timeLimit = 5
allowance = 0.05
clear = lambda: system('clear') if is_unix else system('cls')               # clear screen


class Game():
    def __init__(self, playerAI = None, computerAI = None, N = 7, displayer = None, test_mode = False, verbose = 0):
        '''
        Description
        ----------
        Construct new game given two players, board size and displayer.

        Parameters
        ----------
        playerAI   - Human player AI, of type PlayerAI. default = None 

        computerAI - Human or Computer Opponent. default = None
        
        N  - dimension of grid.

        '''
        self.grid       = Grid(N)
        self.playerAI   = playerAI or ComputerAI() 
        self.computerAI = computerAI or ComputerAI() 
        self.dim        = N
        self.over       = False
        self.displayer  = displayer
        self.test_mode  = test_mode   # don't wait 5 seconds between moves
        self.verbose    = verbose

        if self.verbose:
            if playerAI == None:
                print("Note: Player is using no AI, all moves/throws are random.")
            if computerAI == None:
                print("Note: Opponent is using no AI, all moves/throws are random.")

        # exit('Testing')
        
    def initialize_game(self):

        p1_index, p2_index = (0, self.dim // 2), (self.dim - 1, self.dim // 2)
        
        self.grid.setCellValue(p1_index, 1)
        self.playerAI.setPosition(p1_index)
        self.playerAI.setPlayerNum(1)

        self.grid.setCellValue(p2_index, 2)
        self.computerAI.setPosition(p2_index)
        self.computerAI.setPlayerNum(2)
        
    def is_over(self, turn):
        """Check if game is over, i.e., Player or Opponent has no moves to make"""
        # check if Player has won
        # find available neighbors of player 1
        opponent_neighbors = self.grid.get_neighbors(self.computerAI.getPosition(), only_available=True)
        # if none - win
        if len(opponent_neighbors) == 0:
            self.over = True
            return 1

        # check if Opponent has won
        player_neighbors = self.grid.get_neighbors(self.playerAI.getPosition(), only_available=True)

        if len(player_neighbors) == 0:
            self.over = True
            return 2
        
        elif self.over:
            return turn

        else: 
            return 0

    def is_valid_move(self, grid : Grid, player, move : tuple):

        '''Validate move - cell has to be available and immediate neighbor'''
        
        if grid.getCellValue(move) == 0 and move in grid.get_neighbors(player.getPosition()):
            return True
        
        return False

    def is_valid_trap(self, grid : Grid, trap : tuple):
        '''Validate trap - cell can't be a player'''

        if grid.getCellValue(trap) > 0:
            return False

        return True

    def throw(self, player, grid : Grid, intended_position : tuple) -> tuple:
        '''
        Description
        ----------
        Function returns the coordinates in which the trap lands, given an intended location.

        Parameters
        ----------

        player : the player throwing the trap

        grid : current game Grid

        intended position : the (x,y) coordinates to which the player intends to throw the trap to.

        Returns
        -------
        Position (x_0,y_0) in which the trap landed : tuple

        '''
 
        # find neighboring cells
        neighbors = grid.get_neighbors(intended_position)

        neighbors = [neighbor for neighbor in neighbors if grid.getCellValue(neighbor) <= 0]
        n = len(neighbors)
        
        probs = np.ones(1 + n)
        
        # compute probability of success, p
        p = 1 - 0.05*(manhattan_distance(player.getPosition(), intended_position) - 1)

        probs[0] = p

        probs[1:] = np.ones(len(neighbors)) * ((1-p)/n)

        # add desired coordinates to neighbors
        neighbors.insert(0, intended_position)
        
        # return 
        result = np.random.choice(np.arange(n + 1), p = probs)
        
        return neighbors[result]

    def updateAlarm(self, currTime):
        if currTime - self.prevTime > timeLimit + allowance:
            self.over = True
            print("Went over time. Doll Shot!")
        else:
            while time.process_time() - self.prevTime < timeLimit + allowance:
                pass

            self.prevTime = time.process_time()

    def play(self):
        """ DO NOT MODIFY """
        total_player_moves = 0
        total_player_traps = 0
                
        print("")
        # cprint('\n', on_color = 'on_yellow')
        cprint(" " * 49, on_color = 'on_yellow') if is_unix else print("")
        cprint("                  AI SQUID GAME                  ", color='blue', on_color = "on_yellow") if is_unix else print("\n\nAI SQUID GAME\n")
        cprint(" " * 49, on_color = 'on_yellow') if is_unix else print("")
        # cprint('  ', on_color = 'on_yellow')
        # print("")
        
        self.initialize_game()

        self.displayer.display(self.grid)

        turn = PLAYER_TURN
        
        while not self.over:
            self.prevTime = time.process_time()
            start = self.prevTime
            grid_copy = self.grid.clone()

            move = None
            
            if turn == 1:

                total_player_moves += 1

                cprint(f"Player's Turn {total_player_moves}: ", 'green') if is_unix else print(f"Player's Turn {total_player_moves}:")
                # find best move; should return two coordinates - new position and bombed tile.
                move = self.playerAI.getMove(grid_copy)
                # input()

                # if move is valid, perform it
                if self.is_valid_move(self.grid, self.playerAI, move):
                    self.grid.move(move, turn)
                    self.playerAI.setPosition(move)
                    print(f"Moving to {move}")
                else:
                    self.over = True
                    print(f"Tried to move to : {move}")
                    print("invalid Player AI move!")
                
                total_player_traps += 1

                intended_trap = self.playerAI.getTrap(self.grid.clone())
                # input()
                if self.is_valid_trap(self.grid, intended_trap):
                    print(f"Throwing a trap to: {intended_trap}... ", end='')
                    trap = self.throw(self.playerAI, self.grid, intended_trap)
                    print(f"Trap landed in {trap}", end='')
                    if self.grid.getCellValue(trap) == -1:
                        print(', which already had a trap, no effect.')
                    print('.')
                    self.grid.trap(trap)

                else: 
                    self.over = True
                    print(f"Tried to put trap in {intended_trap}")
                    print("Invalid trap!")

                end = time.process_time()
                total_time = end-start

                if self.verbose:
                    cprint(f'Player\'s move + throw took {total_time:.3f} seconds.', 'green') if is_unix else print(f'Player\'s move + throw took {total_time:.3f} seconds.')
                
                if not self.test_mode:                                  # for testing, can allow exceeding time
                    if total_time >= timeLimit + allowance:         # i.e. 5.05
                        # raise Exception('Exceeded time limit.')
                        cprint('\nExceeded 5 second time limit!', on_color='on_yellow') if is_unix else print('\nExceeded 5 second time limit!')
                        # print()
                        # raise RuntimeError('Exceeded 5 second time limit.')
                        raise RuntimeError('GAME OVER')
                        # self.over = True

            else:

                cprint(f"Opponent's Turn {total_player_moves}: ", 'magenta') if is_unix else print(f"Opponent's Turn {total_player_moves}: ")
                
                # make move
                move = self.computerAI.getMove(grid_copy)

                # check if move is valid; perform if it is.
                if self.is_valid_move(self.grid, self.computerAI, move):
                    self.grid.move(move, turn)
                    self.computerAI.setPosition(move)
                    print(f"Moving to {move}")

                else:
                    self.over = True
                    print("invalid Computer AI Move")

                intended_trap = self.computerAI.getTrap(self.grid.clone())

                if self.is_valid_trap(self.grid, intended_trap):
                    print(f"Throwing a trap to: {intended_trap}...", end='')
                    trap = self.throw(self.computerAI, self.grid, intended_trap)
                    self.grid.trap(trap)
                    print(f"Trap landed in {trap}")
                else: 
                    self.over = True
                    print(f"Tried to put trap in {intended_trap}")
                    print("Invalid trap!")

                end = time.process_time()
                total_time = end-start

                if self.verbose:
                    cprint(f'Opponent\'s move + throw took {total_time:.3f} seconds.', 'magenta') if is_unix else print(f'Opponent\'s move + throw took {total_time:.3f} seconds.')
                
                # clear()

            if self.is_over(turn):
                self.over = True

            if not self.test_mode:              # wait time in between
                self.updateAlarm(time.process_time())
            turn = 3 - turn
            
            
            self.displayer.display(self.grid)

        return self.is_over(turn), total_player_moves, total_player_traps

def main():
    depth_limit = 0
    test_mode = False
    verbose = 0
    heur = False
    # heur = 'graphcut'
    opp_ai_int = 0
    opp_ai_level = 'MediumAI()'
    opp_depth_limit = 2                         # only applicable for AI level higher than Easy/Medium AI
    
    if len(argv)>1:
        if '-t' in argv:                        # enable to skip the 5 seconds wait in between moves
            test_mode = True
        if '-v' in argv:                        # for more game info
            verbose = 1
        if '-vv' in argv:                       # for extra information
            verbose = 2
        if '-vvv' in argv:                      # for extra debugging
            verbose = 3
        if '-h' in argv:
            heur = 'graphcut'
        if '-h2' in argv:
            heur = 'geodesics'
        if '-d' in argv:                        # search depth limit
            try:
                dl_flag_index = argv.index('-d')
                depth_limit = int(argv[dl_flag_index+1])
            except:
                pass
        if '-oa' in argv:                       # opponent AI difficulty
            try:
                od_flag_index = argv.index('-oa')
                opp_ai_int = int(argv[od_flag_index+1])
            except:
                pass
        if opp_ai_int > 0 and opp_ai_int != 0 and '-od' in argv:    # applicable if higher than Easy/Medium AI and not Human Opponent
            try:
                opp_dl_flag_index = argv.index('-od')
                opp_depth_limit = int(argv[opp_dl_flag_index+1])
            except:
                pass
            
    # clear()
            
    #### EDIT HERE ####
    
    playerAI = PlayerAI(depth_limit, heur, verbose)    # change this to PlayerAI() to test your player!
    match heur:
        case False:
            heur_str = "standard"
        case 'graphcut':
            heur_str = "standard + graph cut advanced"
    
    if verbose:
        print('Running AI Squid Game with ', end='')
        cprint('verbose', 'blue', end='') if is_unix else print('verbose', end='')
        print(f' level = {verbose}.\n')
        print(f"Player is using Expectiminimax with depth limit of {depth_limit} and {heur_str} heuristics.")
    
    
    # playerAI = None                                    # will use random moves / throws in ComputerAI.py, for testing only
    # computerAI = None                                  # will use random moves / throws in ComputerAI.py, for testing only

    match opp_ai_int:
        case -1:
            opp_ai_level = 'EasyAI()'
            print("Opponent is using Easy AI.") if verbose else None
        case 0:
            opp_ai_level = 'MediumAI()'
            print("Opponent is using Medium AI.") if verbose else None
        case 1:
            opp_ai_level = 'PlayerAIOppV1(opp_depth_limit)'
            print("Opponent is using custom AI version 1.") if verbose else None
        case 2:
            opp_ai_level = 'PlayerAIOppV2(opp_depth_limit, verbose)'
            print("Opponent is using custom AI version 2.") if verbose else None
        case 3:
            opp_ai_level = 'PlayerAIOppV3(opp_depth_limit, heur, verbose)'
            print("Opponent is using custom AI version 3.") if verbose else None
        case 9:
            # print("Opponent will be a human player.👴👵") if verbose else None
            print("Opponent will be a human player.") if verbose else None
        case _:
            opp_ai_level = 'MediumAI()'
            print("Opponent is defaulting to Medium AI.") if verbose else None
            
    if opp_ai_int == 9:
        computerAI = HumanOpp(verbose = verbose)
    else:
        computerAI = eval(opp_ai_level)
        
    # depth_limit = 0, for testing
    #### EDIT HERE ####

    
    displayer = Displayer(N = 7)
    game = Game(playerAI = playerAI, computerAI = computerAI, N = 7, displayer=displayer, test_mode=test_mode)
    result, moves, traps = game.play()

    exit_code = int(str(result) + str(moves))
    if result == 1: 
        print("Player 1 wins!")
        print(f"Total turns: {moves}")
    elif result == 2:
        print("Player 1 loses!")
        print(f"Total turns: {moves}")
    exit(exit_code)

if __name__ == "__main__":
    main()
