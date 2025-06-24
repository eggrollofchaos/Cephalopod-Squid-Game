"""
Basic MinimaxAI Class module.
Implements Expectiminimax search algo.
Commented by WAX.
"""
import os 
import sys
import time

import numpy as np

sys.path.append(os.getcwd())        # setting path to parent directory
from BaseAI import BaseAI
from Grid import Grid

MAX_DEPTH = 3
MOVE_TIME_LIMIT = 0.49
TRAP_TIME_LIMIT = 0.49

class MinimaxAI(BaseAI):
    """
    Minimax Computer AI player, implements Expectiminimax.
    Default depth limit is 3.
    TODO: write docstring
    
    """

    def __init__(self, depth_limit: int = MAX_DEPTH, position: tuple[int, int] = None, verbose: int = 0) -> None:
        super().__init__()
        print('Running basic MinimaxAI()...') if verbose else None
        self.depth_limit = depth_limit
        self.pos = position
        self.player_num = None

    def getPosition(self) -> tuple[int, int]:
        return self.pos
    
    def setPosition(self, new_position: tuple[int, int] = None) -> None:
        self.pos = new_position
    
    def getPlayerNum(self):
        return self.player_num
        
    def setPlayerNum(self, num: int):
        self.player_num = num

    def getMove(self, grid: Grid):
        move, util = self._best_move(grid)

        return move

    def _best_move(self, grid: Grid):

        start = time.process_time()

        # Funny edge case: check if player has won by trapping Opponent with previous move. Throwing randomly.
        if len(grid.get_neighbors(grid.find(3 - self.player_num), only_available=True)) == 0:
            return grid.getAvailableCells()[0], 10000
        
        return self.maximize_move(grid, alpha = -np.inf, beta = np.inf, depth = 0, start_time = start)

    def maximize_move(self, grid: Grid, alpha: float, beta: float, depth: int, start_time):
        """ 
        Description
        -----------
        The Max node of the Minimax search tree of Move.
        The function maximizes utility over a *static* opponent throwing traps strategically.
        Uses Alpha-Beta Pruning to skip unpromising branches of the tree.

        Parameters
        ----------
        grid (Grid) : a state of the game described by a Grid object.

        alpha : Maximizer's lower bound on utility

        beta : Minimizer's upper bound on utility

        depth : current depth of the state in the search tree

        start_time : a timestamp of when the turn has started to make sure move does not exceed it.

        Returns
        -------
        The action corresponding to the maximum utility move, given an intelligent opponent.

        """
        if self.terminal_test(grid, time = time.process_time() - start_time, depth = depth, mode = 'move'):
            return None, self.utility(grid)
        
        maxMove, maxUtility = None, -np.inf
        
        available_moves = grid.get_neighbors(grid.find(self.player_num), only_available = True)

        states = [grid.clone().move(mv, self.player_num) for mv in available_moves]
       
        ## add sorting of moves and states?

        for (move, state) in zip(available_moves, states):

            _, utility = self.minimize_move(state, alpha, beta, depth + 1, start_time)

            if utility > maxUtility:
                maxMove, maxUtility = move, utility
            
            # if Max lower bound crosses Min's upper bound - break
            if maxUtility >= beta:
                break

            # update lower bound
            alpha = max(alpha, maxUtility)

        return maxMove, maxUtility

    def minimize_move(self, grid : Grid, alpha, beta, depth, start_time):
        """ 
        Description
        -----------
        The Min node of the Minimax search tree of Move.
        
        The function minimizes utility over the player's future moves by finding ideal position to place a trap.
        
        Uses Alpha-Beta Pruning to skip unpromising branches of the tree.

        Parameters
        ----------
        grid (Grid) : a state of the game described by a Grid object.

        alpha : Maximizer's lower bound on utility

        beta : Minimizer's upper bound on utility

        depth : current depth of the state in the search tree

        start_time : a timestamp of when the turn has started to make sure move does not exceed it.

        Returns
        -------
        The action corresponding to the maximum utility move, given an intelligent opponent.
        
        """
        if self.terminal_test(grid, time = time.process_time() - start_time, depth = depth, mode = 'move'):
            return None, self.utility(grid)

        minChild, minUtility = None, np.inf
        # 
        actions = grid.get_neighbors(grid.find(self.player_num), only_available=True)
        
        states = [grid.clone().trap(pos = a) for a in actions]

        for (action, state) in zip(actions, states):

            _, utility = self.maximize_move(state, alpha, beta, depth + 1, start_time)

            if utility < minUtility:
                minChild, minUtility = action, utility
            
            if minUtility <= alpha:
                break

            beta = min(beta, minUtility)

        return minChild, minUtility


    def terminal_test(self, state : Grid, time, depth, mode = 'move'):
        
        lose = not state.get_neighbors(state.find(self.player_num), only_available=True)
        
        win  = not state.get_neighbors(state.find(3 - self.player_num), only_available=True)

        if mode == 'move' :
            # return lose or win or time >= MOVE_TIME_LIMIT or depth >= MAX_DEPTH
            return lose or win or time >= MOVE_TIME_LIMIT or depth >= self.depth_limit
        else :
            # return lose or win or time >= TRAP_TIME_LIMIT or depth >= MAX_DEPTH
            return lose or win or time >= TRAP_TIME_LIMIT or depth >= self.depth_limit

    def getTrap(self, grid : Grid):
        trap, _ = self._best_trap(grid)
        return trap

    def _best_trap(self, grid: Grid):
        start = time.process_time()
        # Funny edge case: check if player has won by trapping Opponent with previous move. Throwing randomly.
        if len(grid.get_neighbors(grid.find(3 - self.player_num), only_available=True)) == 0:
            return grid.getAvailableCells()[0], 100
        return self.maximize_trap(grid, -np.inf, np.inf, depth = 0, start_time = start)

    def maximize_trap(self, grid : Grid, alpha, beta, depth, start_time):
        """ 
        Description
        -----------
        The Max node of the Minimax search tree of Trap.
        The function maximizes utility over Opponent's Move actions.
        Uses Alpha-Beta Pruning to skip unpromising branches of the tree.

        Parameters
        ----------
        grid (Grid) : a state of the game described by a Grid object.

        alpha : Maximizer's lower bound on utility

        beta : Minimizer's upper bound on utility

        depth : current depth of the state in the search tree

        start_time : a timestamp of when the turn has started to make sure move does not exceed it.

        Returns
        -------
        The action corresponding to the maximum utility Trap, given an intelligent opponent.

        """
        if self.terminal_test(grid, time.process_time() - start_time, depth, mode = 'trap'):
            return None, self.utility(grid)
        
        maxUtility = -np.inf
        
        # only consider immediate neighbors of Opponent
        positions = grid.get_neighbors(grid.find(3 - self.player_num), only_available = True)
        
        # create states corresponding to each action
        states = [grid.clone().trap(position) for position in positions]

        for (action, state) in zip(positions, states):

            _, utility = self.minimize_trap(state, alpha, beta, depth + 1, start_time)

            if utility > maxUtility:
                maxTrap, maxUtility = action, utility

            if utility >= beta:
                break
            
            alpha = max(alpha, utility)

        return maxTrap, maxUtility
        
    def minimize_trap(self, grid : Grid, alpha, beta, depth, start_time):
        """ 
        Description
        -----------
        The Min node of the Minimax search tree of Trap.
        Finds best Move action by opponent, in response to a trap thrown by player.
        Uses Alpha-Beta Pruning to skip unpromising branches of the tree.

        Parameters
        ----------
        grid (Grid) : a state of the game described by a Grid object.

        alpha : Maximizer's lower bound on utility

        beta : Minimizer's upper bound on utility

        depth : current depth of the state in the search tree

        start_time : a timestamp of when the turn has started to make sure move does not exceed it.

        Returns
        -------
        The action corresponding to the maximum utility move, given an intelligent opponent.

        """
        if self.terminal_test(grid, time.process_time() - start_time, depth, mode = 'trap'):
            return None, self.utility(grid)

        minMove, minUtility = None, np.inf
        
        # find all possible immediate moves by Opponent
        available_moves = grid.get_neighbors(grid.find(3 - self.player_num), only_available=True)

        # create states corresponding to those possible moves
        states = [grid.clone().move(mv, player = 3 - self.player_num) for mv in available_moves]

        for (move, state) in zip(available_moves, states):

            _, utility = self.maximize_trap(state, alpha, beta, depth + 1, start_time)
            
            if utility < minUtility:
                minMove, minUtility = move, utility

            if utility <= alpha: 
                break

            beta = min(beta, minUtility)

        return minMove, minUtility


    def utility(self, state: Grid) -> float:
        """Set a numeric utility for win and loss states."""

        # if win
        if not state.get_neighbors(state.find(3 - self.player_num), only_available=True):
            return 100
        # if lose
        if not state.get_neighbors(state.find(self.player_num), only_available=True):
            return -100
        
        return IS(state, player_num = self.player_num)
        

def IS(grid: Grid, player_num: int) -> int:

    # find all available moves by Player
    player_moves    = grid.get_neighbors(grid.find(player_num), only_available = True)
    
    # find all available moves by Opponent
    opp_moves       = grid.get_neighbors(grid.find(3 - player_num), only_available = True)
    
    return len(player_moves) - len(opp_moves)      
