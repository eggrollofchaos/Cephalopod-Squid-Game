"""
PlayerAIOppV2 Class module.
Primarily authored by WAX.
Early contributions by @mhr and @gongchen161.
"""
import os
import queue as Q
import random
import sys
import time

import numpy as np

from BaseAI import BaseAI
from Grid import Grid
from Utils import manhattan_distance, grid_distance
# from scipy.sparse import csr_matrix
# from scipy.sparse.csgraph import connected_components

DEFAULT_DEPTH_LIMIT = 3
SIZE = 7                            # default dimension of square grid, SIZE = side length

class PlayerAIOppV2(BaseAI):
    """
    Custom AI Opponent Version 2.
    Uses Expectiminimax, with limits to search power.
    Only applies connected squares heuristics.
    Set DEFAULT_DEPTH_LIMIT = 3.
    """
        
    def __init__(self, depth_limit: int = DEFAULT_DEPTH_LIMIT, verbose: int = 0, grid_size: int = SIZE) -> None:

        super().__init__()
        # self.cape_color = 'blue'
        self.verbose = verbose
        self.pos = None
        self.opp_pos = None
        self.player_num = None
        self.optimal_trap_position = None
        self.depth_limit = depth_limit
        if self.depth_limit < 1:
            self.depth_limit = DEFAULT_DEPTH_LIMIT
        self.dim = grid_size
        self.turns = 1              # early game = 1-3, mid = 4-6, late to 7+; generally early game <= grid.dim/2, mid = 2xearly
        # self.use_advanced_heuristics = heur        # none implemented
        # self.curr_conn_sq = 48
        self.printed = False
        self.search_start_pos = (3, 3)
        self.reached_max_search = False

    def getPosition(self):                      # used by Game to find CURRENT player position
        return self.pos

    def setPosition(self, new_position):        # used by Game to set position once move is validated
        self.pos = new_position

    def setPlayerNum(self, num):                # used by Game to set player num
        self.player_num = num

    def getPlayerNum(self):
        return self.player_num

    def getPlayerPosition(self, grid):          # of current Grid object
        return grid.find(self.getPlayerNum())

    def getOpponentNum(self):
        return 3 - self.getPlayerNum()

    def getOpponentPosition(self, grid: Grid):  # of current Grid object
        return grid.find(self.getOpponentNum())

    def getMove(self, grid: Grid) -> tuple:
        """
        YOUR CODE GOES HERE

        The function should return a tuple of (x,y) coordinates to which the player moves.

        It should be the result of the ExpectiMinimax algorithm, maximizing over the Opponent's *Trap* actions, 
        taking into account the probabilities of them landing in the positions you believe they'd throw to.

        Note that you are not required to account for the probabilities of it landing in a different cell.

        You may adjust the input variables as you wish (though it is not necessary). Output has to be (x,y) coordinates.
        
        """
        if self.verbose:
            self.heur_evals = 0
            self.heur_time = 0
            self.child_nodes_seen = 0
            self.utility = 0
            self.current_depth = 0

        self.reached_max_search = False
        self.opp_pos = self.getOpponentPosition(grid)               # find opponent's current position

        # depth_delta = int((self.turns+3)**(2)/150)            # adjust based on turn
        depth_delta = 0                                         # removing for now
        depth_limit = self.depth_limit + depth_delta

        # get players's current connected sq 
        curr_conn_sq_me, curr_conn_sq_list_me = self.__connected_sq_heur(grid, pos=self.pos, max_size=27, return_pos=True)
        curr_conn_sq_opp, curr_conn_sq_list_opp = self.__connected_sq_heur(grid, pos=self.opp_pos, max_size=27, return_pos=True)
        self.curr_conn_sq_me = curr_conn_sq_me/5
        self.curr_conn_sq_opp = curr_conn_sq_opp/5
        self.curr_conn_sq_list_me = curr_conn_sq_list_me
        self.curr_conn_sq_list_opp = curr_conn_sq_list_opp
        self.search_start_pos = self.__get_search_start_pos(grid)           # get square to start trap searches on
        
        # get maximum traps to search per call to Expectiminimax
        turn_adjust = int(((self.turns+5)**2)/30 - (1/2)*(self.turns+5))    # adjustment based on turn
        depth_adjust = 3*(self.depth_limit//4)                              # adjustment based on depth
        if self.depth_limit >= 6:                                           # set max_traps to return based on depth
            max_search_traps = max(min(3 + turn_adjust - depth_adjust, 47), 8)
        elif self.depth_limit >= 5:
            max_search_traps = max(min(6 + turn_adjust - depth_adjust, 47), 8)
        elif self.depth_limit >= 4:
            max_search_traps = max(min(10 + turn_adjust - depth_adjust, 47), 8)
        else:
            max_search_traps = 5                                            # max = 7 for depth_limit < 4 
        self.max_search_traps = max_search_traps
        self.max_trap_candidates = 0
        print(f'Max search traps = {self.max_search_traps}.')

        alpha = -np.inf
        beta = np.inf
        max_grid = self.__decision(grid, alpha, beta, depth_limit)
        self.turns += 1
        return max_grid.move_position


    def __is_over(self, grid: Grid, turn) -> int:
        """
        Check if game is over, i.e., Player or Opponent has no moves to make
        """
        # check if Player has won
        opponent_neighbors = self.__get_valid_neighbors(grid, self.getOpponentPosition(grid))
        if len(opponent_neighbors) == 0:
            return self.getPlayerNum()

        # check if Opponent has won
        player_neighbors = self.__get_valid_neighbors(grid, self.getPlayerPosition(grid))
        if len(player_neighbors) == 0:
            return self.getOpponentNum()


    def __evaluate(self, grid: Grid, gameover_result) -> tuple:
        """
        Function for returning high or low utility based on gameover state.
        """
        if gameover_result:
            if gameover_result == self.getPlayerNum():
                return grid, 99999
            else:
                return grid, -99999


    def __connected_sq_heur(self, grid: Grid, pos, max_size=27, return_pos=False) -> tuple:
        """
        Get number of connected squares for current player in current Grid object.
        This is a DFS algo
        Returns a tuple of heuristic int, list of positions
        v4
        """
        if self.verbose:
            self.heur_evals += 1

        # if not pos:
        #     if is_me:
        #         pos = self.getPlayerPosition(grid)
        #     else:
        #         pos = self.getOpponentPosition(grid)

        if max_size == -1:
            max_size = 49

        explored = []                   # track explored squares
        search_frontier = [pos]         # initialize with current position
        conn_sq_list = [pos]            # list of connected squares, initialize at player position
        conn_sq_heur = 1                # number of squares connected to position, initialize at 1

        while search_frontier and conn_sq_heur <= max_size:
            current_sq = search_frontier.pop()
            explored.append(current_sq)
            if current_sq not in conn_sq_list:
                conn_sq_list.append(current_sq)
                conn_sq_heur += 1
            for child_pos in self.__get_valid_neighbors(grid, current_sq):
                if child_pos not in explored and child_pos not in search_frontier:
                    # don't need to check if in conn_sq_list, because it is a subset of explored
                    search_frontier.append(child_pos)

        if return_pos:
            # conn_sq_heur = len(conn_sq_list)      # redundant
            # print(f'Current player has {conn_sq_heur} connected squares.')
            return 5*conn_sq_heur, conn_sq_list
        else:
            # print(f'Current player has {conn_sq_heur} connected squares.')
            return 5*conn_sq_heur


    def __get_heuristics(self, grid: Grid) -> tuple:
        """
        Apply heuristics
        Returns a tuple of Grid object, heuristic
        """
        pos = self.getPlayerPosition(grid)
        opp_pos = self.getOpponentPosition(grid)
        return grid, self.__n_neighbors_heur(grid, pos) - self.__n_neighbors_heur(grid, opp_pos)
        # return self.__connected_sq_heur(grid, is_me)


    def __n_neighbors_heur(self, grid: Grid, pos) -> int:
        """
        Returns the difference in available squares around player vs available squares around opponent.
        """
        # if is_me:
        #     pos = self.getPlayerPosition(grid)
        #     # other_pos = self.getOpponentPosition(grid)
        # else:
        #     pos = self.getOpponentPosition(grid)
        #     # other_pos = self.getPlayerPosition(grid)

        # return grid, len(self.__get_valid_neighbors(grid, self.getPlayerPosition(grid))) - len(
        #     self.__get_valid_neighbors(grid, self.getOpponentPosition(grid)))
        # return ( len(self.__get_valid_neighbors(grid, pos)) - len(
        #     self.__get_valid_neighbors(grid, other_pos)) )
        return len(self.__get_valid_neighbors(grid, pos))


    def __probability(self, position, trap_position) -> float:
        """
            Calculates probability of a trap landing in an intended square.
        """
        alpha = manhattan_distance(position, trap_position)
        # alpha = grid_distance(position, trap_position)
        p = 1 - 0.05 * (alpha - 1)
        return p


    def __get_all_neighbors(self, grid, pos, radius=1) -> list:
        """
        Same as __get_valid_neighbors, but includes trap positions
        Returns a list of positions
        
        """
        x,y = pos
        valid_range = lambda t: range(max(t-radius, 0), min(t+radius+1, 7))
        return [(a,b) for a in valid_range(x) for b in valid_range(y)]


    # TODO: not currently used
    def __get_all_neighbors_avoid_edge(self, grid, pos, radius=1) -> list:
        """
        Same as __get_all_valid_neighbors, but avoiding edges
        Returns a list of positions
        
        """
        x,y = pos
        valid_range = lambda t: range(max(t-radius, 1), min(t+radius+1, 6))
        return [(a,b) for a in valid_range(x) for b in valid_range(y)]


    def __get_valid_neighbors(self, grid, pos, radius=1) -> list:
        """
        Description
        -----------
        The function returns the neighboring cells of a certain cell in the board, given its x,y coordinates
        Added additional logic for choosing moves

        Parameters
        -----------
        pos : position (x,y) whose neighbors are desired
        radius (int) : search distance in grid_distance [0, 6]

        Returns a list of positions
        
        """
        x,y = pos
        valid_range = lambda t: range(max(t-radius, 0), min(t+radius+1, 7))
        return [(a,b) for a in valid_range(x) for b in valid_range(y) if grid.map[(a,b)] == 0]


    def __get_valid_neighbors_avoid_edge(self, grid, pos, radius=1) -> list:
        """
        Same as __get_valid_neighbors, but avoiding edges
        Returns a list of positions, including current position
        
        """
        x,y = pos
        valid_range = lambda t: range(max(t-radius, 1), min(t+radius+1, 6))
        return [(a,b) for a in valid_range(x) for b in valid_range(y) if grid.map[(a,b)] == 0]


    def __move_children(self, grid: Grid, is_me=True) -> list:
        """
        get neighbors of a player's intended Move
        returns a list of Grid objects
        """
        if is_me:
            player = self.getPlayerNum()
            position = self.getPlayerPosition(grid)
            other_position = self.getOpponentPosition(grid)
        else:
            player = self.getOpponentNum()
            position = self.getOpponentPosition(grid)
            other_position = self.getPlayerPosition(grid)

        # print('Strategy: move toward center.')


        children = []
        available_moves = []
        if self.turns <= 5:                  # strategy optimization
            available_moves = self.__get_valid_neighbors_avoid_edge(grid, position)
        if not available_moves:
            available_moves = self.__get_valid_neighbors(grid, position)
        for move_position in available_moves:
            move_clone = grid.clone()
            move_clone.move(move_position, player)
            # dynamically creating class attributes at runtime for access in level above in recursive tree
            move_clone.move_position = move_position
            children.append(move_clone)

        return children


    def __get_search_start_pos(self, grid: Grid) -> tuple:
        """
        Start search midway between players, but not on a trap
        Iteratively get closer to the opponent, can be opponent
        """
        def __find_center(pos, opp_pos):
            inc_ceil = lambda i,j: int(np.ceil( (i + j) / 2 ))
            inc_floor = lambda i,j: int(np.floor( (i + j) / 2 ))
            if pos[0] <= opp_pos[0]:
                x = inc_ceil(pos[0], opp_pos[0])
            else:
                x = inc_floor(pos[0], opp_pos[0])

            if pos[1] <= opp_pos[1]:
                y = inc_ceil(pos[1], opp_pos[1])
            else:
                y = inc_floor(pos[1], opp_pos[1])
            # print(x, y)
            # print(x,y)
            # print(grid.map[x,y])
            # print(self.curr_conn_sq_list_opp)
            # print('Here')
            return x, y
        
        pos = self.getPlayerPosition(grid)
        opp_pos = self.getOpponentPosition(grid)
        pos = __find_center(pos, opp_pos)
        while grid.map[pos] == -1 or pos not in self.curr_conn_sq_list_opp:
            pos = __find_center(pos, opp_pos)
            # print(x,y)
            # print(opp_pos)
            # if (x,y) == opp_pos:
            #     print('Starting trap candidate search at Opponent.')
        return pos


    def __get_trap_candidates(self, grid: Grid, pos) -> list:
        """
        Helper function to intelligently locate trap candidates to feed into Expectiminimax.
        First picks a starting location based on behind halfway between the player and opponent.
        Uses a BFS algo to look for connected traps, and then a basic IDS to expand the circle.
        Augmented to use a class param specifying threshold of max traps.
        Returns a list of tuples (positions)
        """
        start_pos = self.search_start_pos                   # find the ideal spot to start trap candidate search
        radius = 1                                          # initialize at 1
        max_radius = 7                                      # not really meaningful but good to be rigorous
        search_frontier = [pos]                             # initialize with position
        explored = []                                       # initialize with position
        trap_candidates = []                                # initialize list of trap candidates to return
        trap_count = 0                                      # initialize count of traps
        new_neighbors = [pos]

        while search_frontier and radius <= max_radius:
            pos = search_frontier.pop()                     # get next items
            explored.append(pos)                            # add to explored
            new_neighbors.remove(pos)
            if pos in self.curr_conn_sq_list_opp and pos not in trap_candidates and not grid.map[pos]:
                # check we're in the opponent's connected component
                # not already in our trap_candidates list
                # and is currently empty 
                trap_candidates.append(pos)                 # add to trap candidates
                trap_count += 1
            # else:
                # print(f'{pos} is not a trap candidate.')
            # print('trap_count =', trap_count)
            # print('self.max_search_traps =', self.max_search_traps)
            if self.verbose and trap_count >= self.max_search_traps:
                self.max_trap_candidates = trap_count
                print(f'Reached max # of trap positions to consider after visiting {self.child_nodes_seen} child nodes.')
                self.reached_max_search = True                             # nothing left to do
                return trap_candidates
            # print('Radius is', radius)
            while not new_neighbors:
                if self.turns <= 5:                         # strategy optimization
                    new_neighbors = self.__get_valid_neighbors_avoid_edge(grid, start_pos, radius)
                if not new_neighbors:                       # 
                    new_neighbors = self.__get_valid_neighbors(grid, start_pos, radius)
                radius += 1
                # print(new_neighbors)
            search_frontier = list(set(search_frontier + new_neighbors) - set(explored))
            # slightly faster than list comprehension
            # no need to intersect with trap_candidates, because it is a subset of explored

        if trap_count == 0:
            # print(pos)
            # return [grid.getAvailableCells()[0]]
            # print(f'Error at depth = {self.current_depth}.')        # incorrect reason
            
            print(f'No trap positions left after visiting {self.child_nodes_seen} child nodes.') if self.verbose else None
            self.reached_max_search = True                             # nothing left to do
            
            catch_trap = self.__get_valid_neighbors_avoid_edge(grid, start_pos, radius)
            if not catch_trap:
                catch_trap = PlayerAI.__get_valid_neighbors(grid, start_pos, radius)
            if not catch_trap:
                catch_trap = grid.getAvailableCells()[0]
            return catch_trap
            # raise Exception('trap_count is 0')                      # only for testing

        if trap_count > self.max_trap_candidates:
            self.max_trap_candidates = trap_count
            # print('Reached max trap candidates.')
            # input()
            # print(trap_candidates)

        # if not self.printed:
            # print(f'Traps seen: {trap_count}')
            # self.printed = True
        return trap_candidates


    def __trap_children(self, grid: Grid, is_me=True) -> list:
        """
        a player (either player or opponent) throws Trap
        function expands node of trades from param position
        returns a list of Grid objects
        """
        if is_me:
            player = self.getPlayerNum()
            position = grid.move_position # hypothetical move
            other_position = self.getOpponentPosition(grid)
        else:
            player = self.getOpponentNum()
            position = grid.move_position # hypothetical move
            other_position = self.getPlayerPosition(grid)

        children = []
        available_traps = self.__get_trap_candidates(grid, other_position)
        for trap_position in available_traps:
            if trap_position != position:
                trap_clone = grid.clone()
                trap_clone.trap(trap_position)
                # dynamically creating class attributes at runtime for access at the very top of the search tree
                trap_clone.trap_position = trap_position
                trap_clone.probability = self.__probability(position, trap_position)
                children.append(trap_clone)

        return children


    def __trap_neighbors(self, grid: Grid, trap_position) -> list:
        """
        get neighbors of intended trap_pos
        returns a list of Grid objects
        """
        neighbors = []

        for trap_position in self.__get_valid_neighbors(grid, trap_position):
            trap_clone = grid.clone()
            trap_clone.trap(trap_position)
            # dynamically creating class attributes at runtime
            trap_clone.trap_position = trap_position
            neighbors.append(trap_clone)

        return neighbors


    def __trap_minimize(self, grid: Grid, alpha, beta, depth, depth_limit) -> tuple:
        """
        opponent Min node for throwing Trap
        returns a tuple of Grid object, utility
        returns a grid object and associated utility
        """
        if self.verbose > 2:
            self.child_nodes_seen += 1
            self.current_depth += 1
            # print(f'In Trap Min, Current depth = {self.current_depth}.')

        gameover_result = self.__is_over(grid, self.getPlayerNum())
        if gameover_result:
            # if game ends because move above results in a gameover, then we need to place a valid trap somewhere randomly
            # grid = self.__trap_children(grid, is_me=False)[0]
            return self.__evaluate(grid, gameover_result)

        # break if hit depth limit
        if depth >= depth_limit:
            self.reached_max_search = True                             # nothing left to do
            _, utility = self.__get_heuristics(grid)
            return _, utility

        minTrap, minUtility = None, np.inf
        cache = {}

        for trap in self.__trap_children(grid, is_me=False):
            # initialize with the main trap's probability-weighted utility, then move on to those of the neighbors
            key = tuple(map(tuple, trap.map))
            if key in cache:
                utility = cache[key]
            else:
                _, utility = self.__move_maximize(trap, alpha, beta, depth+1, depth_limit)
                cache[key] = utility
            expected_utility = trap.probability * utility

            neighbors = self.__trap_neighbors(grid, trap.trap_position)
            for neighbor in neighbors:
                key = tuple(map(tuple, neighbor.map))
                if key in cache:
                    utility = cache[key]
                else:
                    _, utility = self.__move_maximize(neighbor, alpha, beta, depth+1, depth_limit)
                    cache[key] = utility
                expected_utility += (1-trap.probability)/len(neighbors) * utility

            if expected_utility < minUtility:
                minUtility = expected_utility

            if self.reached_max_search:         # if there is nothing more to do
                break

        return minTrap, minUtility


    def __move_minimize(self, grid: Grid, alpha, beta, depth, depth_limit) -> tuple:
        """
        opponent Min node for making Move
        returns a tuple of Grid object, utility
        returns a grid object and associated utility
        """
        if self.verbose:
            self.child_nodes_seen += 1
            self.current_depth += 1
            # print(f'In Move Min, Current depth = {self.current_depth}.')
            # print(f'Depth = {depth}, depth_limit = {depth_limit}')

        gameover_result = self.__is_over(grid, self.getOpponentNum())
        if gameover_result:
            return self.__evaluate(grid, gameover_result)

        # break if hit depth limit
        if depth >= depth_limit:
            self.reached_max_search = True                             # nothing left to do
            _, utility = self.__get_heuristics(grid)
            return _, utility

        minChild, minUtility = None, np.inf

        for child in self.__move_children(grid, is_me=False):
            _, utility = self.__trap_minimize(child, alpha, beta, depth+1, depth_limit)

            if utility < minUtility:
                minChild, minUtility = child, utility

            if minUtility <= alpha:
                break

            if minUtility < beta:
                beta = minUtility

            if self.reached_max_search:         # if there is nothing more to do
                break

        return minChild, minUtility


    def __trap_maximize(self, grid: Grid, alpha, beta, depth, depth_limit) -> tuple:
        """
        player Max node for throwing Trap
        returns a tuple of Grid object, utility
        returns a grid object and associated utility
        """
        if self.verbose:
            self.child_nodes_seen += 1
            self.current_depth += 1
            # print(f'In Trap Max, Current depth = {self.current_depth}.')

        gameover_result = self.__is_over(grid, self.getPlayerNum())
        if gameover_result:
            # if game ends because move above results in a gameover, then we need to place a valid trap somewhere randomly
            # grid = self.__trap_children(grid, is_me=True)[0]
            return self.__evaluate(grid, gameover_result)

        # break if (exceed) hit depth limit
        if depth > depth_limit:
            self.reached_max_search = True                             # nothing left to do
            _, utility = self.__get_heuristics(grid)
            return _, utility

        maxTrap, maxUtility = None, -np.inf
        cache = {}

        for trap in self.__trap_children(grid, is_me=True):
            # initialize with the main trap's probability-weighted utility, then move on to those of the neighbors
            key = tuple(map(tuple, trap.map))
            if key in cache:
                utility = cache[key]
            else:
                _, utility = self.__move_minimize(trap, alpha, beta, depth+1, depth_limit)
                cache[key] = utility
            expected_utility = trap.probability * utility

            neighbors = self.__trap_neighbors(grid, trap.trap_position)
            for neighbor in neighbors:
                key = tuple(map(tuple, neighbor.map))
                if key in cache:
                    utility = cache[key]
                else:
                    _, utility = self.__move_minimize(neighbor, alpha, beta, depth+1, depth_limit)
                    cache[key] = utility
                expected_utility += (1-trap.probability)/len(neighbors) * utility

            if expected_utility > maxUtility:
                maxTrap, maxUtility = trap, expected_utility

            if self.reached_max_search:         # if there is nothing more to do
                break

        # returns max trap so maximize can cache it
        return maxTrap, maxUtility


    def __move_maximize(self, grid: Grid, alpha, beta, depth, depth_limit) -> tuple:
        """
        player Min node for making Move
        returns a tuple of Grid object, utility
        """
        if self.verbose:
            self.child_nodes_seen += 1
            self.current_depth += 1
            # print(f'In Move Max, Current depth = {self.current_depth}.')

        gameover_result = self.__is_over(grid, self.getPlayerNum())
        if gameover_result:
            return self.__evaluate(grid, gameover_result)

        # break if (exceed) hit depth limit
        if depth > depth_limit:
            self.reached_max_search = True                             # nothing left to do
            _, utility = self.__get_heuristics(grid)
            return _, utility

        maxMove, maxTrap, maxUtility = None, None, -np.inf

        for move in self.__move_children(grid, is_me=True):
            trap, utility = self.__trap_maximize(move, alpha, beta, depth+1, depth_limit)
            if utility > maxUtility:
                maxMove, maxTrap, maxUtility = move, trap, utility

            if maxUtility >= beta:
                break

            if maxUtility > alpha:
                alpha = maxUtility

            if self.reached_max_search:         # if there is nothing more to do
                break

        if hasattr(maxTrap, 'trap_position'):
            self.optimal_trap_position = maxTrap.trap_position

        return maxMove, maxUtility


    def __decision(self, grid: Grid, alpha, beta, depth_limit=DEFAULT_DEPTH_LIMIT) -> object:
        """
        Helper function to start the Expectiminimax algo
        returns a Grid object
        """
        start = time.time()
        # child, _ = self.__move_maximize(grid, alpha, beta, depth=0, depth_limit=depth_limit)
        child, self.utility = self.__move_maximize(grid, alpha, beta, depth=0, depth_limit=depth_limit)
        end = time.time()
        # print(f'This move took {end-start:.5f} seconds.')
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

        # edge case - if there are no available cells around opponent, then OppV2 player will win
        # throw to first available cell, starting from upper-left-most square
        # TODO: change the trap position to be somewhere not in the vicinity of current player
        if not self.__get_valid_neighbors(grid, self.getOpponentPosition(grid)):
            print('OppV2')
            input(f'No available cells around player {3 - self.player_num}! Press enter to continue.') if self.verbose else None
            return grid.getAvailableCells()[0]

        # use cached optimal trap position that we computed in getMove()
        if self.optimal_trap_position is None:
            input('No optional trap position in cache.') if self.verbose else None
            return grid.getAvailableCells()[0]
        return self.optimal_trap_position