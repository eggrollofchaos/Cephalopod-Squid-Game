# gc2950
# mhr2145
# wax1
import numpy as np
import random
import time
import sys
import os
import queue as Q
from BaseAI import BaseAI
from Grid import Grid
from Utils import manhattan_distance, grid_distance
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from termcolor import cprint

DEFAULT_DEPTH_LIMIT = 4

class PlayerAI(BaseAI):
    def __init__(self, depth_limit=DEFAULT_DEPTH_LIMIT, heur=None) -> None:
        self.cape_color = 'blue'
        super().__init__()
        self.pos = None
        self.opp_pos = None
        self.player_num = None
        self.optimal_trap_pos = None
        self.depth_limit = depth_limit
        if self.depth_limit == 0:
            self.depth_limit = DEFAULT_DEPTH_LIMIT
        self.turns = 1              # early game = 1-3, mid = 4-6, late to 7+; generally early game <= grid.dim/2, mid = 2xearly
        self.use_advanced_heuristics = heur
        self.use_graph_me = False
        self.use_graph_opp = False
        self.curr_conn_sq = 48
        self.printed = False
        self.search_start_pos = (3, 3)

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
        # if self.turns > 1:
        #     exit(1)
        self.heur_evals = 0
        self.heur_time = 0
        self.opp_pos = self.getOpponentPosition(grid)               # find opponent's current position
        self.utility = 0
        self.current_depth = 0

        # depth_delta = int((self.turns+3)**(2)/150)            # adjust based on turn
        depth_delta = 0                                         # adjust based on turn
        depth_limit = self.depth_limit + depth_delta
        print(f'Depth limit is currently {depth_limit}.')

        # get players's current connected sq 
        curr_conn_sq_me, curr_conn_sq_list_me = self.__connected_sq_heur(grid, pos=self.pos, max_size=27, return_pos=True, is_me=True)
        curr_conn_sq_opp, curr_conn_sq_list_opp = self.__connected_sq_heur(grid, pos=self.opp_pos, max_size=27, return_pos=True, is_me=False)
        self.curr_conn_sq_me = curr_conn_sq_me/5
        self.curr_conn_sq_opp = curr_conn_sq_opp/5
        self.curr_conn_sq_list_me = curr_conn_sq_list_me
        self.curr_conn_sq_list_opp = curr_conn_sq_list_opp
        self.search_start_pos = self.__get_search_start_pos(grid)       # get square to start trap searches on
        if self.curr_conn_sq_me >= 27 or self.curr_conn_sq_opp >= 27:
            print(f'Player\'s and opponent\'s component both have a lot of connected squares.')
        else:
            print(f'Player\'s component has {int(self.curr_conn_sq_me)} connected squares; opponent\'s component has {int(self.curr_conn_sq_opp)}.')
        print(f'Trap search will start at {self.search_start_pos}.')
        
        # get maximum traps to search per call to Expectiminimax
        turn_adjust = int(((self.turns+5)**2)/30 - (1/2)*(self.turns+5))     # adjustment based on turn
        depth_adjust = 3*(self.depth_limit//4)               # adjustment based on depth
        if self.depth_limit >= 6:                           # set max_traps to return based on depth
            max_search_traps = max(min(3 + turn_adjust - depth_adjust, 47), 1)
        elif self.depth_limit >= 5:
            max_search_traps = max(min(6 + turn_adjust - depth_adjust, 47), 1)
        elif self.depth_limit >= 4:
            max_search_traps = max(min(10 + turn_adjust - depth_adjust, 47), 1)
        else:
            max_search_traps = 47                           # effectively no limit
        self.max_search_traps = max_search_traps
        self.max_trap_candidates = 0

        print(f'Max traps to search = {self.max_search_traps}; ', end='')

        alpha = -np.inf
        beta = np.inf
        max_move = self.__decision(grid, alpha, beta, depth_limit)
        print(f'Total iterative search evals = {self.heur_evals}')
        self.turns += 1
        return max_move


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


    def __evaluate(self, grid: Grid, gameover_result) -> int:
        """
        Function for returning high or low utility based on gameover state.
        """
        if gameover_result:
            if gameover_result == self.getPlayerNum():
                return grid, 99999
            else:
                return grid, -99999


    def __get_dof(self, grid: Grid, pos, opp_pos) -> tuple:
        """
        Apply degrees of freedom calculation heuristic based on early, mid, late game.
        Returns a tuple of Grid object, heuristic
        """
        center_heur = 0
        toward_opp_heur = 0
        n_neighbors_heur_me = 0
        n_neighbors_heur_opp = 0
        n_neighbors_heur = 0
        edge_touch_heur = 0
        n_conn_sq_heur_me = 48
        conn_sq_list_me = []
        n_conn_sq_heur_opp = 48
        conn_sq_list_opp = []
        n_conn_sq_heur = 0
        conn_sq_depth_lim_me = 0
        conn_sq_depth_lim_opp = 0
        conn_sq_depth_lim_heur = 0
        graph_cut_heur_me = 0
        graph_cut_heur_opp = 0
        graph_cut_heur = 0

        center_heur = self.__center_heur(grid, pos) - self.__center_heur(grid, opp_pos)
        toward_opp_heur = self.__toward_opp_heur(grid, pos, opp_pos)
        if self.turns >= 1:
            n_neighbors_heur_me = self.__n_neighbors_heur(grid, pos)
        if self.turns >= 2:
            n_neighbors_heur_opp = self.__n_neighbors_heur(grid, opp_pos)
        if self.turns >= 3:
            edge_touch_heur = self.__edge_touch_heur(grid, pos) - self.__edge_touch_heur(grid, opp_pos)
        
        if self.turns >= 4:
            conn_sq_depth_lim_me = self.__connected_sq_heur(grid, pos, max_size=10)
        elif self.turns >= 5:
            conn_sq_depth_lim_me = self.__connected_sq_heur(grid, pos, max_size=10)
            conn_sq_depth_lim_opp = self.__connected_sq_heur(grid, opp_pos, max_size=10)
        elif self.turns >= 6:
            n_conn_sq_heur_me = self.__connected_sq_heur(grid, pos, max_size=18)
        elif self.turns >= 7:
            n_conn_sq_heur_me = self.__connected_sq_heur(grid, pos, max_size=18)
            n_conn_sq_heur_opp = self.__connected_sq_heur(grid, opp_pos, max_size=18)
        elif self.turns >= 8:
            n_conn_sq_heur_me, conn_sq_list_me = self.__connected_sq_heur(grid, pos, max_size=35, return_pos=True)
            n_conn_sq_heur_opp, conn_sq_list_opp = self.__connected_sq_heur(grid, opp_pos, max_size=35, return_pos=True)
            # grid.print_grid()
        # if self.turns >= 9:

        turns_limit = (self.turns - 2*np.ceil(self.depth_limit/3) >= 9)
        conn_sq_limit_me = (n_conn_sq_heur_me + 3*(self.depth_limit//4) <= 30)
        conn_sq_limit_opp = (n_conn_sq_heur_opp + 3*(self.depth_limit//4) <= 30)

        if turns_limit and conn_sq_limit_me:
            self.use_graph_me = True
            # cprint('Using graph cut heuristic on player.', 'blue')
            graph_cut_heur_me = self.__graph_cut_heur(grid, pos, comp_size=n_conn_sq_heur_me, conn_sq=conn_sq_list_me)

        if turns_limit and conn_sq_limit_opp:
        # if self.turns >= 12 or ( self.curr_conn_sq_opp * < 24:
            self.use_graph_opp = True
            # cprint('Using graph cut heuristic on opponent.', 'blue')
            graph_cut_heur_opp = self.__graph_cut_heur(grid, pos, comp_size=n_conn_sq_heur_opp, conn_sq=conn_sq_list_opp)

        n_neighbors_heur = n_neighbors_heur_me - n_neighbors_heur_opp
        conn_sq_depth_lim_heur = conn_sq_depth_lim_me - conn_sq_depth_lim_opp
        n_conn_sq_heur = n_conn_sq_heur_me - n_conn_sq_heur_opp
        graph_cut_heur = graph_cut_heur_me - graph_cut_heur_opp

        return grid, center_heur + toward_opp_heur + conn_sq_depth_lim_heur + n_conn_sq_heur + n_neighbors_heur + edge_touch_heur + graph_cut_heur

    def __clone(self, grid: Grid) -> object:
        """
        Makes a full copy of current grid
        """
        grid_copy = Grid(7)
        grid_copy.map = grid.map.copy()
        return grid_copy

      
    def __edge_touch_heur(self, grid: Grid, pos) -> int:
        """
        Heuristic that penalizes being on an edge (more if on two edges, i.e. a corner)
        Returns a heuristic int
        """
        edges_touched = 0
        if pos[0] in (0,6):
            edges_touched -= 1
        if pos[1] in (0,6):
            edges_touched -= 1
        return 10*edges_touched


    def __center_heur(self, grid: Grid, pos) -> int:
        """
        Heuristic that priotizes being closer to center
        Returns a heuristic int
        """
        dist = 6 - grid_distance(pos, (3,3))
        return 1*dist


    def __toward_opp_heur(self, grid: Grid, pos, opp_pos) -> int:
        """
        Heuristic that prioritizes being closer to opponent
        Returns a heuristic int
        """
        dist = 6-grid_distance(pos, opp_pos)
        return 2*dist


    def __connected_sq_heur(self, grid: Grid, pos, max_size=27, return_pos=False, is_me=True) -> tuple:
        """
        Get number of connected squares for current player in current Grid object.
        This is a DFS algo
        Returns a tuple of heuristic int, list of positions
        v4
        """
        self.heur_evals += 1

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


    def __graph_cut_heur(self, grid: Grid, pos, comp_size, conn_sq) -> int:
        """
        Heuristic based on how easy it is to increase the number of connected components within the board,
        and how drastic the decrease in freedom resulting from the removal of one free space
        Returns a heuristic int
        """
        # conn_comps = self.__num_connected_components(grid, is_me=is_me)         # number of connected components
        
        graph_cut_heur = 0
        for trap_pos in conn_sq:                                                # iterate over connected squares
            trap_clone = self.__clone(grid)
            # trap_clone.trap(trap_pos)
            trap_clone.map[trap_pos] = -1
            # new_conn_comps = self.__num_connected_components(grid, is_me=is_me)
            new_comp_size = self.__connected_sq_heur(grid, pos, pos, return_pos=False)
            # comp_delta = new_conn_comps - conn_comps                          # this will be 0 or -1
            size_delta = new_comp_size - comp_size
            # utility = 10*comp_delta + size_delta - (47-new_comp_size)**3
            utility = 10*size_delta - (47-new_comp_size)**3
            graph_cut_heur += utility
        return graph_cut_heur


    def __num_connected_components(self, grid: Grid, return_labels=False, is_me=True) -> int:
        """
        Convert board to graph; get number of connected components.
        """
        graph = grid.map
        graph = np.where(graph==0, 1, graph)
        graph = np.where(graph==2, 0, graph)
        graph = np.where(graph==-1, 0, graph)
        graph = csr_matrix(graph)
        return connected_components(csgraph=graph, directed=False, return_labels=return_labels)


    def __get_heuristics(self, grid: Grid, is_me) -> tuple:
        """
        Apply heuristics
        Returns a tuple of Grid object, heuristic
        """
        pos = self.getPlayerPosition(grid)
        opp_pos = self.getOpponentPosition(grid)
        if self.use_advanced_heuristics == 'graphcut':
            start = time.time()
            dof_heur = self.__get_dof(grid, pos=pos, opp_pos=opp_pos)
            end = time.time()
            self.heur_time += end-start
            return dof_heur
        elif self.use_advanced_heuristics == 'geodesics':
            start = time.time()
            # conn_sq_dl_me = self.__conn_sq_depth_lim_heur(grid, pos)
            # conn_sq_dl_opp = self.__conn_sq_depth_lim_heur(grid, opp_pos)
            conn_sq_dl_me = self.__connected_sq_heur(grid, pos, max_size=25)
            conn_sq_dl_opp = self.__connected_sq_heur(grid, opp_pos, max_size=25)
            dl_heur = conn_sq_dl_me-conn_sq_dl_opp
            end = time.time()
            self.heur_time += end-start
            return grid, dl_heur                   # for next set of heuristics Matthew/Gong
        else:
            return grid, self.__n_neighbors_heur(grid, pos) - self.__n_neighbors_heur(grid, opp_pos)
        # return self.__connected_sq_heur(grid, is_me)


    def __n_neighbors_heur(self, grid: Grid, pos) -> int:
        """
        Returns the difference in available squares around player vs available squares around opponent.
        Now this just returns the same as __get_valid_neighbors - will be deprecated soon.
        """
        return len(self.__get_valid_neighbors(grid, pos))


    def __probability(self, pos, trap_pos) -> float:
        """
            Calculates probability of a trap landing in an intended square.
        """
        alpha = manhattan_distance(pos, trap_pos)
        # alpha = grid_distance(pos, trap_pos)
        p = 1 - 0.05 * (alpha - 1)
        return p


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


    def __get_all_neighbors(self, grid, pos, radius=1) -> list:
        """
        Same as __get_valid_neighbors, but includes trap positions
        Returns a list of positions
        
        """
        x,y = pos
        valid_range = lambda t: range(max(t-radius, 0), min(t+radius+1, 7))
        return [(a,b) for a in valid_range(x) for b in valid_range(y)]


    def __get_all_neighbors_avoid_edge(self, grid, pos, radius=1) -> list:
        """
        Same as __get_all_valid_neighbors, but avoiding edges
        Returns a list of positions
        
        """
        x,y = pos
        valid_range = lambda t: range(max(t-radius, 1), min(t+radius+1, 6))
        return [(a,b) for a in valid_range(x) for b in valid_range(y)]


    def __move_children(self, grid: Grid, is_me=True) -> list:
        """
        get neighbors of a player's intended Move
        returns a list of Grid objects
        """
        if is_me:
            player = self.getPlayerNum()
            pos = self.getPlayerPosition(grid)
            other_pos = self.getOpponentPosition(grid)
        else:
            player = self.getOpponentNum()
            pos = self.getOpponentPosition(grid)
            other_pos = self.getPlayerPosition(grid)

        # print('Strategy: move toward center.')


        children = []
        available_moves = []
        if self.turns <= 5:                  # strategy optimization
            available_moves = self.__get_valid_neighbors_avoid_edge(grid, pos)
        if not available_moves:
            available_moves = self.__get_valid_neighbors(grid, pos)
        for move_pos in available_moves:
            move_clone = self.__clone(grid)
            move_clone.move(move_pos, player)
            # dynamically creating class attributes at runtime for access in level above in recursive tree
            move_clone.move_pos = move_pos
            children.append(move_clone)

        return children


    def __get_search_start_pos(self, grid: Grid) -> tuple:
        '''
        Start search midway between players, but not on a trap
        Iteratively get closer to the opponent, can be opponent
        '''
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
            if trap_count >= self.max_search_traps:
                self.max_trap_candidates = trap_count
                # print('Reached max # of trap positions to consider.')
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
            print(f'Error at depth = {self.current_depth}.')
            catch_trap = self.__get_valid_neighbors_avoid_edge(grid, start_pos, radius)
            if not catch_trap:
                catch_trap = self.__get_valid_neighbors(grid, start_pos, radius)
            return catch_trap
            raise Exception('trap_count is 0')

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
            pos = grid.move_position # hypothetical move
            other_pos = self.getOpponentPosition(grid)
        else:
            player = self.getOpponentNum()
            pos = grid.move_position # hypothetical move
            other_pos = self.getPlayerPosition(grid)

        children = []
        available_traps = self.__get_trap_candidates(grid, other_position)
        for trap_pos in available_traps:
            if trap_pos != pos:
                trap_clone = self.__clone(grid)
                # trap_clone.trap(trap_pos)
                trap_clone.map[trap_pos] = -1
                # dynamically creating class attributes at runtime for access at the very top of the search tree
                trap_clone.trap_pos = trap_pos
                trap_clone.probability = self.__probability(pos, trap_pos)
                children.append(trap_clone)

        return children


    def __trap_neighbors(self, grid: Grid, trap_pos) -> list:
        """
        get neighbors of intended trap_pos
        returns a list of Grid objects
        """
        neighbors = []

        for trap_pos in self.__get_valid_neighbors(grid, trap_pos):
            trap_clone = self.__clone(grid)
            # trap_clone.trap(trap_pos)
            trap_clone.map[trap_pos] = -1
            # dynamically creating class attributes at runtime
            trap_clone.trap_pos = trap_pos
            neighbors.append(trap_clone)

        return neighbors


    def __trap_minimize(self, grid: Grid, alpha, beta, depth, depth_limit) -> tuple:
        """
        opponent Min node for throwing Trap
        returns a tuple of Grid object, utility
        returns a grid object and associated utility
        """
        self.current_depth += 1
        gameover_result = self.__is_over(grid, self.getPlayerNum())
        if gameover_result:
            # if game ends because move above results in a gameover, then we need to place a valid trap somewhere randomly
            # grid = self.__trap_children(grid, is_me=False)[0]
            return self.__evaluate(grid, gameover_result)
        if depth >= depth_limit:
            return self.__get_heuristics(grid, is_me=True)

        minTrap, minUtility = None, np.inf
        cache = {}
        player_pos = self.getPlayerPosition(grid)
        opponent_pos = self.getOpponentPosition(grid)
        # trap_pos is a list of e.g. [(2,3), (3,4)]
        for trap_pos in self.__get_trap_candidates(grid, player_pos):
            # initialize with the main trap's probability-weighted utility, then move on to those of the neighbors
            grid.map[trap_pos] = -1
            key = tuple(map(tuple, grid.map))
            if key in cache:
                utility = cache[key]
            else:
                _, utility = self.__move_maximize(grid, alpha, beta, depth+1, depth_limit)
                cache[key] = utility
            target_prob = self.__probability(opponent_pos, trap_pos)
            expected_utility = target_prob * utility
            # backtrack
            grid.map[trap_pos] = 0

            neighbors = grid.get_neighbors(trap_pos, only_available=True)
            for neighbor in neighbors:
                grid.map[neighbor] = -1
                key = tuple(map(tuple, grid.map))
                if key in cache:
                    utility = cache[key]
                else:
                    _, utility = self.__move_maximize(grid, alpha, beta, depth + 1, depth_limit)
                    cache[key] = utility
                grid.map[neighbor] = 0
                expected_utility += (1 - target_prob) / len(neighbors) * utility

            if expected_utility < minUtility:
                minTrap, minUtility = trap_pos, expected_utility
        return minTrap, minUtility


    def __move_minimize(self, grid: Grid, alpha, beta, depth, depth_limit) -> tuple:
        """
        opponent Min node for making Move
        returns a tuple of Grid object, utility
        returns a grid object and associated utility
        """
        self.current_depth += 1
        gameover_result = self.__is_over(grid, self.getOpponentNum())
        if gameover_result:
            return self.__evaluate(grid, gameover_result)

        # break if hit depth limit
        if depth >= depth_limit:
            return self.__get_heuristics(grid, is_me=True)

        minChild, minUtility = None, np.inf
        opponent_pos = self.getOpponentPosition(grid)
        opponent_num = self.getOpponentNum()
        for neighbor_to_move in grid.get_neighbors(opponent_pos, only_available=True):
            grid.map[opponent_pos] = 0
            grid.map[neighbor_to_move] = opponent_num
            _, utility = self.__trap_minimize(grid, alpha, beta, depth + 1, depth_limit)
            grid.map[opponent_pos] = opponent_num
            grid.map[neighbor_to_move] = 0
            if utility < minUtility:
                minChild, minUtility = neighbor_to_move, utility

            if minUtility <= alpha:
                break

            if minUtility < beta:
                beta = minUtility

        return minChild, minUtility


    def __trap_maximize(self, grid: Grid, alpha, beta, depth, depth_limit) -> tuple:
        """
        player Max node for throwing Trap
        returns a tuple of Grid object, utility
        returns a grid object and associated utility
        """
        self.current_depth += 1
        gameover_result = self.__is_over(grid, self.getPlayerNum())
        if gameover_result:
            # if game ends because move above results in a gameover, then we need to place a valid trap somewhere randomly
            # grid = self.__trap_children(grid, is_me=True)[0]
            return self.__evaluate(grid, gameover_result)

        if depth >= depth_limit:
            return self.__get_heuristics(grid, is_me=True)

        maxTrap, maxUtility = None, -np.inf
        cache = {}

        player_pos = self.getPlayerPosition(grid)
        opponent_pos = self.getOpponentPosition(grid)
        for trap_pos in self.__get_trap_candidates(grid, opponent_pos):
            # initialize with the main trap's probability-weighted utility, then move on to those of the neighbors
            grid.map[trap_pos] = -1
            key = tuple(map(tuple, grid.map))
            if key in cache:
                utility = cache[key]
            else:
                _, utility = self.__move_minimize(grid, alpha, beta, depth + 1, depth_limit)
                cache[key] = utility
            target_prob = self.__probability(player_pos, trap_pos)
            expected_utility = target_prob * utility
            grid.map[trap_pos] = 0

            neighbors = grid.get_neighbors(trap_pos, only_available=True)
            for neighbor in neighbors:
                grid.map[neighbor] = -1
                key = tuple(map(tuple, grid.map))
                if key in cache:
                    utility = cache[key]
                else:
                    _, utility = self.__move_minimize(grid, alpha, beta, depth + 1, depth_limit)
                    cache[key] = utility
                grid.map[neighbor] = 0
                expected_utility += (1 - target_prob) / len(neighbors) * utility

            if expected_utility > maxUtility:
                maxTrap, maxUtility = trap_pos, expected_utility
        # returns max trap so maximize can cache it
        return maxTrap, maxUtility


    def __move_maximize(self, grid: Grid, alpha, beta, depth, depth_limit) -> tuple:
        """
        player Min node for making Move
        returns a tuple of Grid object, utility
        """
        self.current_depth += 1
        gameover_result = self.__is_over(grid, self.getPlayerNum())
        if gameover_result:
            return self.__evaluate(grid, gameover_result)

        # break if hit depth limit
        if depth >= depth_limit:
            return self.__get_heuristics(grid, is_me=True)

        maxMove, maxTrap, maxUtility = None, None, -np.inf
        player_pos = self.getPlayerPosition(grid)
        player_num = self.getPlayerNum()
        for neighbor_to_move in grid.get_neighbors(player_pos, only_available=True):
            grid.map[player_pos] = 0
            grid.map[neighbor_to_move] = player_num
            trap, utility = self.__trap_maximize(grid, alpha, beta, depth + 1, depth_limit)
            grid.map[player_pos] = player_num
            grid.map[neighbor_to_move] = 0

            if utility > maxUtility:
                maxMove, maxTrap, maxUtility = neighbor_to_move, trap, utility
            if maxUtility >= beta:
                break

            if maxUtility > alpha:
                alpha = maxUtility

        self.optimal_trap_pos = maxTrap        # maxTrap is now just a position, not a Grid object

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
        print(f'max trap candidates seen from one child: {self.max_trap_candidates}')
        if self.use_advanced_heuristics:
            print(f'Heuristics took {self.heur_time:.3f} seconds to complete.')
        # if end-start >= 5.05:
        if self.use_graph_me:
            print('Used graph cut heuristic on player.')
        if self.use_graph_opp:
            print('Used graph cut heuristic on opponent.')
        print(f'Best move found has utility of {self.utility:.2f}. Hey, doing what we can.')
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
        # use cached optimal trap position that we computed in getMove()
        if not self.optimal_trap_pos:
            print('No trap positions')
            input()
            return grid.getAvailableCells()[0]
        return self.optimal_trap_pos