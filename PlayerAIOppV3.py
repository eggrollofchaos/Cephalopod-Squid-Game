# wax
import numpy as np
import random
import time
import sys
import os
import queue as Q
from BaseAI import BaseAI
from Grid import Grid
from Utils import manhattan_distance, grid_distance
from termcolor import cprint

DEFAULT_DEPTH_LIMIT = 4

class PlayerAIOppV3(BaseAI):
    def __init__(self, depth_limit=DEFAULT_DEPTH_LIMIT, heur=None, verbose=False) -> None:
        '''
        Custom AI Opponent Version 3.
        Uses Expectiminimax.
        Applies most of the same heuristics as PlayerAI.
        Set DEFAULT_DEPTH_LIMIT = 4.
        Set starting max_search_traps to minimum of 5.
        '''
        self.verbose = verbose
        super().__init__()
        self.pos = None
        self.opp_pos = None
        self.player_num = None
        self.optimal_trap_pos = None
        self.depth_limit = depth_limit
        if self.depth_limit < 1:
            self.depth_limit = DEFAULT_DEPTH_LIMIT
        self.turns = 1              # early game = 1-3, mid = 4-6, late to 7+; generally early game <= grid.dim/2, mid = 2xearly
        self.use_advanced_heuristics = heur
        # self.curr_conn_sq = 48
        self.search_start_pos = (3, 3)
        self.graph_cut_size_cap = 8

        self.max_search_traps = 12
        # max_search_traps starts at 12, but needs to be adjusted down at early game if depth_level is higher than 5
        self.max_search_traps += min(2*(5 - self.depth_limit), 0)
        self.max_search_traps = min(5, abs(self.max_search_traps))      # no less than 5 to begin with

        self.max_radius = 3
        self.use_graph_me = False
        self.use_graph_d2_me = False
        self.use_graph_opp = False
        self.use_graph_d2_opp = False

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

        # get player numbers variables
        player_num = self.getPlayerNum()
        opp_num = self.getOpponentNum()
        self.opp_pos = tuple(np.argwhere(grid.map == opp_num)[0])              # find opponent's current position

        if self.verbose:
            self.heur_evals = 0
            self.heur_time = 0
            self.child_nodes_seen = 0
            self.printed = False
            self.utility = 0
            self.current_depth = 0
            self.max_child_nodes = 50000

        print(f'Depth limit is {self.depth_limit}.') if self.verbose else None

        # get players's current connected sq 
        # curr_conn_sq_me, curr_conn_sq_list_me = self.__connected_sq_heur(grid, pos=self.pos, max_size=27, return_pos=True)
        # curr_conn_sq_opp, curr_conn_sq_list_opp = self.__connected_sq_heur(grid, pos=self.opp_pos, max_size=27, return_pos=True)
        curr_conn_sq_me, curr_conn_sq_list_me = self.__conn_sq_depth_lim_heur(grid, pos=self.pos, max_radius=3, return_pos=True)
        curr_conn_sq_opp, curr_conn_sq_list_opp = self.__conn_sq_depth_lim_heur(grid, pos=self.opp_pos, max_radius=3, return_pos=True)
        self.curr_conn_sq_me = curr_conn_sq_me/5
        self.curr_conn_sq_opp = curr_conn_sq_opp/5
        self.curr_conn_sq_list_me = curr_conn_sq_list_me
        self.curr_conn_sq_list_opp = curr_conn_sq_list_opp
        self.search_start_pos = self.__get_search_start_pos(grid, self.pos, self.opp_pos)       # get square to start trap searches on
        
        if self.verbose:
            # if self.curr_conn_sq_me >= 28 or self.curr_conn_sq_opp >= 28:
            #     print(f'Player\'s and opponent\'s component both have a lot of connected squares.')
            # else:
            #     print(f'Player\'s component has {int(self.curr_conn_sq_me)} connected squares, opponent\'s component has {int(self.curr_conn_sq_opp)}; max_radius=3.')
            print(f'Trap search will start at {self.search_start_pos}.')

        # get maximum trap candidates to search per call to Expectiminimax
        # get maximum graph_cut candidates

        match self.turns:
            case 2:
            # if self.turns == 2:
                self.max_search_traps += 1
            case 3:
            # if self.turns == 3:
                self.max_search_traps += 1
            case 4:
            # if self.turns == 4:                     # graph_cut starts here - but why?
                self.max_search_traps = 9          # reset to 9
            case 5:
            # if self.turns == 5:
                self.max_search_traps += 1
            case 6:
            # if self.turns == 6:                     # 2-ply graph_cut starts here
                self.max_search_traps = 7           # reset to 7
                self.graph_cut_size_cap = 5         # starting lower for graph cut 2-ply
                self.max_radius = 2                 # starting lower for graph cut 2-ply, changed from 2
            case 7:
            # if self.turns == 7:
                # self.max_search_traps += 1
                self.graph_cut_size_cap += 1
            case 8:
            # if self.turns == 8:
                self.max_search_traps += 1
                # self.graph_cut_size_cap += 1
            case 9:
            # if self.turns == 9:
                self.max_search_traps += 1
                self.graph_cut_size_cap += 1
                self.max_radius += 1                # now to 4, changed from 3
            case 10:
            # if self.turns == 10:
                self.max_search_traps += 1
                self.graph_cut_size_cap += 1
            case 11:
            # if self.turns == 11:
                self.max_search_traps += 1
                self.graph_cut_size_cap += 1
            case 12:
            # if self.turns == 12:
                self.max_search_traps += 1
                self.graph_cut_size_cap += 1
            case 13:
            # if self.turns == 13:
                self.max_search_traps += 1
                self.graph_cut_size_cap += 1
            case 14:
            # if self.turns == 14:
                self.graph_cut_size_cap += 1
                self.max_radius += 1                # pretty overkill at this point

        self.max_trap_candidates = 0                        # initialize count
        if self.verbose:
            print(f'Max traps to search = {self.max_search_traps}; ', end='')

        alpha = -np.inf
        beta = np.inf
        max_move = self.__decision(grid, alpha, beta, player_num, opp_num, self.depth_limit)
        self.turns += 1
        self.current_move = max_move
        
        if self.verbose:
            print(f'Total iterative search evals = {self.heur_evals}, ', end='')
            print(f'child nodes visited = {self.child_nodes_seen}.')

        return max_move


    def __is_over(self, grid: Grid, player_num, player_pos, opp_num, opp_pos) -> int:
        """
        Check if game is over, i.e., Player or Opponent has no moves to make
        """
        # check if Player has won
        opp_neighbors = self.__get_valid_neighbors(grid, opp_pos)
        if len(opp_neighbors) == 0:
            return player_num

        # check if Opponent has won
        player_neighbors = self.__get_valid_neighbors(grid, player_pos)
        if len(player_neighbors) == 0:
            return opp_num


    def __evaluate(self, grid: Grid, gameover_result, player_num) -> int:
        """
        Function for returning high or low utility based on gameover state.
        """
        if gameover_result:
            if gameover_result == player_num:
                return grid, 99999
            else:
                return grid, -99999


    def __get_dof(self, grid: Grid, player_pos, opp_pos) -> tuple:
        """
        Apply degrees of freedom calculation heuristic based on early, mid, late game.
        Returns a tuple of Grid object, heuristic
        """
        center_heur = 0
        near_opp_heur = 0
        avoid_traps_heur = 0
        edge_touch_heur = 0
        n_neighbors_heur_me = 0
        n_neighbors_heur_opp = 0
        n_neighbors_heur = 0
        n_conn_sq_heur_me = 48*5
        conn_sq_list_me = []
        n_conn_sq_heur_opp = 48*5
        conn_sq_list_opp = []
        n_conn_sq_heur = 0
        conn_sq_depth_lim_me = 0
        conn_sq_depth_lim_opp = 0
        conn_sq_depth_lim_heur = 0
        graph_cut_heur_me = 0
        graph_cut_heur_opp = 0
        graph_cut_heur = 0

        if self.turns <= 3:
            center_heur = self.__center_heur(grid, player_pos) - self.__center_heur(grid, opp_pos)
            near_opp_heur = self.__near_opp_heur(grid, player_pos, opp_pos)
            edge_touch_heur = self.__edge_touch_heur(grid, player_pos) - self.__edge_touch_heur(grid, opp_pos)
            avoid_traps_heur = self.__avoid_traps_heur(grid, player_pos) - self.__avoid_traps_heur(grid, opp_pos)
        n_neighbors_heur_me = self.__n_neighbors_heur(grid, player_pos)
        n_neighbors_heur_opp = self.__n_neighbors_heur(grid, opp_pos)

        max_radius = self.max_radius
        if self.turns <= 3:
            conn_sq_depth_lim_me, conn_sq_list_me = self.__conn_sq_depth_lim_heur(grid, player_pos, max_radius=2, return_pos=True)
            conn_sq_depth_lim_opp, conn_sq_list_opp = self.__conn_sq_depth_lim_heur(grid, opp_pos, max_radius=2, return_pos=True)
        else:
            size_cap = self.graph_cut_size_cap
            # n_conn_sq_heur_me, conn_sq_list_me = self.__connected_sq_heur(grid, player_pos, max_size=connected_max_size, return_pos=True)
            # n_conn_sq_heur_opp, conn_sq_list_opp = self.__connected_sq_heur(grid, opp_pos, max_size=connected_max_size, return_pos=True)
            n_conn_sq_heur_me, conn_sq_list_me = self.__conn_sq_depth_lim_heur(grid, player_pos, max_radius=3, return_pos=True)
            n_conn_sq_heur_opp, conn_sq_list_opp = self.__conn_sq_depth_lim_heur(grid, opp_pos, max_radius=3, return_pos=True)
            max_comp_size_me = min(int(n_conn_sq_heur_me/5), size_cap)
            max_comp_size_opp = min(int(n_conn_sq_heur_opp/5), size_cap)

        # stepper func to determine if we use graph_cut algo based on turn # and # of connected squares
        turns_limit = (self.turns - 2*np.ceil(self.depth_limit/3) >= 0) # starts at 4 if depth_limit in (4,5,6)
        # turns_limit = (self.turns - 2*np.ceil(self.depth_limit/3) >= 1) # starts at 5 if depth_limit in (4,5,6)
        conn_sq_limit_me = (n_conn_sq_heur_me/5 + 3*(self.depth_limit//4) <= 45)
        conn_sq_limit_opp = (n_conn_sq_heur_opp/5 + 3*(self.depth_limit//4) <= 45)

        if turns_limit and conn_sq_limit_me and conn_sq_list_me:
            if self.use_graph_me == True:
                graph_cut_heur_me = self.__graph_cut_heur(grid, player_pos, \
                    comp_size=max_comp_size_me, conn_sq_list=conn_sq_list_me[:max_comp_size_me], max_radius=max_radius)

            if self.use_graph_opp == True:
                graph_cut_heur_opp = self.__graph_cut_heur(grid, player_pos, \
                    comp_size=max_comp_size_opp, conn_sq_list=conn_sq_list_opp[:max_comp_size_opp], max_radius=max_radius)

            if self.use_graph_d2_me == True:
                graph_cut_heur_me = self.__graph_cut_2_ply_heur(grid, player_pos, \
                    comp_size=max_comp_size_me, conn_sq_list=conn_sq_list_me[:max_comp_size_me], max_radius=max_radius)

            if self.use_graph_d2_opp == True:
                graph_cut_heur_opp = self.__graph_cut_2_ply_heur(grid, player_pos, \
                    comp_size=max_comp_size_opp, conn_sq_list=conn_sq_list_opp[:max_comp_size_opp], max_radius=max_radius)

        else:
            self.use_graph_me = False
            self.use_graph_opp = False
            self.use_graph_d2_me = False
            self.use_graph_d2_opp = False
        
        n_neighbors_heur = n_neighbors_heur_me - n_neighbors_heur_opp
        conn_sq_depth_lim_heur = conn_sq_depth_lim_me - conn_sq_depth_lim_opp
        n_conn_sq_heur = n_conn_sq_heur_me - n_conn_sq_heur_opp
        graph_cut_heur = graph_cut_heur_me - graph_cut_heur_opp

        return grid, avoid_traps_heur + center_heur + near_opp_heur + 10*conn_sq_depth_lim_heur + n_conn_sq_heur + n_neighbors_heur + edge_touch_heur + graph_cut_heur

    def __clone(self, grid: Grid) -> object:
        """
        Makes a full copy of current grid
        """
        grid_copy = Grid(7)
        grid_copy.map = grid.map.copy()
        return grid_copy


    def __move(self, grid: Grid, move_pos, player_num) -> object:
        """
        Faster move method
        """
        old_pos = np.where(grid.map == player_num)
        grid.map[old_pos], grid.map[move_pos] = grid.map[move_pos], grid.map[old_pos]

      
    def __edge_touch_heur(self, grid: Grid, pos) -> int:
        """
        Heuristic that penalizes being on an edge (more if on two edges, i.e. a corner)
        Returns a heuristic int
        """
        edges_touched = 0
        if pos[0] in (0,6):
            edges_touched += 1
        if pos[1] in (0,6):
            edges_touched += 1
        return -5*edges_touched # was -10*edges_touched


    def __center_heur(self, grid: Grid, pos) -> int:
        """
        Heuristic that slightly priotizes being closer to center
        Returns a heuristic int
        """
        dist = 6 - grid_distance(pos, (3,3))
        return 0.25*dist # was 1*dist


    def __near_opp_heur(self, grid: Grid, player_pos, opp_pos) -> int:
        """
        Heuristic that prioritizes being about 3 grid_distances away from opponent
        Returns a heuristic int
        """
        dist = abs(3 - grid_distance(player_pos, opp_pos))
        return -0.5*dist # was -5*dist


    def __avoid_traps_heur(self, grid: Grid, pos) -> int:
        """
        Heuristic that penalizes being next to a trap
        Returns a heuristic int
        """
        neighbors = self.__get_all_neighbors(grid, pos)
        num_traps = len([neighbor for neighbor in neighbors if grid.map[neighbor] == -1])
        return -3*num_traps # was -3*num_traps


    def __conn_sq_depth_lim_heur(self, grid: Grid, pos, max_radius=2, max_size=0, return_pos=False) -> int:
        """
        Get number of connected squares for current player in current Grid object.
        This is a depth-limited BFS algo, sort of an IDS
        Returns a heuristic int
        """
        if self.verbose:
            self.heur_evals += 1

        if max_radius==-1:
            max_radius = None
        radius = 1

        explored = [pos]                                    # initialize explored list with position # don't need
        search_frontier = self.__get_valid_neighbors(grid, pos, radius=1)
        conn_sq_list = [pos]                                # initialize connected squares list
        conn_sq_heur = 1                                    # initialize count
        current_layer = search_frontier.copy()              # initialize current layer
        next_layer = []                                     # initialize next layer
        new_neighbors = []                                  # initialize new neighbors

        while search_frontier and radius <= max_radius:
            for pos in current_layer:
                if pos not in conn_sq_list:
                    conn_sq_list.append(pos)                # add to connected square
                    conn_sq_heur += 1
                new_neighbors = self.__get_valid_neighbors(grid, pos, radius=1)
                next_layer = list(set(next_layer + new_neighbors))

            explored = list(set(explored + current_layer))
            search_frontier = list(set(search_frontier + next_layer) - set(explored))
            current_layer = search_frontier.copy()
            next_layer = []
            radius += 1

        if max_size:
            conn_sq_list = conn_sq_list[:max_size]
            conn_sq_heur = max_size

        if return_pos:
            return 5*conn_sq_heur, conn_sq_list # was 5*conn_sq_heur
        else:
            return 5*conn_sq_heur # was 5*conn_sq_heur


    def __graph_cut_heur(self, grid: Grid, pos, comp_size, conn_sq_list, max_radius=3) -> int:
        """
        Heuristic based on how drastic is the decrease in degrees of freedom resulting from the removal of one free space.
        Originally also computed number of connected components, but removed for performance.
        Returns a heuristic int
        """
        grid_clone = self.__clone(grid)
        graph_cut_heur = 0
        for i, trap_pos in enumerate(conn_sq_list):                                  # iterate over connected squares
            grid_clone.map[trap_pos] = -1
            # new_comp_size = self.__connected_sq_heur(grid_clone, pos, max_size=comp_size, return_pos=False)
            new_comp_size = self.__conn_sq_depth_lim_heur(grid_clone, pos, max_radius=max_radius, max_size=comp_size, return_pos=False)
            grid_clone.map[trap_pos] = 0
            size_delta = new_comp_size - comp_size
            utility = 10*size_delta - (comp_size-new_comp_size)**2 - 5*i
            graph_cut_heur += utility
        return graph_cut_heur


    def __graph_cut_2_ply_heur(self, grid: Grid, pos, comp_size, conn_sq_list, max_radius=3) -> int:
        """
        Same as __graph_cut_heur(), but going 2 levels deep instead of 1.
        Returns a heuristic int
        """
        def __gch_d2(grid: Grid, trap_pos, comp_size) -> int:
            grid.map[trap_pos] = -1
            # new_comp_size = self.__connected_sq_heur(grid, pos, max_size=comp_size, return_pos=False)
            new_comp_size = self.__conn_sq_depth_lim_heur(grid, pos, max_radius=max_radius, max_size=comp_size, return_pos=False)
            grid.map[trap_pos] = 0
            size_delta = new_comp_size - comp_size
            utility = 10*size_delta - (comp_size-new_comp_size)**2 -5*j

            return utility

        graph_cut_heur = 0
        grid_clone = self.__clone(grid)
        for i, trap_pos in enumerate(conn_sq_list):                                   # iterate over connected squares
            conn_sq_list_2 = conn_sq_list.copy()
            conn_sq_list_2.remove(trap_pos)
            grid_clone.map[trap_pos] = -1
            for j, trap_pos_2 in enumerate(conn_sq_list_2):
                graph_cut_heur += __gch_d2(grid_clone, trap_pos_2, comp_size) -5*i
            grid_clone.map[trap_pos] = 0

        return graph_cut_heur


    def __get_heuristics(self, grid: Grid, player_num, opp_num) -> tuple:
        """
        Apply heuristics
        Returns a tuple of Grid object, heuristic
        """
        player_pos = tuple(np.argwhere(grid.map == player_num)[0])
        opp_pos = tuple(np.argwhere(grid.map == opp_num)[0])
        if self.use_advanced_heuristics == 'graphcut':
            if self.verbose:
                start = time.time()
                dof_heur = self.__get_dof(grid, player_pos=player_pos, opp_pos=opp_pos)
                end = time.time()
                self.heur_time += end-start
            else:
                dof_heur = self.__get_dof(grid, player_pos=player_pos, opp_pos=opp_pos)
            return dof_heur
        elif self.use_advanced_heuristics == 'geodesics':
            if self.verbose:
                start = time.time()
                conn_sq_dl_me = self.__connected_sq_heur(grid, player_pos, max_size=25)
                conn_sq_dl_opp = self.__connected_sq_heur(grid, opp_pos, max_size=25)
                dl_heur = conn_sq_dl_me-conn_sq_dl_opp
                end = time.time()
                self.heur_time += end-start
            else:
                conn_sq_dl_me = self.__connected_sq_heur(grid, player_pos, max_size=25)
                conn_sq_dl_opp = self.__connected_sq_heur(grid, opp_pos, max_size=25)
                dl_heur = conn_sq_dl_me-conn_sq_dl_opp
            return grid, dl_heur                    # for next set of heuristics Matthew/Gong
        else:
            return grid, self.__n_neighbors_heur(grid, player_pos) - self.__n_neighbors_heur(grid, opp_pos)


    def __n_neighbors_heur(self, grid: Grid, pos) -> int:
        """
        Returns the difference in available squares around player vs available squares around opponent.
        Now this just returns the same as __get_valid_neighbors; deprecated.
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
        Modification of original function for returning neighboring cells.
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


    def __find_center(self, player_pos, opp_pos):
        inc_ceil = lambda i,j: int(np.ceil( (i + j) / 2 ))
        inc_floor = lambda i,j: int(np.floor( (i + j) / 2 ))
        if player_pos[0] <= opp_pos[0]:
            x = inc_ceil(player_pos[0], opp_pos[0])
        else:
            x = inc_floor(player_pos[0], opp_pos[0])

        if player_pos[1] <= opp_pos[1]:
            y = inc_ceil(player_pos[1], opp_pos[1])
        else:
            y = inc_floor(player_pos[1], opp_pos[1])
        return x, y


    def __get_search_start_pos(self, grid: Grid, player_pos, opp_pos) -> tuple:
        '''
        Start search midway between players, but not on a trap
        Iteratively get closer to the opponent, can be opponent
        '''
        start_pos = self.__find_center(player_pos, opp_pos)
        while grid.map[start_pos] == -1 or start_pos not in self.curr_conn_sq_list_opp:
            start_pos = self.__find_center(start_pos, opp_pos)
        return start_pos


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
        search_frontier = [start_pos]                       # initialize with position
        explored = []                                       # initialize with position
        trap_candidates = []                                # initialize list of trap candidates to return
        trap_count = 0                                      # initialize count of traps
        new_neighbors = [start_pos]

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
            if trap_count >= self.max_search_traps:
                self.max_trap_candidates = trap_count
                return trap_candidates
            while not new_neighbors:
                if self.turns <= 4:                         # strategy optimization
                    new_neighbors = self.__get_valid_neighbors_avoid_edge(grid, start_pos, radius)
                if not new_neighbors:                       # 
                    new_neighbors = self.__get_valid_neighbors(grid, start_pos, radius)
                radius += 1
            search_frontier = list(set(search_frontier + new_neighbors) - set(explored))
            # no need to intersect with trap_candidates, because it is a subset of explored

        if trap_count == 0:
            if self.verbose:
                print(f'No trap positions left after visiting {self.child_nodes_seen} child nodes.')
            catch_trap = self.__get_valid_neighbors_avoid_edge(grid, start_pos, radius)
            if not catch_trap:
                catch_trap = self.__get_valid_neighbors(grid, start_pos, radius)
            if not catch_trap:
                catch_trap = grid.getAvailableCells()[0]
            return catch_trap

        if trap_count > self.max_trap_candidates:
            self.max_trap_candidates = trap_count

        return trap_candidates


    def __trap_minimize(self, grid: Grid, alpha, beta, player_num, opp_num, depth, depth_limit) -> tuple:
        """
        opponent Min node for throwing Trap
        returns a tuple of pos, utility
        """
        if self.verbose:
            self.child_nodes_seen += 1

        player_pos = tuple(np.argwhere(grid.map == player_num)[0])
        opp_pos = tuple(np.argwhere(grid.map == opp_num)[0])            # this is the position in question
        gameover_result = self.__is_over(grid, player_num, player_pos, opp_num, opp_pos)
        if gameover_result:
            return self.__evaluate(grid, gameover_result, player_num)

        # break if hit depth limit
        # if depth >= depth_limit or self.child_nodes_seen >= self.max_child_nodes:
        if depth >= depth_limit:
            _, utility = self.__get_heuristics(grid, player_num, opp_num)
            return _, utility

        minTrap, minUtility = None, np.inf
        cache = {}
        player_pos_a = np.array(player_pos)
        opp_pos_a = np.array(opp_pos)
        x_0, x_1 = player_pos_a - opp_pos_a
        center_a = np.array(self.__find_center(player_pos, opp_pos))
        # trap_pos is a list of e.g. [(2,3), (3,4)]
        for trap_pos in self.__get_trap_candidates(grid, player_pos):
            # initialize with the main trap's probability-weighted utility, then move on to those of the neighbors
            grid.map[trap_pos] = -1
            key = tuple(map(tuple, grid.map))
            if key in cache:
                utility = cache[key]
            else:
                _, utility = self.__move_maximize(grid, alpha, beta, player_num, opp_num, depth+1, depth_limit)
                # STRATEGY: prioritize certrain trap locations:
                # should be closer to midpoint between two players
                # prefer lateral movements along the minor axis, helps to wall off opponent (better than diagonal placements)
                # prefer trap locations closer to opponent than to player 
                trap_pos_a = np.array(trap_pos)
                dist_me = abs(player_pos_a - trap_pos_a)
                dist_opp = 2.1*abs(opp_pos_a - trap_pos_a)
                dist_mid = abs(player_pos_a - opp_pos_a) - 2*abs(center_a-trap_pos_a)    # heuristic tested on paper
                if x_0 > x_1:
                    trap_heur = dist_me + dist_opp + 4*dist_mid[0] + 2*dist_mid[1]
                elif x_0 < x_1:
                    trap_heur = dist_me + dist_opp + 2*dist_mid[0] + 4*dist_mid[1]
                else:
                    trap_heur = dist_me + dist_opp + 2*dist_mid[0] + 2*dist_mid[1]
                # cache[key] = utility
                cache[key] = utility + sum(trap_heur)
            target_prob = self.__probability(opp_pos, trap_pos)
            expected_utility = target_prob * utility
            # backtrack
            grid.map[trap_pos] = 0

            neighbors = self.__get_all_neighbors(grid, trap_pos)
            for neighbor in neighbors:
                if grid.map[neighbor] in (0, -1):             # to avoid player and opponent location
                    temp = grid.map[neighbor]
                    grid.map[neighbor] = -1
                    key = tuple(map(tuple, grid.map))
                    if key in cache:
                        utility = cache[key]
                    else:
                        # _, utility = self.__move_maximize(grid, alpha, beta, player_num, opp_num, depth + 1, depth_limit)
                        _, utility = self.__get_heuristics(grid, player_num, opp_num)
                        cache[key] = utility
                    grid.map[neighbor] = temp
                    expected_utility += (1 - target_prob) / len(neighbors) * utility

            if expected_utility < minUtility:
                minTrap, minUtility = trap_pos, expected_utility
        return minTrap, minUtility


    def __move_minimize(self, grid: Grid, alpha, beta, player_num, opp_num, depth, depth_limit) -> tuple:
        """
        opponent Min node for making Move
        returns a tuple of pos, utility
        """
        if self.verbose:
            self.child_nodes_seen += 1

        player_pos = tuple(np.argwhere(grid.map == player_num)[0])
        opp_pos = tuple(np.argwhere(grid.map == opp_num)[0])
        gameover_result = self.__is_over(grid, player_num, player_pos, opp_num, opp_pos)
        if gameover_result:
            return self.__evaluate(grid, gameover_result, player_num)

        # break if hit depth limit
        # if depth >= depth_limit or self.child_nodes_seen >= self.max_child_nodes:
        if depth >= depth_limit:
            _, utility = self.__get_heuristics(grid, player_num, opp_num)
            return _, utility

        minChild, minUtility = None, np.inf
        for neighbor_to_move in self.__get_valid_neighbors(grid, opp_pos):
            grid.map[opp_pos] = 0
            grid.map[neighbor_to_move] = opp_num
            _, utility = self.__trap_minimize(grid, alpha, beta, player_num, opp_num, depth + 1, depth_limit)
            grid.map[opp_pos] = opp_num
            grid.map[neighbor_to_move] = 0
            if utility < minUtility:
                minChild, minUtility = neighbor_to_move, utility

            if minUtility <= alpha:
                break

            if minUtility < beta:
                beta = minUtility

        return minChild, minUtility


    def __trap_maximize(self, grid: Grid, alpha, beta, player_num, opp_num, depth, depth_limit) -> tuple:
        """
        player Max node for throwing Trap
        returns a tuple of pos, utility
        """
        if self.verbose:
            self.child_nodes_seen += 1

        player_pos = tuple(np.argwhere(grid.map == player_num)[0])              # this is the position in question
        opp_pos = tuple(np.argwhere(grid.map == opp_num)[0])
        gameover_result = self.__is_over(grid, player_num, player_pos, opp_num, opp_pos)
        if gameover_result:
            return self.__evaluate(grid, gameover_result, player_num)

        # break if (exceed) hit depth limit
        # if depth >= depth_limit or self.child_nodes_seen >= self.max_child_nodes:
        if depth > depth_limit:
            _, utility = self.__get_heuristics(grid, player_num, opp_num)
            return _, utility

        maxTrap, maxUtility = None, -np.inf
        cache = {}
        player_pos_a = np.array(player_pos)
        opp_pos_a = np.array(opp_pos)
        x_0, x_1 = player_pos_a - opp_pos_a
        center_a = np.array(self.__find_center(player_pos, opp_pos))
        # player_pos = self.getPlayerPosition(grid)
        # opp_pos = self.getOpponentPosition(grid)
        for trap_pos in self.__get_trap_candidates(grid, opp_pos):
            # initialize with the main trap's probability-weighted utility, then move on to those of the neighbors
            grid.map[trap_pos] = -1
            key = tuple(map(tuple, grid.map))
            if key in cache:
                utility = cache[key]
            else:
                _, utility = self.__move_minimize(grid, alpha, beta, player_num, opp_num, depth + 1, depth_limit)
                # STRATEGY: prioritize certrain trap locations:
                # should be closer to midpoint between two players
                # prefer lateral movements along the minor axis, helps to wall off opponent (better than diagonal placements)
                # prefer trap locations closer to opponent than to player 
                trap_pos_a = np.array(trap_pos)
                dist_me = 2.1*abs(player_pos_a - trap_pos_a)
                dist_opp = abs(opp_pos_a - trap_pos_a)
                dist_mid = abs(player_pos_a - opp_pos_a) - 2*abs(center_a - trap_pos_a)    # heuristic tested on paper
                if x_0 > x_1:
                    trap_heur = dist_me + dist_opp + 4*dist_mid[0] + 2*dist_mid[1]
                elif x_0 < x_1:
                    trap_heur = dist_me + dist_opp + 2*dist_mid[0] + 4*dist_mid[1]
                else:
                    trap_heur = dist_me + dist_opp + 2*dist_mid[0] + 2*dist_mid[1]
                # cache[key] = utility
                cache[key] = utility + sum(trap_heur)
            target_prob = self.__probability(player_pos, trap_pos)
            expected_utility = target_prob * utility
            grid.map[trap_pos] = 0

            neighbors = self.__get_all_neighbors(grid, trap_pos)
            for neighbor in neighbors:
                if grid.map[neighbor] in (0, -1):             # to avoid player and opponent location
                    temp = grid.map[neighbor]
                    grid.map[neighbor] = -1
                    key = tuple(map(tuple, grid.map))
                    if key in cache:
                        utility = cache[key]
                    else:
                        # _, utility = self.__move_minimize(grid, alpha, beta, player_num, opp_num, depth + 1, depth_limit)
                        _, utility = self.__get_heuristics(grid, player_num, opp_num)
                        cache[key] = utility
                    grid.map[neighbor] = temp
                    expected_utility += (1 - target_prob) / len(neighbors) * utility

            if expected_utility > maxUtility:
                maxTrap, maxUtility = trap_pos, expected_utility
        # returns max trap so maximize can cache it
        return maxTrap, maxUtility


    def __move_maximize(self, grid: Grid, alpha, beta, player_num, opp_num, depth, depth_limit) -> tuple:
        """
        player Max node for making Move
        returns a tuple of pos, utility
        """
        if self.verbose:
            self.child_nodes_seen += 1
        player_pos = tuple(np.argwhere(grid.map == player_num)[0])
        opp_pos = tuple(np.argwhere(grid.map == opp_num)[0])
        gameover_result = self.__is_over(grid, player_num, player_pos, opp_num, opp_pos)
        if gameover_result:
            return self.__evaluate(grid, gameover_result, player_num)

        # break if (exceed) hit depth limit
        # if depth >= depth_limit or self.child_nodes_seen >= self.max_child_nodes:
        if depth > depth_limit:
            _, utility = self.__get_heuristics(grid, player_num, opp_num)
            return _, utility

        maxMove, maxTrap, maxUtility = None, None, -np.inf
        for neighbor_to_move in self.__get_valid_neighbors(grid, player_pos):
            grid.map[player_pos] = 0
            grid.map[neighbor_to_move] = player_num
            trap, utility = self.__trap_maximize(grid, alpha, beta, player_num, opp_num, depth + 1, depth_limit)
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


    def __decision(self, grid: Grid, alpha, beta, player_num, opp_num, depth_limit=DEFAULT_DEPTH_LIMIT) -> object:
        """
        Helper function to start the Expectiminimax algo
        returns a Grid object
        """
        if self.verbose:
            # start = time.time()
            child, self.utility = self.__move_maximize(grid, alpha, beta, player_num, opp_num, depth=0, depth_limit=depth_limit)
            if self.utility >= 90000:
                print(f'Best move found has utility of {self.utility:.2f}. Win imminent? 😱')
            elif self.utility >= 1000:
                print(f'Best move found has utility of {self.utility:.2f}. Doing decent! 😲')
            elif self.utility <= -1000:
                print(f'Best move found has utility of {self.utility:.2f}. Not looking so good. 😟')
            elif self.utility <= -90000:
                print(f'Best move found has utility of {self.utility:.2f}, omg no... 😵‍💫')
            else:
                print(f'Best move found has utility of {self.utility:.2f}. Hey, doing what we can. 😐')
            # end = time.time()
            # print(f'This move took {end-start:.5f} seconds.')
        else:
            child, _ = self.__move_maximize(grid, alpha, beta, player_num, opp_num, depth=0, depth_limit=depth_limit)
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
        if not self.__get_valid_neighbors(grid, self.getOpponentPosition(grid)):
            return grid.getAvailableCells()[0]

        # use cached optimal trap position that we computed in getMove()
        if not self.optimal_trap_pos:
            print('No trap positions') if self.verbose else None
            return grid.getAvailableCells()[0]
        if self.optimal_trap_pos == self.current_move:
            # if game ends because we took the last spot
            print('We\'ve taken up the last spot. Throwing trap randomly because we\'ll win anyway.')
            return grid.getAvailableCells()[0]
        return self.optimal_trap_pos

