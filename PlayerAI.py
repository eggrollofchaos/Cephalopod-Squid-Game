"""
PlayerAI Class module.
Primarily authored by WAX.
Early contributions by @mhr and @gongchen161.
Hard-coded to a 7x7 grid size.
"""
import numpy as np
import random
import time
import sys
import os
import queue as Q
from statistics import fmean
from BaseAI import BaseAI
from Grid import Grid
from Utils import manhattan_distance, grid_distance
from termcolor import cprint

DEFAULT_DEPTH_LIMIT = 4

class PlayerAI(BaseAI):
    # def __init__(self, depth_limit = DEFAULT_DEPTH_LIMIT, heur = 'graphcut', verbose = False) -> None:
    def __init__(self, depth_limit = DEFAULT_DEPTH_LIMIT, heur = None, verbose = 0) -> None:
        '''
        Player AI.
        Uses Expectiminimax.
        Applies various heuristics.
        Note that Expectiminimax needs a depth of 4 in order to go through one full cycle:
            Move Maximize -> Trap Maximize -> Move Minimize -> Trap Minimize
        Set DEFAULT_DEPTH_LIMIT = 4.
        This means that finding the best move happens concurrently with finding the best trap.
        Set starting max_search_traps to minimum of 21, boosted significantly from 8.
        '''

        super().__init__()
        self.verbose = verbose
        self.pos = None
        self.opp_pos = None
        self.player_num = None
        self.optimal_trap_pos = None
        self.depth_limit = max(depth_limit, 1)  # Expectiminimax needs at least a search depth of 1, but really 2
        # print(f'Depth limit arg is {depth_limit}, setting depth_limit to {self.depth_limit}.')
        # if self.depth_limit < 1:
        #     self.depth_limit = DEFAULT_DEPTH_LIMIT
        self.turns = 1                          # early game = 1-3, mid = 4-6, late to 7+; generally early game <= grid.dim/2, mid = 2xearly
        self.use_advanced_heuristics = heur     # graphcut, geodesics, None
        # self.curr_conn_sq = 48                # essentially dim^2 - 1
        self.search_start_pos_me = (3, 3)
        self.search_start_pos_opp = (3, 3)
        self.graph_cut_size_cap = 8

        self.max_search_traps = 21
        # max_search_traps starts at 18, but needs to be adjusted down at early game if depth_level is higher than 5
        self.max_search_traps += min(2*(5 - self.depth_limit), 0)
        self.max_search_traps = max(8, abs(self.max_search_traps))      # no less than 8 to begin with

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

    def getPlayerNum(self):                     # using this to be rigorous
        return self.player_num

    def getPlayerPosition(self, grid):          # of current Grid object - CHECK when using?
        return grid.find(self.getPlayerNum())

    def getOpponentNum(self):                   # using this to be rigorous
        return 3 - self.getPlayerNum()

    def getOpponentPosition(self, grid: Grid):  # of current Grid object - CHECK when using?
        return grid.find(self.getOpponentNum())

    def getMove(self, grid: Grid) -> tuple:
        '''
        Entry point for starting a move.

        The function should return a tuple of (x,y) coordinates to which the Player moves.

        It should be the result of the ExpectiMinimax algorithm, maximizing over the Opponent's *Trap* actions, 
        taking into account the probabilities of them landing in the positions you believe they'd throw to.

        Note that you are not required to account for the probabilities of it landing in a different cell.

        You may adjust the input variables as you wish (though it is not necessary). Output has to be (x,y) coordinates.
        
        '''
        # if self.turns > 1:
        #     exit(1)

        # get player numbers variables
        player_num = self.getPlayerNum()
        opp_num = self.getOpponentNum()
        # self.opp_pos = tuple(np.argwhere(grid.map == opp_num)[0])              # find opponent's current position
        self.opp_pos = grid.find(3 - self.player_num)                            # find opponent's current position using method

        if self.verbose:
            self.heur_evals = 0
            self.heur_time = 0
            self.child_nodes_seen = 0
            # self.printed = False
            self.utility = 0
            self.current_depth = 0

            # debug
            self.trap_min_utility = np.inf
            self.move_min_utility = np.inf
            self.trap_max_utility = -np.inf
            self.move_max_utility = -np.inf

            print(f'Player is at {self.pos}, Opponent is at {self.opp_pos}, they are currently {grid_distance(self.pos, self.opp_pos)} units apart. ', end='')
            print(f'This is a heuristic of {PlayerAI.__near_opp_heur(grid, self.pos, self.opp_pos)}.')

            print(f'There are {PlayerAI.__avoid_traps_heur(grid, self.pos)[0]} trap(s) next to Player and {PlayerAI.__avoid_traps_heur(grid, self.opp_pos)[0]} trap(s) next to Opponent.')


        print(f'Looking for best move. Depth limit is {self.depth_limit}.') if self.verbose else None

        # get both players' current connected sq 
        # curr_conn_sq_me, curr_conn_sq_list_me = self.__connected_sq_heur(grid, pos=self.pos, max_size=27, return_pos=True)        # previous DFS iteration of this function
        # curr_conn_sq_opp, curr_conn_sq_list_opp = self.__connected_sq_heur(grid, pos=self.opp_pos, max_size=27, return_pos=True)  # previous DFS iteration of this function
        curr_conn_sq_heur_me, curr_conn_sq_me, curr_conn_sq_list_me = self.__conn_sq_depth_lim_heur(grid, pos=self.pos, max_radius=3, only_heur=False)
        curr_conn_sq_heur_opp, curr_conn_sq_opp, curr_conn_sq_list_opp = self.__conn_sq_depth_lim_heur(grid, pos=self.opp_pos, max_radius=3, only_heur=False)
        self.curr_conn_sq_heur_me = curr_conn_sq_heur_me
        self.curr_conn_sq_heur_opp = curr_conn_sq_heur_opp
        self.curr_conn_sq_me = curr_conn_sq_me
        self.curr_conn_sq_opp = curr_conn_sq_opp
        self.curr_conn_sq_list_me = curr_conn_sq_list_me
        self.curr_conn_sq_list_opp = curr_conn_sq_list_opp
        self.search_start_pos_me = self.__get_search_start_pos(grid, self.curr_conn_sq_list_opp, self.pos, self.opp_pos)       # get start square for Player trap search
        self.search_start_pos_opp = self.__get_search_start_pos(grid, self.curr_conn_sq_list_me, self.opp_pos, self.pos)       # get start square for Opponent trap search
        if self.verbose:
            if self.curr_conn_sq_me >= 28 or self.curr_conn_sq_opp >= 28:
                print(f'Player\'s and opponent\'s component both have a lot of connected squares.')
            else:
                print(f'Player\'s component has {int(self.curr_conn_sq_me)} connected squares, opponent\'s component has {int(self.curr_conn_sq_opp)}; max_radius=3.')
            print(f'Trap search will start at {self.search_start_pos_me}.', end=' ')
        
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
                # self.max_search_traps = 11          # reset to 11, changed from 9; re,oved
                pass
            case 5:
            # if self.turns == 5:
                self.max_search_traps += 1
            case 6:
            # if self.turns == 6:                     # 2-ply graph_cut starts here
                # self.max_search_traps = 9           # reset to 9, changed from 7
                self.graph_cut_size_cap = 8         # starting lower for graph cut 2-ply, changed from 6
                self.max_radius = 3                 # starting lower for graph cut 2-ply, changed from 2
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

        # graph_cut initialization
        # setting these to true for now, but will not necessarily be running these
        if self.use_advanced_heuristics == 'graphcut':
            self.use_graph_me = True
            self.use_graph_opp = True

            # start using 2-py graph cut
            # testing always using 1-ply
            if self.turns >= 6:
                self.use_graph_me = False
                self.use_graph_d2_me = True
                self.use_graph_opp = False
                self.use_graph_d2_opp = True

        self.max_trap_candidates = 0                        # initialize count
        if self.verbose:
            print(f'Max traps to search = {self.max_search_traps}... ')

        alpha = -np.inf
        beta = np.inf
        max_move = self.__decision(grid, alpha, beta, player_num, opp_num, self.depth_limit)
        self.turns += 1
        self.current_move = max_move
        
        if self.verbose:
            print(f'Total iterative search evals = {self.heur_evals}, ', end='')
            print(f'child nodes visited = {self.child_nodes_seen}.')

        return max_move


    @staticmethod
    def __clone(grid: Grid) -> object:
        '''
        Makes a full copy of current grid
        '''
        grid_copy = Grid(7)
        grid_copy.map = grid.map.copy()
        return grid_copy


    @staticmethod
    def __evaluate(grid: Grid, gameover_result, player_num) -> tuple:
        '''
        Function for returning high or low utility based on gameover state.
        Adjusted from 99999 and -99999
        '''
        if gameover_result:
            if gameover_result == player_num:
                return grid, 999999
            else:
                return grid, -999999


    @staticmethod
    def __find_center(pos1, pos2):
        '''
        Helper function for finding a square roughly midway between two players.
        When there are similar number of closeness, this will find squares CLOSER to pos2.
        
        '''
        inc_ceil = lambda i,j: int(np.ceil( (i + j) / 2 ))
        inc_floor = lambda i,j: int(np.floor( (i + j) / 2 ))
        if pos1[0] <= pos2[0]:
            x = inc_ceil(pos1[0], pos2[0])
        else:
            x = inc_floor(pos1[0], pos2[0])

        if pos1[1] <= pos2[1]:
            y = inc_ceil(pos1[1], pos2[1])
        else:
            y = inc_floor(pos1[1], pos2[1])
        return x, y


    @staticmethod
    def __get_all_neighbors(pos, radius=1) -> list:
        '''
        Same as __get_valid_neighbors, but includes trap positions.
        Can specify radius, default is 1.
        Updated to not return current position.
        Returns a list of positions as tuples.
        '''
        x,y = pos
        valid_range = lambda t: range(max(t-radius, 0), min(t+radius+1, 7))
        return list({(a,b) for a in valid_range(x) for b in valid_range(y)} - {(x,y)})


    # TODO: not currently used
    @staticmethod
    def __get_all_neighbors_avoid_edge(pos, radius=1) -> list:
        '''
        Same as __get_all_neighbors, but avoiding edges.
        Can specify radius, default is 1.
        Updated to not return current position.
        Returns a list of positions as tuples.
        '''
        x,y = pos
        valid_range = lambda t: range(max(t-radius, 1), min(t+radius+1, 6))
        return list({(a,b) for a in valid_range(x) for b in valid_range(y)} - {(x,y)})


    @staticmethod
    def __get_valid_neighbors(grid: Grid, pos, radius=1) -> list:
        '''
        Modification of function grid.get_neighbors() for returning neighboring cells, excluding traps.
        Can specify radius, default is 1.
        Includes current position.
        Returns a list of positions as tuples.
        '''
        x,y = pos
        valid_range = lambda t: range(max(t-radius, 0), min(t+radius+1, 7))
        return [(a,b) for a in valid_range(x) for b in valid_range(y) if grid.map[(a,b)] == 0]


    @staticmethod
    def __get_valid_neighbors_avoid_edge(grid: Grid, pos, radius=1) -> list:
        '''
        Same as __get_valid_neighbors, but avoiding edges.
        Can specify radius, default is 1.
        Includes current position.
        Returns a list of positions as tuples.
        '''
        x,y = pos
        valid_range = lambda t: range(max(t-radius, 1), min(t+radius+1, 6))
        return [(a,b) for a in valid_range(x) for b in valid_range(y) if grid.map[(a,b)] == 0]


    @staticmethod
    def __is_over(grid: Grid, player_num, player_pos, opp_num, opp_pos) -> int:
        '''
        Check if game is over, i.e., Player or Opponent has no moves to make
        '''
        # check if Player has won
        opp_neighbors = PlayerAI.__get_valid_neighbors(grid, opp_pos)
        if len(opp_neighbors) == 0:
            return player_num

        # check if Opponent has won
        player_neighbors = PlayerAI.__get_valid_neighbors(grid, player_pos)
        if len(player_neighbors) == 0:
            return opp_num


    # TODO: this @staticmethod exists in for other AIs, which is duplicative;
    # possibly move it to a new, shared AI_utils module
    @staticmethod
    def __probability(pos, trap_pos) -> float:
        '''
        Calculates probability of a trap landing in an intended square.
        '''
        alpha = manhattan_distance(pos, trap_pos)
        # alpha = grid_distance(pos, trap_pos)
        p = 1 - 0.05 * (alpha - 1)
        return p


######################################################
################# STATIC HEURISTICS ##################
######################################################


    @staticmethod
    def __avoid_traps_heur(grid: Grid, pos) -> int:
        '''
        Heuristic that penalizes being next to a trap.
        Returns a heuristic int.
        '''
        neighbors = PlayerAI.__get_all_neighbors(pos)
        num_traps = len([neighbor for neighbor in neighbors if grid.map[neighbor] == -1])
        return num_traps, -3*num_traps      # added trap explict return


    @staticmethod
    def __center_heur(grid: Grid, pos) -> int:
        '''
        Heuristic that slightly prioritizes being closer to center.
        Returns a heuristic int.
        '''
        dist = 6 - grid_distance(pos, (3,3))
        return 0.25*dist # was 1*dist


    @staticmethod
    def __edge_touch_heur(grid: Grid, pos) -> int:
        '''
        Heuristic that penalizes being on an edge (more if on two edges, i.e. a corner)
        Returns a heuristic int
        '''
        edges_touched = 0
        if pos[0] in (0,6):
            edges_touched += 1
        if pos[1] in (0,6):
            edges_touched += 1
        return -5*edges_touched # was -10*edges_touched


    @staticmethod
    def __near_opp_heur(grid: Grid, player_pos, opp_pos) -> int:
        '''
        Heuristic that prioritizes being about 3 grid_distances away from opponent, changed from 4
        Returns a heuristic int
        '''
        dist = abs(4 - grid_distance(player_pos, opp_pos))
        return -0.5*dist # was -5*dist


    @staticmethod
    def __n_neighbors_heur(grid: Grid, pos) -> int:
        '''
        Returns the difference in available squares around player vs available squares around opponent.
        Now this just returns the same as __get_valid_neighbors; deprecated.
        '''
        return len(PlayerAI.__get_valid_neighbors(grid, pos))


######################################################
################ INSTANCE HEURISTICS #################
######################################################


    def __get_dof(self, grid: Grid, player_pos, opp_pos) -> tuple:
        '''
        Apply degrees of freedom calculation heuristic based on early, mid, late game.
        Called by each leaf (end node) of Expectiminimax via get_heuristics().
        Returns a tuple of Grid object, heuristic.
        '''
        center_heur = 0
        near_opp_heur = 0
        avoid_traps_heur = 0
        edge_touch_heur = 0
        n_neighbors_heur_me = 0
        n_neighbors_heur_opp = 0
        n_neighbors_heur = 0
        n_conn_sq_heur_me = 48*5                    # initializing as dim^2 * 5
        n_conn_sq_heur_opp = 48*5                   # initializing as dim^2 * 5
        n_conn_sq_heur = 0
        # conn_sq_depth_lim_me = 0                  # redundant
        # conn_sq_depth_lim_opp = 0                 # redundant
        # conn_sq_depth_lim_heur = 0                # redundant

        # if self.use_advanced_heuristics == 'graphcut':
        conn_sq_list_me = []                        # only used when graphcut is enabled
        conn_sq_list_opp = []
        graph_cut_heur_me = 0
        graph_cut_heur_opp = 0
        graph_cut_heur = 0

        # simple heuristics
        # if self.turns <= 15:                            # changed from 3, or 15, should be always
        _, avoid_traps_heur_me = PlayerAI.__avoid_traps_heur(grid, player_pos)
        _, avoid_traps_heur_opp = PlayerAI.__avoid_traps_heur(grid, opp_pos)
        center_heur = PlayerAI.__center_heur(grid, player_pos) - PlayerAI.__center_heur(grid, opp_pos)
        avoid_traps_heur = avoid_traps_heur_me - avoid_traps_heur_opp
        edge_touch_heur = PlayerAI.__edge_touch_heur(grid, player_pos) - PlayerAI.__edge_touch_heur(grid, opp_pos)
        near_opp_heur = PlayerAI.__near_opp_heur(grid, player_pos, opp_pos)
        n_neighbors_heur_me = PlayerAI.__n_neighbors_heur(grid, player_pos)
        n_neighbors_heur_opp = PlayerAI.__n_neighbors_heur(grid, opp_pos)
        n_neighbors_heur = n_neighbors_heur_me - n_neighbors_heur_opp

        # move max radius up to the top TODO?
        max_radius = self.max_radius

        # advanced heuristic not allowed until turn 4?
        # if self.turns <= 3:

        # connected squares (component) heuristics
        # if self.use_advanced_heuristics == 'graphcut':
        #     n_conn_sq_heur_me, n_conn_sq_count_me, conn_sq_list_me = self.__conn_sq_depth_lim_heur(grid, player_pos, max_radius=max_radius, only_heur=False)
        #     n_conn_sq_heur_opp, n_conn_sq_count_opp, conn_sq_list_opp = self.__conn_sq_depth_lim_heur(grid, opp_pos, max_radius=max_radius, only_heur=False)
        # else:
        n_conn_sq_heur_me, n_conn_sq_count_me, conn_sq_list_me = self.__conn_sq_depth_lim_heur(grid, player_pos, max_radius=max_radius, only_heur=False)
        n_conn_sq_heur_opp, n_conn_sq_count_opp, conn_sq_list_opp = self.__conn_sq_depth_lim_heur(grid, opp_pos, max_radius=max_radius, only_heur=False)

        # advanced heuristics
        if self.use_advanced_heuristics == 'graphcut':
            # stepper func to determine if we use graph_cut algo based on turn # and # of connected squares
            turns_limit = (self.turns - 2*np.ceil(self.depth_limit/3) >= 0) # starts at 4 if depth_limit in (4,5,6)
            # turns_limit = (self.turns - 2*np.ceil(self.depth_limit/3) >= 1) # starts at 5 if depth_limit in (4,5,6)
            conn_sq_limit_me = (self.curr_conn_sq_heur_me/5 + 3*(self.depth_limit//4) <= 90)
            # print(f'Graphcut? turns_limit = {turns_limit}, conn_sq_limit_me = {conn_sq_limit_me}')
            # print(f'self.curr_conn_sq_heur_me = {self.curr_conn_sq_heur_me}.')
            # conn_sq_limit_opp = (n_conn_sq_heur_opp/5 + 3*(self.depth_limit//4) <= 45)      # not used

            if turns_limit and conn_sq_limit_me and conn_sq_list_me:                        # enabled graphcut if passes these conditions


                size_cap = self.graph_cut_size_cap
                max_comp_size_me = min(int(n_conn_sq_heur_me/5), size_cap)
                max_comp_size_opp = min(int(n_conn_sq_heur_opp/5), size_cap)

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
        
        # aggregation of component and advanced heuristics
        # conn_sq_depth_lim_heur = conn_sq_depth_lim_me - conn_sq_depth_lim_opp           # redundant now
        n_conn_sq_heur = n_conn_sq_heur_me - n_conn_sq_heur_opp
        graph_cut_heur = graph_cut_heur_me - graph_cut_heur_opp

        # return grid, avoid_traps_heur + center_heur + edge_touch_heur + near_opp_heur + n_neighbors_heur + conn_sq_depth_lim_heur + n_conn_sq_heur + graph_cut_heur
        return grid, avoid_traps_heur + center_heur + edge_touch_heur + near_opp_heur + n_neighbors_heur + n_conn_sq_heur + graph_cut_heur   # removing redundant



    # not used
    # def __move(self, grid: Grid, move_pos, player_num) -> object:
    #     '''
    #     Faster move method
    #     '''
    #     old_pos = np.where(grid.map == player_num)
    #     # grid.map[old_pos], grid.map[move_pos] = grid.map[move_pos], grid.map[old_pos]
    #     grid.map[old_pos], grid.map[move_pos] = (0,0), grid.map[old_pos]       # just use 0?


    def __conn_sq_depth_lim_heur(self, grid: Grid, pos, max_radius=2, max_size=0, only_heur=True) -> int:
        '''
        Get number of connected squares for current player in current Grid object.
        Primary heuristic.
        This is a depth-limited BFS algo, sort of an IDS.
        max_size not currently used.
        Returns a heuristic int.
        '''
        if self.verbose:
            self.heur_evals += 1

        if max_radius==-1:
            max_radius = None
        radius = 1

        explored = [pos]                                    # initialize explored list with current player's position
        search_frontier = PlayerAI.__get_valid_neighbors(grid, pos, radius=1)   # a list of positions
        conn_sq_list = [pos]                                # initialize connected squares list to include current player's square
        conn_sq_count = 1                                   # initialize count at 1, the current player's square
        conn_sq_heur = 5                                    # initialize heuristic at 1*5
        current_layer = search_frontier.copy()              # initialize current layer
        next_layer = []                                     # initialize next layer
        new_neighbors = []                                  # initialize new neighbors

        # main algorithm with WEIGHTS added
        while search_frontier and radius <= max_radius:
            for this_pos in current_layer:
                if this_pos not in conn_sq_list:
                    conn_sq_list.append(this_pos)                           # add to connected square
                    conn_sq_heur += 5 + 5*(max(3-radius,0)**2)              # add to heuristic with weighting for closer squares
                    conn_sq_count += 1                                      # increment count of number of squares
                new_neighbors = PlayerAI.__get_valid_neighbors(grid, this_pos, radius=1)
                next_layer = list(set(next_layer + new_neighbors))

            # print(f'At radius = {radius}, we see {conn_sq_list}.')
            explored = list(set(explored + current_layer))
            search_frontier = list(set(search_frontier + next_layer) - set(explored))
            current_layer = search_frontier.copy()
            next_layer = []
            radius += 1

        # if max_size:                                          # not used
        #     conn_sq_list = conn_sq_list[:max_size]
        #     conn_sq_heur = max_size
        if only_heur:
            return conn_sq_heur                                 # was 5*conn_sq_heur
        else:
            return conn_sq_heur, conn_sq_count, conn_sq_list    # was 5*conn_sq_heur


    def __graph_cut_heur(self, grid: Grid, pos, comp_size, conn_sq_list, max_radius=3) -> int:
        '''
        Heuristic based on how drastic is the decrease in degrees of freedom resulting from the removal of one free space.
        Originally also computed number of connected components, but removed for performance.
            Re-added.
        This iterates over all connected squares (up to a max), and can take some time.
        Returns a heuristic int.
        '''
        grid_clone = PlayerAI.__clone(grid)
        # graph_cut_heur = 0                                                              # changed to empty list, was 0
        graph_cut_heur = []                                                             # changed to empty list, was 0
        for trap_pos in conn_sq_list:                                                   # iterate over connected squares
            grid_clone.map[trap_pos] = -1
            # new_comp_size = self.__connected_sq_heur(grid_clone, pos, max_size=comp_size, return_pos=False)   # previous DFS iteration of this function
            _, new_comp_size, _ = self.__conn_sq_depth_lim_heur(grid_clone, pos, max_radius=max_radius, max_size=comp_size, only_heur=False)
            grid_clone.map[trap_pos] = 0
            size_delta = new_comp_size - comp_size
            # utility = 10*size_delta - (49-new_comp_size)**2 - 5*i                     # updating, removed the i (from enumeration)
            utility = 5*size_delta - (7-new_comp_size)**2 - 0.1*random.random()         # using a randomness to help with tie breaking
            # graph_cut_heur += utility
            graph_cut_heur.append(utility)

        if not graph_cut_heur:
            graph_cut_heur = [0]
        # return graph_cut_heur
        return fmean(graph_cut_heur)


    def __graph_cut_2_ply_heur(self, grid: Grid, pos, comp_size, conn_sq_list, max_radius=3) -> int:
        '''
        Same as __graph_cut_heur(), but going 2 levels deep instead of 1.
        Returns a heuristic int.
        '''
        def __gch_d2(grid: Grid, trap_pos, comp_size) -> int:
            grid.map[trap_pos] = -1
            # new_comp_size = self.__connected_sq_heur(grid, pos, max_size=comp_size, return_pos=False)         # previous DFS iteration of this function
            _, new_comp_size, _ = self.__conn_sq_depth_lim_heur(grid, pos, max_radius=max_radius, max_size=comp_size, only_heur=False)
            grid.map[trap_pos] = 0
            size_delta = new_comp_size - comp_size
            # utility = 10*size_delta - (49-new_comp_size)**2 - 5*j                         # updating, removed the j (from enumeration)
            utility = 5*size_delta - (7-new_comp_size)**2 - 0.1*random.random()           # updating, removed the j (from enumeration)

            return utility

        # graph_cut_heur = 0
        graph_cut_heur = []
        grid_clone = PlayerAI.__clone(grid)
        for trap_pos in conn_sq_list:                                     # iterate over connected squares
            conn_sq_list_2 = conn_sq_list.copy()
            conn_sq_list_2.remove(trap_pos)
            grid_clone.map[trap_pos] = -1
            for trap_pos_2 in conn_sq_list_2:
                # graph_cut_heur += __gch_d2(grid_clone, trap_pos_2, comp_size) -5*i        # updating, removed the i (from enumeration)
                utility = __gch_d2(grid_clone, trap_pos_2, comp_size) - 0.1*random.random()     # updating, removed the i (from enumeration)
                # graph_cut_heur += utility
                graph_cut_heur.append(utility)
            grid_clone.map[trap_pos] = 0

        if not graph_cut_heur:
            graph_cut_heur = [0]
        # return graph_cut_heur
        return fmean(graph_cut_heur)


    def __get_heuristics(self, grid: Grid, player_num, opp_num) -> tuple:
        '''
        Apply heuristics.
        Called by each leaf (end node) of Expectiminimax.
        Starting point of function call, also tracks time if verbose.
        If using 'graphcut' advanced heuristic or no advanced heuristic, calls get_dof.
        Otherwise will only get valid neighbors as of this time because geodesics is not implemented.
        Returns a tuple of Grid object, heuristic.
        '''
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
        elif self.use_advanced_heuristics == 'geodesics':        # not implemented
            return grid, PlayerAI.__n_neighbors_heur(grid, player_pos) - PlayerAI.__n_neighbors_heur(grid, opp_pos)
            # if self.verbose:
            #     start = time.time()
            #     conn_sq_dl_me = self.__connected_sq_heur(grid, player_pos, max_size=25)
            #     conn_sq_dl_opp = self.__connected_sq_heur(grid, opp_pos, max_size=25)
            #     dl_heur = conn_sq_dl_me-conn_sq_dl_opp
            #     end = time.time()
            #     self.heur_time += end-start
            # else:
            #     conn_sq_dl_me = self.__connected_sq_heur(grid, player_pos, max_size=25)
            #     conn_sq_dl_opp = self.__connected_sq_heur(grid, opp_pos, max_size=25)
            #     dl_heur = conn_sq_dl_me-conn_sq_dl_opp
            # return grid, dl_heur                    # for next set of heuristics Matthew/Gong
        else:
            return self.__get_dof(grid, player_pos=player_pos, opp_pos=opp_pos)


    def __get_search_start_pos(self, grid: Grid, curr_conn_sq_list_target: list, attack_pos, target_pos) -> tuple:
        '''
        Find a starting spot for Trap search roughly midway between players.
            This will favor positions closer to the Opponent (target) based on logic in find_center().
        To use this from the perspective in Minimization step, reverse the pos args and use Player component list.
        We don't want to start on square that already has a trap (i.e. has value of -1).
        We also don't want to start on a square that is not part of Opponent's connected component.
            However, with each iterative search, the component being checked isn't updated, for performance.
            This means occasionally the starting search position will evaluate to be the Player's position.
            We need to explicitly allow for this.
        Starting square can be the opponent's square.
        '''
        start_pos = PlayerAI.__find_center(attack_pos, target_pos)
        # run it again to halv the distance again
        start_pos = PlayerAI.__find_center(start_pos, target_pos)

        # while grid.map[start_pos] == -1 or start_pos not in self.curr_conn_sq_list_target:
        # while start_pos not in self.curr_conn_sq_list_target:              # connected component already excludes traps
        # prev_pos = start_pos
        if self.verbose > 2:
            print(f'Looking from perspective of attacking player at {attack_pos}.')
            print(f'First trap search start candidate: {start_pos}.')

        while start_pos not in curr_conn_sq_list_target:                    # taking dynamically from argument
            if self.verbose > 2:
                print('Trap search start not in component.')
                # print(f'Trap search start candidate: {start_pos}.')
                print(f'Component squares: {curr_conn_sq_list_target}.')

            prev_pos = start_pos
            start_pos = PlayerAI.__find_center(start_pos, target_pos)
            print(f'New trap search start candidate: {start_pos}.') if self.verbose > 2 else None

            if start_pos == prev_pos:
                print('Potential loop found.') if self.verbose > 2 else None
                if start_pos == self.pos:
                    return start_pos
                # print(f'Trap search start candidate: {start_pos}.')
                # print(f'Component squares: {curr_conn_sq_list_target}.')
                else:
                    cprint('Terminating.', color='magenta', on_color='on_white')
                    exit('Loop in finding a starting point for trap search.')

        return start_pos


    def __get_trap_candidates(self, grid: Grid, player_pos, opp_pos, turn='player') -> list:
        '''
        Helper function to intelligently locate trap candidates to feed into Expectiminimax.
        First finds the connected component around the Opponent (target).
        First picks a starting location based on being halfway between the Player and Opponent.
            Using get_search_start_pos, the search start location will favor being closer to Opponent.
        Uses a BFS algo to look for connected traps, and then a basic IDS to expand the circle.
        Augmented to use a class param specifying threshold of max traps.
        Returns a list of tuples (positions)
        '''
        # start_pos = self.search_start_pos                 # find the ideal spot to start trap candidate search -- this is hard-coded to Player
        # need to dynamically find start spots
        if turn == 'opp':                                   # i.e. Player is the target
            _, _, curr_conn_sq_list = self.__conn_sq_depth_lim_heur(grid, pos=player_pos, max_radius=3, only_heur=False)
            attack_pos = opp_pos
            target_pos = player_pos
            # start_pos = self.__get_search_start_pos(grid, self.curr_conn_sq_list_me, opp_pos, player_pos)
        else:                                               # turn == 'player', Opponent is the target
            _, _, curr_conn_sq_list = self.__conn_sq_depth_lim_heur(grid, pos=opp_pos, max_radius=3, only_heur=False)
            attack_pos = player_pos
            target_pos = opp_pos
            # start_pos = self.__get_search_start_pos(grid, self.curr_conn_sq_list_opp, player_pos, opp_pos)
        
        start_pos = self.__get_search_start_pos(grid, curr_conn_sq_list, attack_pos, target_pos)       # making this dynamic too

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
            # if pos in self.curr_conn_sq_list_opp and pos not in trap_candidates and not grid.map[pos]:
            if pos in curr_conn_sq_list and pos not in trap_candidates and not grid.map[pos]:
                # need to use dynamically connected component of the current player
                # check that we're in the connected component
                # and not already in trap_candidates list
                # and is currently empty 
                trap_candidates.append(pos)                 # add to trap candidates
                trap_count += 1
            if trap_count >= self.max_search_traps:         # finish once we've collected the max # of candidates
                self.max_trap_candidates = trap_count
                return trap_candidates

            # get neighbors to add to search_frontier
            while not new_neighbors:
                if self.turns <= 3:                         # strategy optimization, don't consider trap positions on a edge until turn 4
                    # print("Don't consider traps on edges.") if self.verbose > 0 else None
                    new_neighbors = PlayerAI.__get_valid_neighbors_avoid_edge(grid, start_pos, radius)
                if not new_neighbors:
                    print(f"Somehow searching traps on edges despite being at turn {self.turns}.") if self.verbose > 0 and self.turns <= 3 else None
                    new_neighbors = PlayerAI.__get_valid_neighbors(grid, start_pos, radius)
                radius += 1
            search_frontier = list(set(search_frontier + new_neighbors) - set(explored))
            # no need to intersect with trap_candidates, because it is a subset of explored

        # contingencies
        if trap_count == 0:
            if self.verbose > 1:
                print(f'No trap positions left after visiting {self.child_nodes_seen} child nodes.')
            catch_trap = PlayerAI.__get_valid_neighbors_avoid_edge(grid, start_pos, radius)
            if not catch_trap:
                catch_trap = PlayerAI.__get_valid_neighbors(grid, start_pos, radius)
            if not catch_trap:
                catch_trap = grid.getAvailableCells()[0]
            return catch_trap
            # can terminate search?

        if trap_count > self.max_trap_candidates:
            self.max_trap_candidates = trap_count

        # if self.verbose > 1:
            # print(f'Traps seen: {trap_count}')
            # self.printed = True

        return trap_candidates


    def __trap_minimize(self, grid: Grid, alpha, beta, player_num, opp_num, depth, depth_limit) -> tuple:
        '''
        Part of Expectiminimax.
        This is the Opponent Min node for throwing Trap.
        Returns a tuple of pos, utility.
        '''
        if self.verbose:
            self.child_nodes_seen += 1
            self.current_depth += 1
            # if self.verbose > 1:
            #     print(f'In Move Min, Current depth = {self.current_depth}.')
            #     print(f'depth = {depth}, depth_limit = {depth_limit}')

        player_pos = tuple(np.argwhere(grid.map == player_num)[0])
        opp_pos = tuple(np.argwhere(grid.map == opp_num)[0])            # this is the position in question
        gameover_result = PlayerAI.__is_over(grid, player_num, player_pos, opp_num, opp_pos)
        if gameover_result:
            return PlayerAI.__evaluate(grid, gameover_result, player_num)

        # if hit depth limit, apply heuristic and break loop
        if depth >= depth_limit:
            _, utility = self.__get_heuristics(grid, player_num, opp_num)
            return _, utility

        minTrap, minUtility = None, np.inf
        cache = {}
        player_pos_a = np.array(player_pos)             # using numpy arrays to allow for faster arithmetic
        opp_pos_a = np.array(opp_pos)
        x_0, x_1 = player_pos_a - opp_pos_a
        center_a = np.array(PlayerAI.__find_center(player_pos, opp_pos))
        
        for trap_pos in self.__get_trap_candidates(grid, player_pos, opp_pos, turn='opp'):
            # initialize with the main trap's probability-weighted utility, then move on to those of the neighbors
            # this is because each throw has a small chance of landing on a square adjacent to intended square
            if self.verbose > 1:
                # print(f'In Trap Min, looking at trap_pos = {trap_pos}.')
                pass

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
                dist_mid = abs(player_pos_a - opp_pos_a) - 2*abs(center_a - trap_pos_a)    # heuristic tested on paper
                if x_0 > x_1:
                    trap_heur = dist_me + dist_opp + 4*dist_mid[0] + 2*dist_mid[1]
                elif x_0 < x_1:
                    trap_heur = dist_me + dist_opp + 2*dist_mid[0] + 4*dist_mid[1]
                else:
                    trap_heur = dist_me + dist_opp + 2*dist_mid[0] + 2*dist_mid[1]
                # cache[key] = utility
                cache[key] = utility + sum(trap_heur)
            target_prob = PlayerAI.__probability(opp_pos, trap_pos)
            expected_utility = target_prob * utility
            # backtrack
            grid.map[trap_pos] = 0

            # look at utility of closest neighbors
            neighbors = PlayerAI.__get_all_neighbors(trap_pos)
            for neighbor in neighbors:
                if grid.map[neighbor] in (0, -1):             # to avoid player and opponent location, but can land on an existing trap
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
                if self.verbose > 2:
                    if minUtility < self.trap_min_utility:
                        self.trap_min_utility = minUtility
                        cprint(f'Found better (lower) trap utility of {minUtility}.', color='red')
                        pass

        return minTrap, minUtility


    def __move_minimize(self, grid: Grid, alpha, beta, player_num, opp_num, depth, depth_limit) -> tuple:
        '''
        Part of Expectiminimax.
        This is the Opponent Min node for making Move.
        Returns a tuple of pos, utility.
        '''
        if self.verbose:
            self.child_nodes_seen += 1
            self.current_depth += 1
            # print(f'In Move Min, Current depth = {self.current_depth}.')
            # print(f'depth = {depth}, depth_limit = {depth_limit}')

        player_pos = tuple(np.argwhere(grid.map == player_num)[0])
        opp_pos = tuple(np.argwhere(grid.map == opp_num)[0])
        gameover_result = PlayerAI.__is_over(grid, player_num, player_pos, opp_num, opp_pos)
        if gameover_result:
            return PlayerAI.__evaluate(grid, gameover_result, player_num)

        # if hit depth limit, apply heuristic and break loop
        if depth >= depth_limit:
            _, utility = self.__get_heuristics(grid, player_num, opp_num)
            return _, utility

        minChild, minUtility = None, np.inf
        for neighbor_to_move in PlayerAI.__get_valid_neighbors(grid, opp_pos):
            if self.verbose > 1:
                # print(f'In Move Min, looking at pos = {neighbor_to_move}.')
                pass

            grid.map[opp_pos] = 0
            grid.map[neighbor_to_move] = opp_num
            _, utility = self.__trap_minimize(grid, alpha, beta, player_num, opp_num, depth + 1, depth_limit)
            grid.map[opp_pos] = opp_num
            grid.map[neighbor_to_move] = 0
            if utility < minUtility:
                minChild, minUtility = neighbor_to_move, utility
                if self.verbose > 2:
                    if minUtility < self.move_min_utility:
                        self.move_min_utility = minUtility
                        cprint(f'Found better (lower) move utility of {minUtility}.', color='yellow')
                        pass

            if minUtility <= alpha:
                break

            if minUtility < beta:
                beta = minUtility

        return minChild, minUtility


    def __trap_maximize(self, grid: Grid, alpha, beta, player_num, opp_num, depth, depth_limit) -> tuple:
        '''
        Part of Expectiminimax.
        This is the Player Max node for throwing Trap.
        Returns a tuple of pos, utility.
        '''
        if self.verbose:
            self.child_nodes_seen += 1
            self.current_depth += 1
            # print(f'In Move Min, Current depth = {self.current_depth}.')
            # print(f'depth = {depth}, depth_limit = {depth_limit}')            

        player_pos = tuple(np.argwhere(grid.map == player_num)[0])              # this is the position in question
        opp_pos = tuple(np.argwhere(grid.map == opp_num)[0])
        gameover_result = PlayerAI.__is_over(grid, player_num, player_pos, opp_num, opp_pos)
        if gameover_result:
            return PlayerAI.__evaluate(grid, gameover_result, player_num)

        # break if (exceed) hit depth limit, apply heuristic and break loop
        if depth > depth_limit:
            _, utility = self.__get_heuristics(grid, player_num, opp_num)
            return _, utility

        maxTrap, maxUtility = None, -np.inf
        cache = {}
        player_pos_a = np.array(player_pos)             # using numpy arrays to allow for faster arithmetic
        opp_pos_a = np.array(opp_pos)
        x_0, x_1 = player_pos_a - opp_pos_a
        center_a = np.array(PlayerAI.__find_center(player_pos, opp_pos))

        for trap_pos in self.__get_trap_candidates(grid, player_pos, opp_pos, turn='player'):
            # initialize with the main trap's probability-weighted utility, then move on to those of the neighbors
            # this is because each throw has a small chance of landing on a square adjacent to intended square
            if self.verbose > 1:
                # print(f'In Trap Max, looking at pos = {trap_pos}.')
                pass

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
            target_prob = PlayerAI.__probability(player_pos, trap_pos)
            expected_utility = target_prob * utility
            # backtrack
            grid.map[trap_pos] = 0

            neighbors = PlayerAI.__get_all_neighbors(trap_pos)
            for neighbor in neighbors:
                if grid.map[neighbor] in (0,-1):               # to avoid player and opponent location, but can land on an existing trap
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
                if self.verbose > 2:
                    if maxUtility > self.trap_max_utility:
                        self.trap_max_utility = maxUtility
                        cprint(f'Found better (high) trap utility of {maxUtility}.', color='green')
                        pass

        # returns max trap so maximize can cache it
        return maxTrap, maxUtility


    def __move_maximize(self, grid: Grid, alpha, beta, player_num, opp_num, depth, depth_limit) -> tuple:
        '''
        Part of Expectiminimax.
        This is the Player Max node for making Move.
        Returns a tuple of pos, utility.
        '''
        if self.verbose:
            self.child_nodes_seen += 1
            self.current_depth += 1
            # print(f'In Move Max, Current depth = {self.current_depth}.')
            # print(f'depth = {depth}, depth_limit = {depth_limit}')

        player_pos = tuple(np.argwhere(grid.map == player_num)[0])
        opp_pos = tuple(np.argwhere(grid.map == opp_num)[0])
        gameover_result = PlayerAI.__is_over(grid, player_num, player_pos, opp_num, opp_pos)
        if gameover_result:
            return PlayerAI.__evaluate(grid, gameover_result, player_num)

        # break if (exceed) hit depth limit, apply heuristic and break loop
        if depth > depth_limit:
            _, utility = self.__get_heuristics(grid, player_num, opp_num)
            return _, utility

        maxMove, maxTrap, maxUtility = None, None, -np.inf
        for neighbor_to_move in PlayerAI.__get_valid_neighbors(grid, player_pos):
            if self.verbose > 1:
                # print(f'In Move Max, looking at pos = {neighbor_to_move}.')
                pass

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
                if self.verbose > 2:
                    if maxUtility > self.move_max_utility:
                        self.move_max_utility = maxUtility
                        cprint(f'Found better (high) move utility of {maxUtility}.', color='cyan')
                        pass

        self.optimal_trap_pos = maxTrap        # maxTrap is now just a position, not a Grid object

        return maxMove, maxUtility


    def __decision(self, grid: Grid, alpha, beta, player_num, opp_num, depth_limit=DEFAULT_DEPTH_LIMIT) -> object:
        '''
        Helper function to start the Expectiminimax algo.
        Called by getMove().
        Returns a Grid object.
        '''
        if self.verbose:
            # start = time.time()
            child, self.utility = self.__move_maximize(grid, alpha, beta, player_num, opp_num, depth=0, depth_limit=depth_limit)
            print(f'Max trap candidates seen from one child: {self.max_trap_candidates}')
            if self.use_advanced_heuristics:
                print(f'Heuristics took {self.heur_time:.3f} seconds to complete.')
            # if end-start >= 5.05:
                if self.use_graph_me:
                    cprint(f'Used graph cut heuristic on player, size_cap={self.graph_cut_size_cap}, max_radius={self.max_radius}.', color='blue')
                if self.use_graph_d2_me:
                    cprint(f'Used 2-ply graph cut heuristic on player, size_cap={self.graph_cut_size_cap}, max_radius={self.max_radius}.', color='blue')
                if self.use_graph_opp:
                    cprint(f'Used graph cut heuristic on opponent, size_cap={self.graph_cut_size_cap}, max_radius={self.max_radius}.', color='red')
                if self.use_graph_d2_opp:
                    cprint(f'Used 2-ply graph cut heuristic on opponent, size_cap={self.graph_cut_size_cap}, max_radius={self.max_radius}.', color='red')
            
            print(f'Best move found has utility of {self.utility:.2f}. ', end='')
            # if self.utility >= 90000:
                # print('Win imminent? 😱')
            # elif self.utility >= 1000:
                # print('Doing decent! 😲')
            # elif self.utility <= -1000:
                # print('Not looking so good. 😟')
            # elif self.utility <= -90000:
                # print('OMG no... 😵‍💫')
            # else:
                # print('Hey, doing what we can. 😐')
            # end = time.time()
            # print(f'This move took {end-start:.5f} seconds.')
        else:
            child, _ = self.__move_maximize(grid, alpha, beta, player_num, opp_num, depth=0, depth_limit=depth_limit)
        return child


    def getTrap(self, grid: Grid) -> tuple:
        '''
        YOUR CODE GOES HERE

        The function should return a tuple of (x,y) coordinates to which the player *WANTS* to throw the trap.
        
        It should be the result of the ExpectiMinimax algorithm, maximizing over the Opponent's *Move* actions, 
        taking into account the probabilities of it landing in the positions you want. 
        
        Note that you are not required to account for the probabilities of it landing in a different cell.

        You may adjust the input variables as you wish (though it is not necessary). Output has to be (x,y) coordinates.
        
        '''

        # TODO - no more
        # need to code this up separately from getMove
        # essentially need two Expectiminimax algos
        # March 15 WAX

        # get player numbers variables
        # player_num = self.getPlayerNum()
        # opp_num = self.getOpponentNum()
        # self.opp_pos = tuple(np.argwhere(grid.map == opp_num)[0])              # find opponent's current position

        # if self.verbose:
        #     self.heur_evals = 0
        #     self.heur_time = 0
        #     self.child_nodes_seen = 0
        #     self.printed = False
        #     self.utility = 0
        #     self.current_depth = 0

        # print(f'Looking for best trap. Depth limit is {self.depth_limit}.') if self.verbose else None

        # if no available valid neighbors around opponent, throw to first available cell, starting from upper-left-most square
        # TODO: change the trap position to be somewhere not in the vicinity of current player
        if not PlayerAI.__get_valid_neighbors(grid, self.getOpponentPosition(grid)):
            print('PlayerAI')
            input(f'No available cells around player {3 - self.player_num}! Press enter to continue.') if self.verbose == 3 else None
            return grid.getAvailableCells()[0]

        # other edge cases
        if not self.optimal_trap_pos:
            input('No optional trap position in cache.') if self.verbose == 3 else None
            return grid.getAvailableCells()[0]
        if self.optimal_trap_pos == self.current_move:
            # if game ends because we took the last spot
            input("We've taken up the last spot. Throwing trap randomly because we'll win anyway.") if self.verbose == 3 else None
            return grid.getAvailableCells()[0]

        # use cached optimal trap position that we computed in getMove()
        print(f'Using cached optimal trap position that was found during Expectiminimax.') if self.verbose else None

        return self.optimal_trap_pos

