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
        self.optimal_trap_position = None
        self.depth_limit = depth_limit
        if self.depth_limit == 0:
            self.depth_limit = DEFAULT_DEPTH_LIMIT
        self.turns = 1              # early game = 1-3, mid = 4-6, late to 7+; generally early game <= grid.dim/2, mid = 2xearly
        self.use_advanced_heuristics = heur
        self.use_graph_me = False
        self.use_graph_opp = False
        self.curr_conn_sq = 48
        self.printed = False

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
        self.heur_evals = 0
        self.heur_time = 0
        self.opp_pos = self.getOpponentPosition(grid)               # find opponent's current position

        # depth_delta = int((self.turns+3)**(2)/150)            # adjust based on turn
        depth_delta = 0                                         # adjust based on turn
        depth_limit = self.depth_limit + depth_delta
        print(f'Depth limit is currently {depth_limit}.')

        # get players's current connected sq 
        curr_conn_sq_me, curr_conn_sq_list_me = self.__connected_sq_heur(grid, max_radius=-1, return_pos=True, is_me=True)
        curr_conn_sq_opp, curr_conn_sq_list_opp = self.__connected_sq_heur(grid, max_radius=-1, return_pos=True, is_me=False)
        self.curr_conn_sq_me = curr_conn_sq_me/5
        self.curr_conn_sq_opp = curr_conn_sq_opp/5
        self.curr_conn_sq_list_me = curr_conn_sq_list_me
        self.curr_conn_sq_list_opp = curr_conn_sq_list_opp
        print(f'Player\'s component r<=3 has {int(self.curr_conn_sq_me)} connected squares; opponent\'s component has {int(self.curr_conn_sq_opp)}.')
        

        # get maximum traps to search per call to Expectiminimax
        # turn_adjust = int((self.turns**2)/8)-2               # adjustment based on turn
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

        # if self.turns > 1:
        #     exit()
        alpha = -np.inf
        beta = np.inf
        max_grid = self.__decision(grid, alpha, beta, depth_limit)
        print(f'Total iterative search evals = {self.heur_evals}')
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


    def __evaluate(self, grid: Grid, gameover_result) -> int:
        """
        Function for returning high or low utility based on gameover state.
        """
        if gameover_result:
            if gameover_result == self.getPlayerNum():
                return grid, 99999
            else:
                return grid, -99999


    def __get_dof(self, grid: Grid, is_me) -> tuple:
        """
        Apply degrees of freedom calculation heuristic based on early, mid, late game.
        Returns a tuple of Grid object, heuristic
        """

        edge_touch_heur = 0
        n_conn_sq_heur_me = 48
        conn_sq_list_me = []
        n_conn_sq_heur_opp = 48
        conn_sq_list_opp = []
        n_conn_sq_heur = 0
        conn_sq_depth_lim_heur = 0
        graph_cut_heur_me = 0
        graph_cut_heur_opp = 0
        graph_cut_heur = 0

        center_heur = self.__center_heur(grid, is_me=True) - self.__center_heur(grid, is_me=False)
        toward_opp_heur = self.__toward_opp_heur(grid)

        # if self.turns >= 1:
        n_neighbors_heur = self.__n_neighbors_heur(grid, is_me=True)
        if self.turns >= 2:
            n_neighbors_heur = self.__n_neighbors_heur(grid, is_me=True) - self.__n_neighbors_heur(grid, is_me=False)
        if self.turns >= 3:
            edge_touch_heur = self.__edge_touch_heur(grid, is_me=True) - self.__edge_touch_heur(grid, is_me=False)
        if self.turns >= 5:
            conn_sq_depth_lim_heur = self.__conn_sq_depth_lim_heur(grid, is_me=True) - self.__conn_sq_depth_lim_heur(grid, is_me=False)
        # if self.turns <= 5:
        #     n_conn_sq_heur_me = self.__connected_sq_heur(grid, max_radius=2, return_pos=False, is_me=True)
        #     n_conn_sq_heur_opp = self.__connected_sq_heur(grid, max_radius=2, return_pos=False, is_me=False)
        if self.turns <= 7:
            n_conn_sq_heur_me = self.__connected_sq_heur(grid, max_radius=3, return_pos=False, is_me=True)
            n_conn_sq_heur_opp = self.__connected_sq_heur(grid, max_radius=3, return_pos=False, is_me=False)
        else: # self.turns >= 8:
            n_conn_sq_heur_me, conn_sq_list_me = self.__connected_sq_heur(grid, max_radius=-1, return_pos=True, is_me=True)
            n_conn_sq_heur_opp, conn_sq_list_opp = self.__connected_sq_heur(grid, max_radius=-1, return_pos=True, is_me=False)
            # grid.print_grid()

        turns_limit = (self.turns - 2*np.ceil(self.depth_limit/3) >= 9)
        # conn_sq_limit_me = (self.curr_conn_sq_me + 3*(self.depth_limit//4) <= 30)
        # conn_sq_limit_opp = (self.curr_conn_sq_opp + 3*(self.depth_limit//4) <= 30)
        conn_sq_limit_me = (n_conn_sq_heur_me + 3*(self.depth_limit//4) <= 30)
        conn_sq_limit_opp = (n_conn_sq_heur_opp + 3*(self.depth_limit//4) <= 30)

        if turns_limit and conn_sq_limit_me:
            self.use_graph_me = True
            # cprint('Using graph cut heuristic on player.', 'blue')
            graph_cut_heur_me = self.__graph_cut_heur(grid, comp_size=n_conn_sq_heur_me, conn_sq=conn_sq_list_me, is_me=True)

        if turns_limit and conn_sq_limit_opp:
        # if self.turns >= 12 or ( self.curr_conn_sq_opp * < 24:
            self.use_graph_opp = True
            # cprint('Using graph cut heuristic on opponent.', 'blue')
            graph_cut_heur_opp = self.__graph_cut_heur(grid, comp_size=n_conn_sq_heur_opp, conn_sq=conn_sq_list_opp, is_me=False)

        n_conn_sq_heur = n_conn_sq_heur_me - n_conn_sq_heur_opp
        graph_cut_heur = graph_cut_heur_me - graph_cut_heur_opp
        return grid, center_heur + toward_opp_heur + conn_sq_depth_lim_heur + n_conn_sq_heur + n_neighbors_heur + edge_touch_heur + graph_cut_heur


    def __edge_touch_heur(self, grid: Grid, is_me=True) -> int:
        """
        Heuristic that penalizes being on an edge (more if on two edges, i.e. a corner)
        Returns a heuristic int
        """
        if is_me:
            pos = self.getPlayerPosition(grid)
        else:
            pos = self.getOpponentPosition(grid)
        edges_touched = 0
        if pos[0] in (0,6):
            edges_touched -= 1
        if pos[1] in (0,6):
            edges_touched -= 1
        return 20*edges_touched


    def __center_heur(self, grid: Grid, is_me=True) -> int:
        """
        Heuristic that priotizes being closer to center
        Returns a heuristic int
        """
        if is_me:
            pos = self.getPlayerPosition(grid)
        else:
            pos = self.getOpponentPosition(grid)
        dist = 6 - grid_distance(pos, (3,3))
        return 4*dist


    def __toward_opp_heur(self, grid: Grid) -> int:
        """
        Heuristic that prioritizes being closer to opponent
        Returns a heuristic int
        """
        pos = self.getPlayerPosition(grid)
        other_pos = self.getOpponentPosition(grid)
        dist = 6-grid_distance(pos, other_pos)
        return 5*dist


    def __connected_sq_heur(self, grid: Grid, pos=None, max_radius=2, return_pos=False, is_me=True) -> tuple:
        """
        Get number of connected squares for current player in current Grid object.
        This is a DFS algo
        Returns a tuple of heuristic int, list of positions
        """
        self.heur_evals += 1
        if not pos:
            if is_me:
                pos = self.getPlayerPosition(grid)
            else:
                pos = self.getOpponentPosition(grid)

        if max_radius == -1:
            max_radius = 47

        explored = [pos]                # track explored squares
        search_frontier = [pos]         # initialize with current position
        conn_sq_list = [pos]            # list of connected squares, initialize at player position
        conn_sq_heur = 1                # number of squares connected to position, initialize at 1

        radius = 0

        while search_frontier and radius <= max_radius:
            current_sq = search_frontier.pop()
            for child_pos in self.__get_valid_neighbors(grid, current_sq):
                if child_pos not in explored and child_pos not in conn_sq_list:
                    explored.append(child_pos)
                    search_frontier.append(child_pos)
                    conn_sq_list.append(child_pos)
                    conn_sq_heur += 1
            radius += 1

        if return_pos:
            # conn_sq_heur = len(conn_sq_list)      # redundant
            # print(f'Current player has {conn_sq_heur} connected squares.')
            return 5*conn_sq_heur, conn_sq_list
        else:
            # print(f'Current player has {conn_sq_heur} connected squares.')
            return 5*conn_sq_heur


    def __conn_sq_depth_lim_heur(self, grid: Grid, pos=None, max_radius=2, is_me=True) -> int:
        """
        Get number of connected squares for current player in current Grid object.
        This is a depth-limited BFS algo
        Returns a heuristic int
        """
        self.heur_evals += 1
        start = time.time()
        if max_radius==-1:
            max_radius = None
        radius = 1
        if not pos:
            if is_me:
                pos = self.getPlayerPosition(grid)
            else:
                pos = self.getOpponentPosition(grid)

        search_frontier = self.__get_valid_neighbors(grid, pos, radius=1)
        explored = [pos]                                    # initialize explored list with position
        conn_sq_list = []                                   # initialize connected squares list
        conn_sq_heur = 0

        while search_frontier and radius <= max_radius:        
            for pos in search_frontier:
                if pos not in explored and pos not in conn_sq_list:
                    explored.append(pos)                    # add to explored list
                    conn_sq_list.append(pos)                # add to connected square
                    conn_sq_heur += 1
                new_neighbors = self.__get_valid_neighbors(grid, pos, radius=1)
                search_frontier.remove(pos)
                search_frontier = list(set(search_frontier) | set(new_neighbors))
            radius += 1

        if conn_sq_heur == 0:
            print(pos)
            raise Exception('Connected squares is 0')
        end = time.time()
        run_time = end-start
        # print(f'Heuristic eval took {run_time} seconds')

        return 10*conn_sq_heur


    def __connected_sq_heur_old(self, grid: Grid, return_pos=False, pos=None, is_me=True) -> tuple:
        """
        Get number of connected squares for current player in current Grid object.
        Returns a tuple of heuristic int, list of positions
        """
        if not pos:
            if is_me:
                pos = self.getPlayerPosition(grid)
            else:
                pos = self.getOpponentPosition(grid)
        explored = [pos]                # track explored squares
        stack = Q.LifoQueue()           # initialize stack
        stack.put(pos)                  # push current position onto stack
        conn_sq_list = [pos]            # list of connected squares, initialize at player position
        conn_sq_heur = 1                # number of squares connected to position, initialize at 1

        while not stack.empty():
            current_sq = stack.get()
            has_children = False
            for child_pos in self.__get_valid_neighbors(grid, current_sq):
                if child_pos not in explored and child_pos not in conn_sq_list:
                    explored.append(child_pos)
                    stack.put(child_pos)
                    conn_sq_list.append(child_pos)
                    conn_sq_heur += 1
                    has_children = True
            if not has_children and not stack.empty():
                stack.get_nowait()

        if return_pos:
            # conn_sq_heur = len(conn_sq_list)      # redundant
            # print(f'Current player has {conn_sq_heur} connected squares.')
            return 5*conn_sq_heur, conn_sq_list
        else:
            # print(f'Current player has {conn_sq_heur} connected squares.')
            return 5*conn_sq_heur


    def __graph_cut_heur(self, grid: Grid, comp_size, conn_sq, is_me=True) -> int:
        """
        Heuristic based on how easy it is to increase the number of connected components within the board,
        and how drastic the decrease in freedom resulting from the removal of one free space
        Returns a heuristic int
        """
        if is_me:
            pos = self.getPlayerPosition(grid)
        else:
            pos = self.getOpponentPosition(grid)
        # conn_comps = self.__num_connected_components(grid, is_me=is_me)         # number of connected components
        
        graph_cut_heur = 0
        for trap_pos in conn_sq:                                                # iterate over connected squares
            trap_clone = grid.clone()
            trap_clone.trap(trap_pos)
            # new_conn_comps = self.__num_connected_components(grid, is_me=is_me)
            new_comp_size = self.__connected_sq_heur(grid, return_pos=False, is_me=is_me)
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
        if self.use_advanced_heuristics == 'graphcut':
            start = time.time()
            dof_heur = self.__get_dof(grid, is_me=is_me)
            end = time.time()
            self.heur_time += end-start
            return dof_heur
        elif self.use_advanced_heuristics == 'geodesics':
            start = time.time()
            # conn_sq_dl_me = self.__conn_sq_depth_lim_heur(grid, is_me=True)
            # conn_sq_dl_opp = self.__conn_sq_depth_lim_heur(grid, is_me=False)
            conn_sq_dl_me = self.__connected_sq_heur(grid, max_radius=2, is_me=True)
            conn_sq_dl_opp = self.__connected_sq_heur(grid, max_radius=2, is_me=False)
            dl_heur = conn_sq_dl_me-conn_sq_dl_opp
            end = time.time()
            self.heur_time += end-start
            return grid, dl_heur                   # for next set of heuristics Matthew/Gong
        else:
            return grid, self.__n_neighbors_heur(grid, is_me=True) - self.__n_neighbors_heur(grid, is_me=False)
        # return self.__connected_sq_heur(grid, is_me)


    def __n_neighbors_heur(self, grid: Grid, is_me=True) -> int:
        """
        Returns the difference in available squares around player vs available squares around opponent.
        """
        if is_me:
            pos = self.getPlayerPosition(grid)
            # other_pos = self.getOpponentPosition(grid)
        else:
            pos = self.getOpponentPosition(grid)
            # other_pos = self.getPlayerPosition(grid)

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
        Returns a list of positions
        
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


    def __get_trap_candidates(self, grid: Grid, pos) -> list:
        """
        helper function to get a list of open spots on the board
        default is max_radius of 2 squares away from position param
        returns a list of tuples (positions)
        augmented to use a class param specifying threshold of max traps, default to 10
        """
        # all_available_pos = grid.getAvailableCells()
        # neighbors = self.__connected_sq_heur(grid, return_pos=True, pos=position, get_traps=True, radius=2)
        def __try_next_search_pos(pos, checked_sq):
            for radius in range(7):
                neighbors = self.__get_valid_neighbors_avoid_edge(grid, pos, radius=radius)
                if not neighbors:
                    neighbors = self.__get_valid_neighbors(grid, pos, radius=radius)
                if not neighbors:
                    radius += 1
                    continue
                for neighbor in neighbors:
                    if neighbor not in checked_sq:
                        checked_sq.append(neighbor)
                        return neighbor, checked_sq

        max_radius = 3
        radius = 1
        search_frontier = []

        start_search_pos = ( int(np.ceil( (self.pos[0] + self.opp_pos[0]) / 2 )), int(np.ceil( (self.pos[1] + self.opp_pos[1]) / 2 )) )   # start search midway between players
        checked_sq = []
        while start_search_pos not in self.curr_conn_sq_list_opp:
            start_search_pos, checked_sq = __try_next_search_pos(start_search_pos, checked_sq)
        pos = start_search_pos
        # print(f'Strategy, start throwing between opponent and us, pos = {pos}.')

        if self.turns <= 5:                                 # strategy optimization
            search_frontier = self.__get_valid_neighbors_avoid_edge(grid, pos, radius=1)
        if not search_frontier:
            search_frontier = self.__get_valid_neighbors(grid, pos, radius=1)
        if not search_frontier:
            search_frontier = self.__get_valid_neighbors(grid, pos, radius=2)
        explored = [pos]                                    # initialize with position
        trap_candidates = []                                # initialize list of trap candidates to return
        trap_count = 0
        self.printed = False

        while search_frontier and radius <= max_radius:
            for pos in search_frontier:                         # iterate over all search frontier
                if pos not in explored and grid.getCellValue(pos) == 0:     # check square is empty
                    if pos in self.curr_conn_sq_list_opp:       # make sure we're in the opponent's connected component
                        explored.append(pos)                    # add to explored
                        trap_candidates.append(pos)             # add to trap candidates
                        trap_count += 1
                if trap_count >= self.max_search_traps:
                    # print('Reached max # of trap positions to consider.')
                    # print(trap_candidates)
                    self.max_trap_candidates = trap_count
                    if trap_count == 0:
                        raise Exception('Whoa')
                    return trap_candidates
                search_frontier.remove(pos)
                new_neighbors = []
                if self.turns <= 5:                             # strategy optimization
                    new_neighbors = self.__get_valid_neighbors_avoid_edge(grid, pos, radius)
                if not new_neighbors:
                    new_neighbors = self.__get_valid_neighbors(grid, pos, radius)
                search_frontier = list(set(search_frontier) | set(new_neighbors))
                radius += 1

                    # if not self.printed:
                        # print(f'Traps seen: {trap_count}')
                        # self.printed = True

        if trap_count == 0:
            # print(pos)
            return [grid.getAvailableCells()[0]]
            # raise Exception('trap_count is 0')

        if trap_count > self.max_trap_candidates:
            self.max_trap_candidates = trap_count

        # if not self.printed:
            # print(f'Traps seen: {trap_count}')
            # self.printed = True

        # if grid_distance(position, trap_pos):
        # print('Reached max search radius.')
        # print(trap_candidates)
        return trap_candidates


    def __get_all_available_traps_old(self, grid, position):
        all_available_pos = grid.getAvailableCells()
        available_traps = []
        threshold = 2
        for trap_pos in all_available_pos:
            if grid_distance(position, trap_pos) < threshold:
                available_traps.append(trap_pos)
        return available_traps


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
        gameover_result = self.__is_over(grid, self.getPlayerNum())
        if gameover_result:
            # if game ends because move above results in a gameover, then we need to place a valid trap somewhere randomly
            # grid = self.__trap_children(grid, is_me=False)[0]
            return self.__evaluate(grid, gameover_result)
        if depth >= depth_limit:
            return self.__get_heuristics(grid, is_me=True)

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
        return minTrap, minUtility


    def __move_minimize(self, grid: Grid, alpha, beta, depth, depth_limit) -> tuple:
        """
        opponent Min node for making Move
        returns a tuple of Grid object, utility
        returns a grid object and associated utility
        """
        gameover_result = self.__is_over(grid, self.getOpponentNum())
        if gameover_result:
            return self.__evaluate(grid, gameover_result)

        # break if hit depth limit
        if depth >= depth_limit:
            return self.__get_heuristics(grid, is_me=True)

        minChild, minUtility = None, np.inf

        for child in self.__move_children(grid, is_me=False):
            _, utility = self.__trap_minimize(child, alpha, beta, depth+1, depth_limit)

            if utility < minUtility:
                minChild, minUtility = child, utility

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
        gameover_result = self.__is_over(grid, self.getPlayerNum())
        if gameover_result:
            # if game ends because move above results in a gameover, then we need to place a valid trap somewhere randomly
            # grid = self.__trap_children(grid, is_me=True)[0]
            return self.__evaluate(grid, gameover_result)

        if depth >= depth_limit:
            return self.__get_heuristics(grid, is_me=True)

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
        # returns max trap so maximize can cache it
        return maxTrap, maxUtility


    def __move_maximize(self, grid: Grid, alpha, beta, depth, depth_limit) -> tuple:
        """
        player Min node for making Move
        returns a tuple of Grid object, utility
        """
        gameover_result = self.__is_over(grid, self.getPlayerNum())
        if gameover_result:
            return self.__evaluate(grid, gameover_result)

        # break if hit depth limit
        if depth >= depth_limit:
            return self.__get_heuristics(grid, is_me=True)

        maxMove, maxTrap, maxUtility = None, None, -np.inf

        for move in self.__move_children(grid, is_me=True):
            trap, utility = self.__trap_maximize(move, alpha, beta, depth+1, depth_limit)
            if utility > maxUtility:
                maxMove, maxTrap, maxUtility = move, trap, utility

            if maxUtility >= beta:
                break

            if maxUtility > alpha:
                alpha = maxUtility

        if hasattr(maxTrap, 'trap_position'):
            self.optimal_trap_position = maxTrap.trap_position

        return maxMove, maxUtility


    def __tie_break(self, Move_A: Grid, Move_B: Grid, Trap_A: Grid, Trap_B: Grid, is_me) -> tuple:
        """
        Breaks ties when two children have equal utility
        returns a tuple of Grid objects

        is_me == True means this is coming from a Max node. we will get the player position and maximize the marginal utility
        is_me == False means this is coming from a Min node. we will get the opponent position and maximize their marginal utility
        """
        def __compare_and_return() -> None:
            if marginal_utility_B > marginal_utility_A:
                return Move_B, Trap_B
            elif marginal_utility_B < marginal_utility_A:
                return Move_A, Trap_A

        if is_me:
            pos_A = self.getPlayerPosition(Move_A)
            pos_B = self.getPlayerPosition(Move_B)
        else:
            pos_A = self.getOpponentPosition(Move_A)
            pos_B = self.getOpponentPosition(Move_B)
        marginal_utility_A = 0
        marginal_utility_B = 0
       
        # maximize free spaces delta
        num_free_spaces_A = self.__n_neighbors_heur(Move_A, is_me)
        marginal_utility_A += num_free_spaces_A
        num_free_spaces_B = self.__n_neighbors_heur(Move_B, is_me)
        marginal_utility_B += num_free_spaces_B
        __compare_and_return()      # waterfalls onward if tie is not broken yet

        # avoid traps
        neighbors = self.__get_valid_neighbors(Mova_A, pos_A)
        num_traps = len([neighbor for neighbor in neighbors if Move_A.map[neighbor] == -1])
        marginal_utility_A -= num_traps
        neighbors = self.__get_valid_neighbors(Move_B, pos_B)
        num_traps = len([neighbor for neighbor in neighbors if Move_B.map[neighbor] == -1])
        marginal_utility_B -= num_traps
        __compare_and_return()

        # avoid edges
        marginal_utility_A += self.__edge_touch_heur(Move_A, is_me)
        marginal_utility_B += self.__edge_touch_heur(Move_B, is_me)
        __compare_and_return()      # waterfalls onward if tie is not broken yet

        # if marginal_utility_B > marginal_utility_A:
            # if is_me:
                # print('Using tie break for Max node.')
            # else:
                # print('Using tie break for Min node.')
            # time.sleep(1)
            # return Move_B, Trap_B
            # return Move_A, Trap_A
        # else:
            # return Move_A, Trap_A
        return Move_A, Trap_A            # default to same behavior as before tie breaker


    def __decision(self, grid: Grid, alpha, beta, depth_limit=DEFAULT_DEPTH_LIMIT) -> object:
        """
        Helper function to start the Expectiminimax algo
        returns a Grid object
        """
        start = time.time()
        child, _ = self.__move_maximize(grid, alpha, beta, depth=0, depth_limit=depth_limit)
        end = time.time()
        print(f'max trap candidates seen from one child: {self.max_trap_candidates}')
        if self.use_advanced_heuristics:
            print(f'Heuristics took {self.heur_time:.3f} seconds to complete.')
        # if end-start >= 5.05:
        if self.use_graph_me:
            print('Used graph cut heuristic on player.')
        if self.use_graph_opp:
            print('Used graph cut heuristic on opponent.')
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
        if self.optimal_trap_position is None:
            return grid.getAvailableCells()[0]
        return self.optimal_trap_position