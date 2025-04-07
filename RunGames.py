# wax
from ast import literal_eval
from os import system, remove
from os.path import exists
from platform import system as os_type
from sys import argv
from subprocess import run
from termcolor import cprint
from time import time, sleep
from tqdm import tqdm

is_unix = os_type()

class RunGames(object):
    '''
    Helper script to run Game.py n times to test successful execution, win/loss, and gather statistics.
    Outputs results to batch_results.txt

    Usage:
    $ python3 RunGames.py [n] -c -v -p -g -h -h2 -d [depth_limit] -oa [opponent AI level] -od [opponent_depth_limit] 
    [N] : number of processes to run
    -c  : clear terminal screen prior to running
    -v  : verbose output to terminal
    -p  : show progress bars
    -g  : show game output (note: very verbose)
    -h  : enable advanced heuristics
    -d  : set player search depth limit of [depth_limit], min 1, default 4
    -oa : set opponent AI level [0-4]
    -od : set opponent search depth limit of [opp_depth_limit], min 1, default 2 (only applicable if opponent AI level is 2 or above)
    
    Examples:
    $ python3 RunGames.py 100 -c -v -p -g -h -d 4 -oa 0 -od 1
    $ python3 RunGames.py 100 -c -p -h -d 6 -oa 4 -od 2 
    '''

    def __init__(self, n, verbose, progress, suppress_output, heur, depth_limit, opp_ai_int, opp_depth_limit, results_filename):
        self.n = n
        self.verbose = verbose
        self.progress = progress
        self.suppress_output = suppress_output
        self.heur = heur
        self.depth_limit = depth_limit
        self.opp_ai_int = opp_ai_int
        self.opp_depth_limit = opp_depth_limit
        self.run_success = 0
        self.player_wins = 0
        self.filename = results_filename
        self.run_arg_list = []
        self.run_arg_list3 = []


    def start_batch(self):
        # begin batch run


        # build command line arguments
        self.run_arg_list = ['python', 'Game.py', '-t', '-d', str(self.depth_limit), '-oa', str(self.opp_ai_int), '-od', str(self.opp_depth_limit)]
        self.run_arg_list3 = ['python3', 'Game.py', '-t', '-d', str(self.depth_limit), '-oa', str(self.opp_ai_int), '-od', str(self.opp_depth_limit)]
        if self.heur == 'graphcut':
            self.run_arg_list.append('-h')
            self.run_arg_list3.append('-h')
        if self.heur == 'geodesics':
            self.run_arg_list.append('-h2')
            self.run_arg_list3.append('-h2')
        if self.verbose == 1:
            self.run_arg_list.append('-v')
            self.run_arg_list3.append('-v')
        if self.verbose == 2:
            self.run_arg_list.append('-vv')
            self.run_arg_list3.append('-vv')
        if self.verbose == 3:
            self.run_arg_list.append('-vvv')
            self.run_arg_list3.append('-vvv')

        # write command line to file for clarity
        delimiter = ' '
        self.run_arg_list3_str = delimiter.join(self.run_arg_list3)
        with open(self.filename, 'a') as f:
            # print(self.run_arg_list3)
            f.write(self.run_arg_list3_str)
            f.write('\n\n')

        start_batch = time()
        run_times = []
        rounds_list = []    

        if self.progress:
            for it in tqdm(range(1, self.n+1)):         # show progress bars
                run_time, rounds = self.__run_process(it)
                run_times.append(run_time)
                rounds_list.append(rounds)

        else:
            for it in range(1, self.n+1):               # no progress bars
                run_time, rounds = self.__run_process(it)
                run_times.append(run_time)
                rounds_list.append(rounds)

        end_batch = time()
        total_time = end_batch - start_batch

        return total_time, run_times, rounds_list, self.run_success, self.player_wins

    def __run_process(self, it):
        
        # print(self.run_arg_list3)
        
        start_run = time()
        try:
            # result = run(['python', 'Game.py', '-t', '-d', str(self.depth_limit), '-a', str(self.opp_depth_limit)], capture_output=self.suppress_output)
            # print(self.run_arg_list3)
            result = run(self.run_arg_list3, capture_output=self.suppress_output)
        except:
            # result = run(['python3', 'Game.py', '-t', '-d', str(self.depth_limit), '-a', str(self.opp_depth_limit)], capture_output=self.suppress_output)
            result = run(self.run_arg_list, capture_output=self.suppress_output)

        end_run = time()
        run_time = end_run-start_run
        stdout = str(result.stdout)
        returncode = result.returncode
        if returncode == 1:                     # encountered error, exit code = 1
            winning_player = -1
            rounds = 1
        else:
            # Game.py returns exit code of format [winningplayer][rounds], e.g. 119 = player 1 wins in 19 rounds
            code_str = str(returncode)
            winning_player = int(code_str[0])
            rounds = int(code_str[1:])
        # print(int(winning_player))
        # exit(0)
        # print(stdout[stdout.rfind(':')+2:stdout.rfind('\\')])
        # try:
            # rounds = int(stdout[stdout.rfind(':')+2:stdout.rfind('\\')])
        # except:
            # rounds = 0

        # animation to show progress indicator if self.verbose is 0
        if not self.verbose:
            tri = it % 3
            if tri == 1:
                print('> .  ', end='\r')
            elif tri == 2:
                print('> .. ', end='\r')
            else:
                print('> ...', end='\r')

        with open(self.filename, 'a') as f:

            # if winning_player == 0:         # normal exit code is 0
            if winning_player == 1:             # normal exit code is 0
                win = f'Run {it}: Player Wins! Process completed in {run_time:.3f} seconds.\n'
                if self.verbose:
                    cprint('\n\n' + win, on_color='on_green') if is_unix else print('\n\n' + win)
                f.write(win)
                self.run_success += 1           # increment number of successful runs
                self.player_wins += 1           # increment number of player wins
                print()

            elif winning_player == 2:       # player has lost
                loss = f'Run {it}: Player Loses! Process completed in {run_time:.3f} seconds.\n'
                if self.verbose:
                    cprint('\n\n' + loss, on_color='on_red') if is_unix else print('\n\n' + loss)
                f.write(loss)
                self.run_success += 1           # increment number of successful runs
                print()

            else:                               # encountered runtime error, exit code was 1
                error = f'Run {it}: Runtime error...\n'
                if self.verbose:
                    cprint('\n\n' + error, on_color='on_yellow') if is_unix else print('\n\n' + error)
                f.write(error)
                stderr = str(result.stderr)
                # search_from = len(stderr) - 200
                # stderr_str = literal_eval("\"" + stderr[stderr.find('line', search_from) : ] + "\"")
                stderr_str = literal_eval("\"" + stderr[stderr.rfind('line') : ] + "\"")
                # stderr_str = literal_eval("\"" + stderr + "\"")
                print(stderr_str) if self.verbose else None
                f.write(stderr_str)
                f.write('\n')
        # sleep(1)
        return run_time, rounds


def main():
    clear = lambda: system('clear')   # pretty cool way to clear the output, should look into more
    n = 100
    verbose = 0
    progress = False
    suppress_output = True
    heur = False
    depth_limit = 4
    opp_ai_int = 0        # if 0 or 1, depth limit won't apply
    opp_ai_level = 'Easy AI'
    opp_depth_limit = 2
    depth_str = ''
    opp_depth_str = ''
    heur_str = ''
    dl_flag_index = -1
    opp_dl_flag_index = -1
    opp_ai_int_flag_index = -1
    comment = False
    com_flag_index = -1
    com_str = ''

    if len(argv)>1:
        dl_flag_index = 0
        if '-d' in argv:
            try:
                dl_flag_index = argv.index('-d')
                depth_limit = argv[dl_flag_index+1]
                depth_str = f'_d_{depth_limit}'
            except:
                pass
        if '-oa' in argv:
            try:
                opp_ai_int_flag_index = argv.index('-oa')
                opp_ai_int = int(argv[opp_ai_int_flag_index+1])
                opp_ai_level_str = f'_oa_{opp_ai_int}'
            except:
                pass
        if opp_ai_int > 1 and '-od' in argv:
            try:
                opp_dl_flag_index = argv.index('-od')
                opp_depth_limit = int(argv[opp_dl_flag_index+1])
                opp_depth_str = f'_od_{opp_depth_limit}'
            except:
                pass
        if '-m' in argv:
            try:
                com_flag_index = argv.index('-m')
                comment = argv[com_flag_index+1]
            except:
                pass

        num = [arg for n, arg in enumerate(argv) if arg.isnumeric() and n!=dl_flag_index+1 and n!=opp_ai_int_flag_index+1 and n!=opp_dl_flag_index+1 ]
        if num:
            n = int(num[0]) if n>0 else 1
        if '-c' in argv:
            clear()
        if '-v' in argv:
            verbose = 1
        if '-vv' in argv:
            verbose = 2
        if '-vvv' in argv:
            verbose = 3
        if '-p' in argv:
            progress = True
        if '-g' in argv:
            suppress_output = False
        if '-h' in argv:
            heur = 'graphcut'
            heur_str = '_h'
        if '-h2' in argv:
            heur = 'geodesics'
            heur_str = '_h2'

    delimiter = ' '
    run_str = delimiter.join(argv)

    match opp_ai_int:
        case 0:
            opp_ai_level = 'Easy AI'
        case 1:
            opp_ai_level = 'Medium AI'
        case 2:
            opp_ai_level = 'custom AI version 1'
        case 3:
            opp_ai_level = 'custom AI version 2'
        case 4:
            opp_ai_level = 'custom AI version 3'
        case _:
            opp_ai_level = 'Easy AI'

    # RunGames batch parameters to output to terminal
    if is_unix:                                                                     # if Unix, print in color
        cprint(f'Running batch test via {argv[0]}, {n} times...', 'blue')
        cprint(f'Command line:', 'blue')
        cprint(f'  $ {run_str}', 'yellow')
        cprint(f'Setting Player search depth limit to {depth_limit}.', 'blue')      # if depth_limit:
        if heur:
            cprint(f'Applying advanced heuristics for Player AI.', 'blue')
        cprint(f'Setting Opponent AI to {opp_ai_level}.', 'blue')
        if opp_ai_int >1 and opp_depth_limit:
            cprint(f'Setting Opponent search depth limit to {opp_depth_limit}.', 'blue')
        if verbose == 1:
            cprint(f'Verbose mode.', 'green')
        if verbose == 2:
            cprint(f'Extra verbose mode.', 'green')
        print('')
    else:
        print(f'Running batch test via {argv[0]}, {n} times...')
        print(f'Command line:')
        print(f'  $ {run_str}')
        print(f'Setting Player search depth limit to {depth_limit}.')
        if heur:
            print(f'Applying advanced heuristics for Player AI.')
        print(f'Setting Opponent AI to {opp_ai_level}.')
        if opp_ai_int >1 and opp_depth_limit:
            print(f'Setting Opponent search depth limit to {opp_depth_limit}.')
        if verbose == 1:
            print(f'Verbose mode.')
        if verbose == 2:
            print(f'Extra verbose mode.')
        print('')


    # delete file if exists
    results_filename = f"batch_results{depth_str}{opp_ai_level_str}{opp_depth_str}{heur_str}.txt"
    if exists(results_filename):
      remove(results_filename)
        
    # initialize RunGames Class, call start_batch Method, get overall stats
    run_games = RunGames(n, verbose, progress, suppress_output, heur, depth_limit, opp_ai_int, opp_depth_limit, results_filename)
    total_time, run_times, rounds_list, run_success, player_wins = run_games.start_batch()
    total_moves = sum(rounds_list)
    avg_rounds = total_moves/n
    avg_move_time = total_time/total_moves
    try:
        win_rate = 100*player_wins/run_success
    except:
        pass
    print('> ... Done.')

    # write summary of batch run to file
    with open(results_filename, 'a') as f:
        f.write('------------------------------')

        f.write(f'\n{run_success} games ran successfully out of {n}.\n')
        if run_success != 0:
            f.write(f'Of those, Player won {player_wins} times, for a win rate of {win_rate:.2f}%.\n') #, 'green')
            error_str = 'complete'
        else:
            error_str = 'reach an error'

        f.write(f'Each run took an average of {total_time/n:.2f} seconds to {error_str}.\n')
        f.write(f'Max runtime: {max(run_times) :.2f} seconds; min runtime: {min(run_times):.2f} seconds.\n')
        f.write(f'The average game length is {avg_rounds:.1f} rounds.\n')
        f.write(f'Max game length: {max(rounds_list)} rounds; min game length: {min(rounds_list)} rounds.\n')
        f.write(f'The average player move time is {avg_move_time:.4f} seconds.\n') if total_moves!=n else None
        f.write(f'All processes took {total_time:.2f} seconds to {error_str}.\n')

    # read lines from file
    with open(results_filename, 'r') as f:
        batch_output = []
        lines = f.read().splitlines()
        started = False
        for line in lines:
            if '------------------------------' in line:
                started = True
            if started:
                batch_output.append(line)

    # print lines to screen
    for line in batch_output:
        if is_unix:                                                 # add color if Unix
            if run_success == 0 and 'successfully' in line:
                cprint(line, 'red')
            elif 'win rate' in line:
                if win_rate >= 75:
                    cprint(line, 'green')
                elif win_rate >= 60:
                    cprint(line, 'yellow')
                else:
                    cprint(line, 'red')
            elif 'move time' in line:
                cprint(line, 'blue')
            else:
                print(line)
        else:
            print(line)

if __name__ == "__main__":
    main()

