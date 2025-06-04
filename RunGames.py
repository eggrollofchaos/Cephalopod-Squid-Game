"""
RunGames Class batch module.
Authored by WAX.
"""
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
    Outputs results to "batch_results[ ... ].txt" with various additional filename decorations based on arguments used.

    Usage:
    $ python3 RunGames.py [n] -c -p -v|-vv|-vvv -g -h -d [depth_limit] -oa [opp_ai_level] -od [opp_depth_limit] -m [comment]
     n  : number of processes to run, default 100
    -c  : clear terminal screen prior to running
    -p  : show progress bars
    -v  : verbose output to terminal: -v shows additional game info, -vv shows all AI steps, -vvv shows detailed Expectiminimax search steps
    -g  : show game output (note: can be very verbose)
    -h  : enable advanced heuristics
    -d  : set Player search depth limit; min 1, default 4
    -oa : set Opponent AI level of {-2,-1,0,1,2,3}; default = 0 (Medium); level = 9 indicates Human Opponent, however batch run is intended for AI
    -od : set Opponent AI search depth limit; min 1, default = 2, only applicable if Opponent AI level in {1,2,3}
    -m  : add a comment to the log file, must enclose in quotes
    
    Examples:
    $ python3 RunGames.py 100 -c -v -p -g -h -d 4 -oa 0 -od 1
    $ python3 RunGames.py 10 -c -p -h -d 6 -oa 3 -od 2 -m "Trying something new"
    
    See main() for additional notes on flags and arguments.
    '''

    def __init__(self, results_filename, n, progress, verbose, suppress_output, heur, depth_limit, opp_ai_int, opp_depth_limit, comment):
        self.filename = results_filename
        self.n = n
        self.progress = progress
        self.verbose = verbose
        self.suppress_output = suppress_output
        self.heur = heur
        self.depth_limit = depth_limit
        self.opp_ai_int = opp_ai_int
        self.opp_depth_limit = opp_depth_limit
        self.comment = comment
        self.run_success = 0
        self.player_wins = 0
        self.run_arg_list = []
        self.run_arg_list3 = []


    def start_batch(self):
        # begin batch run
        # print('In start_batch')

        # build command line arguments for Game.py
        # note that -od (set Opponent AI search depth) is always passed, though ignored by Game.py if -oa not in {1,2,3}
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

        # write Game.py command line statement to file for clarity
        delimiter = ' '
        self.run_arg_list3_str = delimiter.join(self.run_arg_list3)
        # print(f'Filename = {self.filename}')
        with open(self.filename, 'a') as f:
            # print(self.run_arg_list3)
            f.write('Batch command line:\n')
            f.write('  ' + self.run_arg_list3_str + '\n')
            f.write('Game command line:\n')
            f.write('  ' + self.run_arg_list3_str + '\n')
            f.write('\n')

        # write optional comment to file
        if self.comment:
            with open(self.filename, 'a') as f:
                f.write(self.comment)
                f.write('\n\n')

        # run processes and capture elapsed time
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
                
        # start individual process, capture elapsed time
        start_run = time()

        try:
            result = run(self.run_arg_list3, capture_output=self.suppress_output)
        except:
            result = run(self.run_arg_list, capture_output=self.suppress_output)

        end_run = time()
        run_time = end_run-start_run
        # stdout = str(result.stdout)             # don't need this anymore
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

        # animation to show progress indicator for entire batch, if self.verbose is 0
        if not self.verbose:
            tri = it % 3
            if tri == 1:
                print('> .  ', end='\r')
            elif tri == 2:
                print('> .. ', end='\r')
            else:
                print('> ...', end='\r')

        # write the outcome of this run: win, loss, or error
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

            elif winning_player == 2:       # Player has lost
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

# main method, called when running this script directly
def main():
    clear = lambda: system('clear')         # pretty cool way to clear the output, should look into more
    n = 100                                 # number of iterations (processes)
    verbose = 0                             # verbosity level
    progress = False                        # show progress through all tests
    suppress_output = True                  # this is passed to capture_output argument; counterintuitively, adding '-g' means supress_output = False
    heur = False                            # advanced heuristics disabled by default
    depth_limit = 4                         # Player AI search depth defaults to 4
    depth_str = ''                          # part of results filename
    opp_ai_int = 0                          # Opponent AI level defaults 0 (Medium AI); if -1 or 0, depth limit won't apply
    opp_ai_level = 'Medium AI'              # Opponent AI level default to Medium AI
    opp_ai_level_str = ''                   # part of results filename
    opp_depth_limit = 2                     # Opponent AI search depth defaults to 2, only applies if AI level in {2,3,4}
    opp_depth_str = ''                      # part of results filename
    heur_str = ''                           # part of results filename
    comment = False                         # default is no comment
    com_str = ''                            # part of results filename

    dl_flag_index = -1
    opp_dl_flag_index = -1
    opp_ai_int_flag_index = -1
    com_flag_index = -1

    # rebuild original command line statement
    delimiter = ' '
    run_str = delimiter.join(argv)

    # process command line arguments
    if len(argv)>1:
        
        if '-d' in argv:                        # set Player AI search depth
            try:
                dl_flag_index = argv.index('-d')
                depth_limit = argv[dl_flag_index+1]
            except:
                pass
        depth_str = f'_d_{depth_limit}'
        
        if '-oa' in argv:                       # set Opponent AI level
            try:
                opp_ai_int_flag_index = argv.index('-oa')
                opp_ai_int = int(argv[opp_ai_int_flag_index+1])
            except:
                pass
        opp_ai_level_str = f'_oa_{opp_ai_int}'
        
        if opp_ai_int > 0 and '-od' in argv:    # set Opponent AI search depth
            try:
                opp_dl_flag_index = argv.index('-od')
                opp_depth_limit = int(argv[opp_dl_flag_index+1])
            except:
                pass
            opp_depth_str = f'_od_{opp_depth_limit}'
        
        if '-m' in argv:                        # set Comment
            try:
                com_flag_index = argv.index('-m')
                comment = argv[com_flag_index+1]
            except:
                pass
            com_str = f'_m'

        # a fancy way to capture a number anywhere in the command line statement as the number of iterations to run
        num = [ arg for arg_idx, arg in enumerate(argv) if arg.isnumeric()
            and arg_idx!=dl_flag_index+1
            and arg_idx!=opp_ai_int_flag_index+1
            and arg_idx!=opp_dl_flag_index+1 ]
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
        if '-g' in argv:                # this means to show stdout and stderr in terminal, instead of capturing
            suppress_output = False
        if '-h' in argv:
            heur = 'graphcut'
            heur_str = '_h'
        if '-h2' in argv:
            heur = 'geodesics'
            heur_str = '_h2'

    # get Opponent AI level text from argument
    match opp_ai_int:
        case -1:
            opp_ai_level = 'Easy AI'
        case 0:
            opp_ai_level = 'Medium AI'
        case 1:
            opp_ai_level = 'Custom AI version 1'
        case 2:
            opp_ai_level = 'Custom AI version 2'
        case 3:
            opp_ai_level = 'Custom AI version 3'
        case 9:
            opp_ai_level = 'Human player'
        case _:
            opp_ai_int = 0                                      # default
            opp_ai_level = 'Medium AI'

    # output RunGames.py batch parameters to terminal
    if is_unix:                                                                     # if Unix, print in color
        cprint(f'Running batch test via {argv[0]}, {n} times...', 'blue')
        cprint('Command line:', 'blue')
        cprint(f'  $ {run_str}', 'yellow')
        cprint(f'Setting Player AI search depth limit to {depth_limit}.', 'blue')      # if depth_limit:
        if heur:
            cprint('Applying advanced heuristics for Player AI.', 'blue')
        
        # logic for Opponent AI & depth level 
        cprint(f'Setting Opponent to {opp_ai_level}.', 'blue')
        if opp_ai_int > 0 and opp_ai_int != 9 and opp_depth_limit:                  # if above Easy/Medium AI and not Human Opponent
            cprint(f'Setting Opponent AI search depth limit to {opp_depth_limit}.', 'blue')
        
        if verbose == 1:
            cprint('Verbose mode.', 'green')
        if verbose == 2:
            cprint('Extra verbose mode.', 'green')
        if verbose == 3:
            cprint('Extra verbose + trace mode.', 'green')
        print('')
    else:
        print(f'Running batch test via {argv[0]}, {n} times...')
        print('Command line:')
        print(f'  $ {run_str}')
        print(f'Setting Player AI search depth limit to {depth_limit}.')
        if heur:
            print('Applying advanced heuristics for Player AI.')

        # logic for Opponent AI & depth level 
        print(f'Setting Opponent to {opp_ai_level}.')
        if opp_ai_int > 0 and opp_ai_int != 9 and opp_depth_limit:                  # if above Easy/Medium AI and not Human Opponent
            print(f'Setting Opponent AI search depth limit to {opp_depth_limit}.')

        if verbose == 1:
            print('Verbose mode.')
        if verbose == 2:
            print('Extra verbose mode.')
        if verbose == 3:
            print('Extra verbose + trace mode.')
        print('')

    # delete log file if exists
    results_filename = f"batch_results{depth_str}{opp_ai_level_str}{opp_depth_str}{heur_str}{com_str}.txt"
    # print(results_filename)
    if exists(results_filename):
        # print('Deleting existing file')
        remove(results_filename)

    # output RunGames.py batch parameters to file
    with open(results_filename, 'a') as f:
        f.write(f'Running batch test via {argv[0]}, {n} times...\n')
        f.write(f'Setting Player search depth limit to {depth_limit}.\n')
        if heur:
            f.write('Applying advanced heuristics for Player AI.\n')

        # logic for Opponent AI & depth level 
        f.write(f'Setting Opponent to {opp_ai_level}.\n')
        if opp_ai_int > 0 and opp_ai_int != 9 and opp_depth_limit:                  # if above Easy/Medium AI and not Human Opponent
            f.write(f'Setting Opponent AI search depth limit to {opp_depth_limit}.\n')

        if verbose == 1:
            f.write('Verbose mode.\n')
        if verbose == 2:
            f.write('Extra verbose mode.\n')
        if verbose == 3:
            f.write('Extra verbose + trace mode.\n')
        f.write('')
        
    # initialize RunGames Class, call `start_batch` Method, get overall stats
    run_games = RunGames(results_filename, n, progress, verbose, suppress_output, heur, depth_limit, opp_ai_int, opp_depth_limit, comment)
    total_time, run_times, rounds_list, run_success, player_wins = run_games.start_batch()
    total_moves = sum(rounds_list)
    avg_rounds = total_moves/n
    avg_move_time = total_time/total_moves
    try:
        win_rate = 100*player_wins/run_success
    except:
        pass
    print('> ... Done.')

    # write summary of batch run to results (log) file
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

    # read log lines from file
    with open(results_filename, 'r') as f:
        batch_output = []
        lines = f.read().splitlines()
        started = False
        for line in lines:
            if '------------------------------' in line:
                started = True
            if started:
                batch_output.append(line)

    # print log lines to screen
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

