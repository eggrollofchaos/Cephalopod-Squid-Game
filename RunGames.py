# gc2950
# mhr2145
# wax1
from sys import argv
from subprocess import run
from time import time, sleep
from os import system, remove
from os.path import exists
from ast import literal_eval
from termcolor import cprint
from tqdm import tqdm

class RunGames(object):
    '''
    Helper script to run Game.py n times to test successful execution, win/loss, and gather statistics.
    Outputs results to batch_results.txt

    Usage:
    $ python RunGames.py [n] -v -c -p -g -d [depth_limit] -a [opponent_depth_limit]
    [N] : number of processes to run
    -v  : verbose output to terminal
    -c  : clear terminal screen prior to running
    -p  : show progress bars
    -g  : show game output (note: very verbose)
    -d  : set player search depth limit of [depth_limit], min 1, default 4
    -d  : set opponent search depth limit of [opp_depth_limit], min 1, default 2
    '''

    def __init__(self, n, verbose, progress, suppress_output, heur, depth_limit, opp_depth_limit, results_filename):
        self.n = n
        self.verbose = verbose
        self.progress = progress
        self.suppress_output = suppress_output
        self.heur = heur
        self.depth_limit = depth_limit
        self.opp_depth_limit = opp_depth_limit
        self.run_success = 0
        self.player_wins = 0
        self.filename = results_filename


    def start_batch(self):
        # begin batch run
        start_batch = time()
        run_times = []
        rounds_list = []

        if not self.suppress_output:                    # to help find game beginning/end
            cprint('\n\n', on_color='on_white')

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

        if not self.suppress_output:                    # to help find game beginning/end
            cprint('\n\n', on_color='on_white')


        end_batch = time()
        total_time = end_batch - start_batch

        return total_time, run_times, rounds_list, self.run_success, self.player_wins

    def __run_process(self, it):
        run_arg_list = ['python', 'Game.py', '-t', '-d', str(self.depth_limit), '-a', str(self.opp_depth_limit)]
        run_arg_list3 = ['python3', 'Game.py', '-t', '-d', str(self.depth_limit), '-a', str(self.opp_depth_limit)]
        if self.heur == 'graphcut':
            run_arg_list.append('-h')
            run_arg_list3.append('-h')
        if self.heur == 'geodesics':
            run_arg_list.append('-h2')
            run_arg_list3.append('-h2')
        if self.verbose:
            run_arg_list.append('-v')
            run_arg_list3.append('-v')

        start_run = time()
        try:
            # result = run(['python', 'Game.py', '-t', '-d', str(self.depth_limit), '-a', str(self.opp_depth_limit)], capture_output=self.suppress_output)
            result = run(run_arg_list, capture_output=self.suppress_output)
        except:
            # result = run(['python3', 'Game.py', '-t', '-d', str(self.depth_limit), '-a', str(self.opp_depth_limit)], capture_output=self.suppress_output)
            result = run(run_arg_list3, capture_output=self.suppress_output)

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

        # animation to show progress indicator if self.verbose is False
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
                cprint('\n\n' + win, on_color='on_green') if self.verbose else None
                f.write(win)
                self.run_success += 1           # increment number of successful runs
                self.player_wins += 1           # increment number of player wins
                print()

            elif winning_player == 2:       # player has lost
                loss = f'Run {it}: Player Loses! Process completed in {run_time:.3f} seconds.\n'
                cprint('\n\n' + loss, on_color='on_red') if self.verbose else None
                f.write(loss)
                self.run_success += 1           # increment number of successful runs
                print()

            else:                               # encountered runtime error, exit code was 1
                error = f'Run {it}: Runtime error...\n'
                cprint('\n\n' + error, 'yellow') if self.verbose else None
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
    clear = lambda: system('clear')
    n = 100
    verbose = False
    progress = False
    suppress_output = True
    heur = False
    depth_limit = 0
    opp_depth_limit = 0
    depth_str = ''
    opp_depth_str = ''
    heur_str = ''
    dl_flag_index = -1
    opp_dl_flag_index = -1

    if len(argv)>1:
        dl_flag_index = 0
        if '-d' in argv:
            try:
                dl_flag_index = argv.index('-d')
                depth_limit = argv[dl_flag_index+1]
                depth_str = f'_d_{depth_limit}'
            except:
                pass
        if '-a' in argv:
            try:
                opp_dl_flag_index = argv.index('-a')
                opp_depth_limit = int(argv[opp_dl_flag_index+1])
                opp_depth_str = f'_a_{opp_depth_limit}'
            except:
                pass
        num = [arg for n, arg in enumerate(argv) if arg.isnumeric() and n!=dl_flag_index+1 and n!=opp_dl_flag_index+1]
        if num:
            n = int(num[0]) if n>0 else 1
        if '-v' in argv:
            verbose = True
        if '-c' in argv:
            clear()
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

    cprint(f'Running batch test on {argv[0]}, {n} times...', 'blue')
    if depth_limit:
        cprint(f'Setting Player search depth limit to {depth_limit}.', 'blue')
    if opp_depth_limit:
        cprint(f'Setting Opponent search depth limit to {opp_depth_limit}.', 'blue')
    if heur:
        cprint(f'Applying advanced heuristics.\n', 'blue')

    results_filename = f"batch_results{depth_str}{opp_depth_str}{heur_str}.txt"
    if exists(results_filename):
      remove(results_filename)

    run_games = RunGames(n, verbose, progress, suppress_output, heur, depth_limit, opp_depth_limit, results_filename)
    total_time, run_times, rounds_list, run_success, player_wins = run_games.start_batch()
    total_moves = sum(rounds_list)
    avg_rounds = total_moves/n
    avg_move_time = total_time/total_moves
    try:
        win_rate = 100*player_wins/run_success
    except:
        pass

    print('> ... Done.')
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

    with open(results_filename, 'r') as f:
        batch_output = []
        lines = f.read().splitlines()
        started = False
        for line in lines:
            if '------------------------------' in line:
                started = True
            if started:
                batch_output.append(line)

    for line in batch_output:
        if run_success == 0 and 'successfully' in line:
            cprint(line, 'red')
        elif 'win rate' in line:
            if win_rate >= 75:
                cprint(line, 'green')
            else:
                cprint(line, 'red')
        elif 'move time' in line:
            cprint(line, 'blue')
        else:
            print(line)

if __name__ == "__main__":
    main()

