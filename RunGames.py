# gc2950
# mhr2145
# wax1
from sys import argv
from subprocess import run
from time import time
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
    $ python run_games.py [n] -v -c -p -g -d [depth_limit]
    [N] : number of processes to run
    -v  : verbose output to terminal
    -c  : clear terminal screen prior to running
    -p  : show progress bars
    -g  : show game output (note: very verbose)
    -d  : set search depth limit of [depth_limit], min 1, default 4
    '''

    def __init__(self, n, verbose, progress, suppress_output, depth_limit):
        self.n = n
        self.verbose = verbose
        self.progress = progress
        self.suppress_output = suppress_output
        self.depth_limit = depth_limit
        self.run_success = 0
        self.player_wins = 0


    def start_batch(self):
        if exists("batch_results.txt"):
          remove("batch_results.txt")

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
        start_run = time()
        try:
            result = run(['python', 'Game.py', '-t', '-d', str(self.depth_limit)], capture_output=self.suppress_output)
        except:
            result = run(['python3', 'Game.py', '-t', '-d', str(self.depth_limit)], capture_output=self.suppress_output)

        end_run = time()
        run_time = end_run-start_run
        stdout = str(result.stdout)
        returncode = result.returncode
        if returncode == 1:                     # encountered error, exit code = 1
            winning_player = -1
            rounds = 0
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

        with open('batch_results.txt', 'a') as f:

            # if winning_player == 0:         # normal exit code is 0
            if winning_player == 1:             # normal exit code is 0
                win = f'Run {it}: Player Wins! Process completed in {run_time:.3f} seconds.\n'
                cprint(win, 'green') if self.verbose else None
                f.write(win)
                self.run_success += 1           # increment number of successful runs
                self.player_wins += 1           # increment number of player wins

            elif winning_player == 2:       # player has lost
                loss = f'Run {it}: Player Loses! Process completed in {run_time:.3f} seconds.\n'
                cprint(loss, 'red') if self.verbose else None
                f.write(loss)
                self.run_success += 1           # increment number of successful runs

            else:                               # encountered runtime error, exit code was 1
                error = f'Run {it}: Runtime error...\n'
                cprint(error, 'yellow') if self.verbose else None
                f.write(error)
                stderr = str(result.stderr)
                # search_from = len(stderr) - 200
                # stderr_str = literal_eval("\"" + stderr[stderr.find('line', search_from) : ] + "\"")
                stderr_str = literal_eval("\"" + stderr[stderr.rfind('line') : ] + "\"")
                # stderr_str = literal_eval("\"" + stderr + "\"")
                print(stderr_str) if self.verbose else None
                f.write(stderr_str)
                f.write('\n')

        return run_time, rounds


def main():
    clear = lambda: system('clear')
    n = 100
    verbose = False
    progress = False
    suppress_output = True
    depth_limit = 0

    if len(argv)>1:
        dl_flag_index = 0
        if '-d' in argv:
            try:
                dl_flag_index = argv.index('-d')
                depth_limit = argv[dl_flag_index+1]
            except:
                pass
        num = [arg for n, arg in enumerate(argv) if arg.isnumeric() and n != dl_flag_index+1]
        if num:
            n = int(num[0])
        if '-v' in argv:
            verbose = True
        if '-c' in argv:
            clear()
        if '-p' in argv:
            progress = True
        if '-g' in argv:
            suppress_output = False

    if depth_limit:
        cprint(f'Running batch test on {argv[0]}, {n} times, enforcing search depth limit of {depth_limit}...\n', 'blue')
    else:
        cprint(f'Running batch test on {argv[0]}, {n} times...\n', 'blue')

    run_games = RunGames(n, verbose, progress, suppress_output, depth_limit)
    total_time, run_times, rounds_list, run_success, player_wins = run_games.start_batch()
    total_moves = sum(rounds_list)
    avg_rounds = total_moves/n
    avg_move_time = total_time/total_moves

    print('> ... Done.')
    print(f'\n{run_success} scripts ran successfully out of {n}.')
    if run_success != 0:
        cprint(f'Of those, Player won {player_wins} times, for a win rate of {100*player_wins/run_success:.2f}%.', 'green')
        error_str = 'complete'
    else:
        error_str = 'reach an error'
    print(f'Each run took an average of {total_time/n:.2f} seconds to {error_str}.')
    print(f'Max runtime: {max(run_times) :.2f} seconds; min runtime: {min(run_times):.2f} seconds.')
    print(f'The average game length is {avg_rounds:.1f} rounds.')
    print(f'Max game length: {max(rounds_list) :.2f} rounds; min game length: {min(rounds_list):.2f} rounds.')
    cprint(f'The average player move time is {avg_move_time:.2f} seconds.', 'blue')
    print(f'All processes took {total_time:.2f} seconds to {error_str}.\n')

if __name__ == "__main__":
    main()

