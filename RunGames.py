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
        total_moves = 0

        if self.progress:
            for it in tqdm(range(1, self.n+1)):       # show progress bars
                run_time, run_moves = self.__run_process(it)
                run_times.append(run_time)
                total_moves += run_moves

        else:
            for it in range(1, self.n+1):             # no progress bars
                run_time, run_moves = self.__run_process(it)
                run_times.append(run_time)
                total_moves += run_moves


        end_batch = time()
        total_time = end_batch - start_batch

        return total_time, run_times, total_moves, self.run_success, self.player_wins

    def __run_process(self, it):
        start_run = time()
        try:
            # result = run(['python', 'Game.py', '-t', '-d', str(self.depth_limit)]) #, capture_output=True)
            result = run(['python', 'Game.py', '-t', '-d', str(self.depth_limit)], capture_output=self.suppress_output)
        except:
            result = run(['python3', 'Game.py', '-t', '-d', str(self.depth_limit)], capture_output=self.suppress_output)

        # result = run(['python', 'Game.py', '-t'])
        end_run = time()
        run_time = end_run-start_run
        stdout = str(result.stdout)
        winning_player = result.returncode
        # print(stdout[stdout.rfind(':')+2:stdout.rfind('\\')])
        try:
            run_moves = int(stdout[stdout.rfind(':')+2:stdout.rfind('\\')])
        except:
            run_moves = 0

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

            if winning_player == 0:         # normal exit code is 0
                win = f'Run {it}: Player Wins! Process completed in {run_time:.3f} seconds.\n'
                cprint(win, 'green') if self.verbose else None
                f.write(win)
                self.run_success += 1            # increment number of successful runs
                self.player_wins += 1            # increment number of player wins

            elif winning_player == 2:       # player has lost
                loss = f'Run {it}: Player Loses! Process completed in {run_time:.3f} seconds.\n'
                cprint(loss, 'red') if self.verbose else None
                f.write(loss)
                self.run_success += 1            # increment number of successful runs

            else:                           # encountered runtime error, exit code == 1
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

        return run_time, run_moves

def main():
    clear = lambda: system('clear')
    n = 100
    verbose = False
    progress = False
    suppress_output = True
    depth_limit = 0

    if len(argv)>1:
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
        cprint(f'Running batch test on {argv[0]}, {n} times, custom search depth limit of {depth_limit}...\n', 'blue')
    else:
        cprint(f'Running batch test on {argv[0]}, {n} times...\n', 'blue')

    run_games = RunGames(n, verbose, progress, suppress_output, depth_limit)
    total_time, run_times, total_moves, run_success, player_wins = run_games.start_batch()

    print('> ... Done.')
    print(f'\n{run_success} scripts ran successfully out of {n}.')
    print(f'Of those, Player won {player_wins} times, for a win rate of {100*player_wins/run_success:.2f}%.')
    print(f'Each run took an average of {total_time/n:.2f} seconds to complete.')
    print(f'Max runtime: {max(run_times) :.2f} seconds; min runtime: {min(run_times):.2f} seconds.')
    print(f'The average game length is {total_moves/n:.1f} rounds.')
    print(f'All processes took {total_time:.2f} seconds to complete.\n')

if __name__ == "__main__":
    main()

