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

class Run_Games():
    '''
    Helper script to run Game.py n times to test successful execution, win/loss, and gather statistics.
    Outputs results to batch_results.txt

    Usage:
    $ python run_games.py [N] -v -c
    [N] : number of processes to run
    -v  : verbose output to terminal
    -c  : clear terminal screen prior to running
    '''
    def __init__(self, n, progress, verbose):
        self.n = n
        self.progress = progress
        self.verbose = verbose
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

        else:
            for it in range(1, self.n+1):             # no progress bars
                run_time, run_moves = self.__run_process(it)

        run_times.append(run_time)
        total_moves += run_moves

        end_batch = time()
        total_time = end_batch - start_batch

        return total_time, total_moves, self.run_success, self.player_wins

    def __run_process(self, it):
        run_moves = 0
        start_run = time()
        try:
            result = run(['python', 'Game.py', '-t'], capture_output=True)
        except:
            result = run(['python3', 'Game.py', '-t'], capture_output=True)

        # result = run(['python', 'Game.py', '-t'])
        end_run = time()
        run_time = end_run-start_run
        stdout = str(result.stdout)
        winning_player = result.returncode
        print(winning_player)
        print(stdout[-200:])
        # print(stdout[stdout.rfind('!')+3:stdout.rfind('\\')])
        # run_moves = int(stdout[stdout.rfind('!')+3:stdout.rfind('\\')])


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
                search_from = len(stderr) - 200
                stderr_str = literal_eval("\"" + stderr[stderr.find('line', search_from) : ] + "\"")
                print(stderr_str) if self.verbose else None
                f.write(stderr_str)
                f.write('\n')

        return run_time, run_moves

def main():
    clear = lambda: system('clear')
    n = 100
    verbose = False
    progress = False

    if len(argv)>1:
        num = [arg for arg in argv if arg.isnumeric()]
        if num:
            n = int(num[0])
        if '-c' in argv:
            clear()
        if '-p' in argv:
            progress = True
        if '-v' in argv:
            verbose = True

    cprint(f'Running batch test on {argv[0]}, {n} times...\n', 'blue')

    run_games = Run_Games(n, progress, verbose)
    total_time, total_moves, run_success, player_wins = run_games.start_batch()

    print('> ... Done.')
    print(f'\n{run_success} scripts ran successfully out of {n}.')
    print(f'Of those, Player won {player_wins} times, for a win rate of {100*player_wins/run_success:.2f}%.')
    print(f'Each run took an average of {total_time/n:.3f} seconds to complete.')
    print(f'The average game length is {total_moves/n:.3f} rounds.')
    print(f'All processes took {total_time:.3f} seconds to complete.\n')

if __name__ == "__main__":
    main()

