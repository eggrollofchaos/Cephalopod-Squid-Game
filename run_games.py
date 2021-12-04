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

'''
Helper script to run Game.py n times to test successful execution, win/loss, and gather statistics.
Outputs results to batch_results.txt

Usage:
$ python run_games.py [N] -v -c
[N] : number of processes to run
-v  : verbose output to terminal
-c  : clear terminal screen prior to running
'''

CLEAR = lambda: system('clear')
N = 100
VERBOSE = False

if '-v' in argv:
    VERBOSE = True
if '-c' in argv:
    CLEAR()
N = int([arg for arg in argv if arg.isnumeric()][0])

cprint(f'Running batch test on {argv[0]}, {N} times...\n', 'blue')
if VERBOSE:
    print()

if exists("batch_results.txt"):
  remove("batch_results.txt")

run_success = 0
player_wins = 0

start_batch = time()

for i in tqdm(range(1, N+1)):

    start_run = time()
    result = run(['python', 'Game.py', '-t'], capture_output=True)
    end_run = time()

    # animation to show progress indicator
    if not VERBOSE:
        tri = i % 3
        if tri == 1:
            print('> .  ', end='\r')
        elif tri == 2:
            print('> .. ', end='\r')
        else:
            print('> ...', end='\r')

    with open('batch_results.txt', 'a') as f:

        if result.returncode == 0:		# normal exit code
            win = f'Run {i}: Player Wins! Process completed in {end_run-start_run:.3f} seconds.\n'
            cprint(win, 'green') if VERBOSE else None
            f.write(win)
            run_success += 1            # increment number of successful runs
            player_wins += 1            # increment number of player wins

        elif result.returncode == 2:    # player has lost
            loss = f'Run {i}: Player Loses! Process completed in {end_run-start_run:.3f} seconds.\n'
            cprint(loss, 'red') if VERBOSE else None
            f.write(loss)
            run_success += 1            # increment number of successful runs

        else:                           # encountered runtime error
            error = f'Run {i}: Runtime error...\n'
            cprint(error, 'yellow') if VERBOSE else None
            f.write(error)
            stderr = str(result.stderr)
            search_from = len(stderr) - 200
            stderr_str = literal_eval("\"" + stderr[stderr.find('line', search_from) : ] + "\"")
            print(stderr_str) if VERBOSE else None
            f.write(stderr_str)
            f.write('\n')

end_batch = time()
run_time = end_batch - start_batch

print('> ... Done.')
print(f'\n{run_success} scripts ran successfully out of {N}.')
print(f'\nOf those, Player Won {player_wins} times.')
print(f'All processes took {run_time:.3f} seconds to complete.\n')
