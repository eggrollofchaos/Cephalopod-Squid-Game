
from sys import argv
from subprocess import run
from time import time
from os import system, remove
from os.path import exists
from ast import literal_eval
from termcolor import cprint
import tqdm

clear = lambda: system('clear')
clear()

n = 100
VERBOSE = False

if len(argv) > 1:
    n = int(argv[1])
if len(argv) > 2:
    VERBOSE = argv[2]

cprint(f'Running batch test on {argv[0]}, {n} times...\n', 'blue')
if VERBOSE == '-v':
    VERBOSE = True
    print()

if exists("batch_results.txt"):
  remove("batch_results.txt")

run_success = 0
player_wins = 0

start_batch = time()

for i in tqdm.tqdm(range(1, n+1)):

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
print(f'\n{run_success} scripts ran successfully out of {n}.')
print(f'\nOf those, Player Won {player_wins} times.')
print(f'All processes took {run_time:.3f} seconds to complete.\n')
