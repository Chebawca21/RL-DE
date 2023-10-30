import time
import numpy as np
from joblib import Parallel, delayed
from de import DifferentialEvolution
from jade import JADE
from shade import SHADE

MAX_FES_10 = 200000
MAX_FES_20 = 1000000
N_RUNS = 30

D = 10
FUNC_NUM = 8
POPULATION_SIZE = 50

# Differential Evolution
F = 0.8
CR = 0.8
MUTATION_TYPE = 'current-to-best'
CROSSOVER_TYPE = 'bin'

# JADE
ARCHIVE_SIZE = POPULATION_SIZE
P = 0.1
C = 0.1

# SHADE
MEMORY_SIZE = POPULATION_SIZE
ARCHIVE_SIZE = POPULATION_SIZE

N_JOBS = -1


def train_de(max_fes):
    de = DifferentialEvolution(D, FUNC_NUM, POPULATION_SIZE, F, CR, MUTATION_TYPE, CROSSOVER_TYPE)
    for _ in range(int(max_fes / (POPULATION_SIZE * 2))):
        if de.func_evals + (2 * POPULATION_SIZE) > max_fes:
            break
        de.step()
    return de.best_score

def train_jade(max_fes):
    jade = JADE(D, FUNC_NUM, POPULATION_SIZE, ARCHIVE_SIZE, P, C)
    for _ in range(int(max_fes / (POPULATION_SIZE * 2))):
        if jade.func_evals + (2 * POPULATION_SIZE) > max_fes:
            break
        jade.step()
    return jade.best_score

def train_shade(max_fes):
    shade = SHADE(D, FUNC_NUM, POPULATION_SIZE, MEMORY_SIZE, ARCHIVE_SIZE)
    for _ in range(int(max_fes / (POPULATION_SIZE * 2))):
        if shade.func_evals + (2 * POPULATION_SIZE) > max_fes:
            break
        shade.step()
    return shade.best_score


if __name__ == '__main__':
    if D == 10:
        max_fes = MAX_FES_10
    elif D == 20:
        max_fes = MAX_FES_20
    else:
        max_fes = 0

    start = time.perf_counter()
    scores = Parallel(n_jobs=N_JOBS)(delayed(train_shade)(max_fes) for _ in range(N_RUNS))
    end = time.perf_counter()

    scores = np.array(scores)
    print("Best: ", np.min(scores))
    print("Worst: ", np.max(scores))
    print("Median: ", np.median(scores))
    print("Mean: ", np.mean(scores))
    print("Std: ", np.std(scores))
    print("Elapsed time: ", end - start, " seconds")
