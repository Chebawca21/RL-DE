import time
import numpy as np
from joblib import Parallel, delayed
from de import DifferentialEvolution

MAX_FES_10 = 200000
MAX_FES_20 = 1000000
N_RUNS = 30

D = 10
FUNC_NUM = 3
POPULATION_SIZE = 50
F = 0.8
CR = 0.8
MUTATION_TYPE = 'current-to-best'
CROSSOVER_TYPE = 'bin'

N_JOBS = -1


def train_de(max_fes):
    de = DifferentialEvolution(D, FUNC_NUM, POPULATION_SIZE, F, CR, MUTATION_TYPE, CROSSOVER_TYPE)
    for _ in range(int(max_fes / (POPULATION_SIZE * 2))):
        if de.func_evals + (2 * POPULATION_SIZE) > max_fes:
            break
        de.step()
    return de.best_score


if __name__ == '__main__':
    if D == 10:
        max_fes = MAX_FES_10
    elif D == 20:
        max_fes = MAX_FES_20
    else:
        max_fes = 0

    start = time.perf_counter()
    scores = Parallel(n_jobs=N_JOBS)(delayed(train_de)(max_fes) for _ in range(N_RUNS))
    end = time.perf_counter()

    scores = np.array(scores)
    print("Best: ", np.min(scores))
    print("Worst: ", np.max(scores))
    print("Median: ", np.median(scores))
    print("Mean: ", np.mean(scores))
    print("Std: ", np.std(scores))
    print("Elapsed time: ", end - start, " seconds")
