import time
import numpy as np
import opfunu
from joblib import Parallel, delayed
from de import DifferentialEvolution
from cde import CDE
from jade import JADE
from shade import SHADE
from l_shade import L_SHADE
from l_shade_rsp import L_SHADE_RSP
from qde import QDE
from rl_hpsde import RL_HPSDE

MAX_FES_10 = 200000
MAX_FES_20 = 1000000
N_RUNS = 30

D = 10
FUNC = 'F32022'
POPULATION_SIZE = 50

# Differential Evolution
F = 0.8
CR = 0.8
MUTATION_TYPE = 'current-to-best'
CROSSOVER_TYPE = 'bin'

# CDE
STRAT_CONSTANT = 2
DELTA = 1/45

# JADE
ARCHIVE_SIZE = POPULATION_SIZE
P = 0.1
C = 0.1

# SHADE
MEMORY_SIZE = POPULATION_SIZE
ARCHIVE_SIZE = POPULATION_SIZE

# L-SHADE
MAX_POPULATION_SIZE = 18 * D
MIN_POPULATION_SIZE = 4 * D

# L-SHADE-RSP
MAX_POPULATION_SIZE = 18 * D
MIN_POPULATION_SIZE = 4 * D

# RL-HPSDE
NUM_STEPS = 200
STEP_SIZE = 10

N_JOBS = -1

funcs = opfunu.get_functions_by_classname(FUNC)
func = funcs[0](ndim=D)


def train_de(max_fes):
    de = DifferentialEvolution(D, func, POPULATION_SIZE, F, CR, MUTATION_TYPE, CROSSOVER_TYPE)
    best_score = de.train(max_fes)
    return best_score

def train_cde(max_fes):
    cde = CDE(D, func, POPULATION_SIZE, STRAT_CONSTANT, DELTA)
    best_score = cde.train(max_fes)
    return best_score

def train_jade(max_fes):
    jade = JADE(D, func, POPULATION_SIZE, ARCHIVE_SIZE, P, C)
    best_score = jade.train(max_fes)
    return best_score

def train_shade(max_fes):
    shade = SHADE(D, func, POPULATION_SIZE, MEMORY_SIZE, ARCHIVE_SIZE)
    best_score = shade.train(max_fes)
    return best_score

def train_l_shade(max_fes):
    l_shade = L_SHADE(D, func, MAX_POPULATION_SIZE, MIN_POPULATION_SIZE, max_fes, MEMORY_SIZE, ARCHIVE_SIZE)
    best_score = l_shade.train(max_fes)
    return best_score

def train_l_shade_rsp(max_fes):
    l_shade_rsp = L_SHADE_RSP(D, func, MAX_POPULATION_SIZE, MIN_POPULATION_SIZE, max_fes, MEMORY_SIZE, ARCHIVE_SIZE)
    best_score = l_shade_rsp.train(max_fes)
    return best_score

def train_qde(max_fes):
    qde = QDE(D, func, POPULATION_SIZE, MUTATION_TYPE)
    best_score = qde.train(max_fes)
    return best_score

def train_rl_hpsde(max_fes):
    rl_hpsde = RL_HPSDE(D, func, MAX_POPULATION_SIZE, MIN_POPULATION_SIZE, max_fes, MEMORY_SIZE, NUM_STEPS, STEP_SIZE)
    best_score = rl_hpsde.train(max_fes)
    return best_score


if __name__ == '__main__':
    if D == 10:
        max_fes = MAX_FES_10
    elif D == 20:
        max_fes = MAX_FES_20
    else:
        max_fes = MAX_FES_10

    start = time.perf_counter()
    scores = Parallel(n_jobs=N_JOBS)(delayed(train_l_shade_rsp)(max_fes) for _ in range(N_RUNS))
    end = time.perf_counter()

    scores = np.array(scores)
    print("Best: ", np.min(scores))
    print("Worst: ", np.max(scores))
    print("Median: ", np.median(scores))
    print("Mean: ", np.mean(scores))
    print("Std: ", np.std(scores))
    print("Elapsed time: ", end - start, " seconds")
