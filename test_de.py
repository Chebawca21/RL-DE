import time
import numpy as np
import opfunu
from joblib import Parallel, delayed
from models.de import DifferentialEvolution
from models.cde import CDE
from models.jade import JADE
from models.shade import SHADE
from models.l_shade import L_SHADE
from models.l_shade_rsp import L_SHADE_RSP
from models.qde import QDE
from models.rl_hpsde import RL_HPSDE
from config import get_model_parameters

MAX_FES_10 = 2000
MAX_FES_20 = 1000000
N_RUNS = 5

D = 10
FUNC = 'F32022'

N_JOBS = -1


def train_de(max_fes):
    params = get_model_parameters("de", D, FUNC)
    de = DifferentialEvolution(**params)
    best_score = de.train(max_fes)
    return best_score

def train_cde(max_fes):
    params = get_model_parameters("cde", D, FUNC)
    cde = CDE(**params)
    best_score = cde.train(max_fes)
    return best_score

def train_jade(max_fes):
    params = get_model_parameters("jade", D, FUNC)
    jade = JADE(**params)
    best_score = jade.train(max_fes)
    return best_score

def train_shade(max_fes):
    params = get_model_parameters("shade", D, FUNC)
    shade = SHADE(**params)
    best_score = shade.train(max_fes)
    return best_score

def train_l_shade(max_fes):
    params = get_model_parameters("l-shade", D, FUNC)
    l_shade = L_SHADE(**params)
    best_score = l_shade.train(max_fes)
    return best_score

def train_l_shade_rsp(max_fes):
    params = get_model_parameters("l-shade-rsp", D, FUNC)
    l_shade_rsp = L_SHADE_RSP(**params)
    best_score = l_shade_rsp.train(max_fes)
    return best_score

def train_qde(max_fes):
    params = get_model_parameters("qde", D, FUNC)
    qde = QDE(**params)
    best_score = qde.train(max_fes)
    return best_score

def train_rl_hpsde(max_fes):
    params = get_model_parameters("rl-hpsde", D, FUNC)
    rl_hpsde = RL_HPSDE(**params)
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
    scores = Parallel(n_jobs=N_JOBS)(delayed(train_rl_hpsde)(max_fes) for _ in range(N_RUNS))
    end = time.perf_counter()

    scores = np.array(scores)
    print("Best: ", np.min(scores))
    print("Worst: ", np.max(scores))
    print("Median: ", np.median(scores))
    print("Mean: ", np.mean(scores))
    print("Std: ", np.std(scores))
    print("Elapsed time: ", end - start, " seconds")
