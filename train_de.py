import time
import numpy as np
import pandas as pd
import opfunu
from joblib import Parallel, delayed
from de import DifferentialEvolution
from cde import CDE
from jade import JADE
from shade import SHADE
from l_shade import L_SHADE
from l_shade_rsp import L_SHADE_RSP
from qde import QDE

MAX_FES_10 = 200000
MAX_FES_20 = 1000000
MAX_FES_OTHER = 10000
MIN_ERROR = 10 ** (-8)
N_RUNS = 30

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

# L_SHADE
L_SHADE_MEMORY_SIZE = 5
R_N_INIT = 20
R_ARC = 2
MIN_POPULATION_SIZE = 4

# L_SHADE_RSP
MAX_POPULATION_SCALAR = 75

N_JOBS = -1

def get_seed(problem_size, func_num, runs, run_id):
    with open("SO_BO/input_data/Rand_Seeds.txt") as f:
        lines = f.read().splitlines()
    seed_id = (problem_size // 10 * func_num * runs + run_id) - runs
    seed_id = seed_id % 1000
    return int(float(lines[seed_id].strip()))

def count_fes(D, k, max_fes):
    return D ** (k/5 - 3) * max_fes

def get_de(D, func, max_fes, model='de'):
    if model == 'de':
        de = DifferentialEvolution(D, func, POPULATION_SIZE, F, CR, MUTATION_TYPE, CROSSOVER_TYPE)
    elif model == 'cde':
        de = CDE(D, func, POPULATION_SIZE, STRAT_CONSTANT, DELTA)
    elif model == 'jade':
        de = JADE(D, func, POPULATION_SIZE, ARCHIVE_SIZE, P, C)
    elif model == 'shade':
        de = SHADE(D, func, POPULATION_SIZE, MEMORY_SIZE, ARCHIVE_SIZE)
    elif model == 'l_shade':
        max_population_size = int(D * R_N_INIT)
        de = L_SHADE(D, func, max_population_size, MIN_POPULATION_SIZE, max_fes, L_SHADE_MEMORY_SIZE, max_population_size)
    elif model == 'l_shade_rsp':
        de = SHADE(D, func, POPULATION_SIZE, MEMORY_SIZE, ARCHIVE_SIZE)
    elif model == 'qde':
        de = QDE(D, func, POPULATION_SIZE, MUTATION_TYPE)
    return de

def single_run(D, func, func_num, run_id, model='de'):
    seed = get_seed(D, func_num, N_RUNS, run_id)
    np.random.seed(seed)

    if D == 10:
        max_fes = MAX_FES_10
    elif D == 20:
        max_fes = MAX_FES_20
    else:
        max_fes = MAX_FES_OTHER

    de = get_de(D, func(ndim=D), max_fes, model)

    k = 0
    fes = count_fes(de.D, k, max_fes)
    scores = []
    while de.func_evals <= max_fes:
        if de.best_score - de.func.f_global <= MIN_ERROR:
            while fes <= max_fes:
                scores.append(MIN_ERROR)
                k += 1
                fes = count_fes(de.D, k, max_fes)
            break
        if de.next_func_evals() > fes:
            scores.append(de.best_score - de.func.f_global)
            k += 1
            fes = count_fes(de.D, k, max_fes)
        if de.next_func_evals() > max_fes:
            break
        de.step()

    scores.append(de.func_evals)
    return scores

def train(Ds, funcs, model='de'):
    start_total = time.perf_counter()
    for D in Ds:
        results_file_name = f"out/{model}_{D}.txt"
        results_file_name_tex = f"out/{model}_{D}.tex"
        results = []
        columns = ["Best", "Worst", "Median", "Mean", "Std"]
        func_nums = [int(x.__name__[1:-4]) for x in funcs]
        for id, func in enumerate(funcs):
            start = time.perf_counter()
            data = Parallel(n_jobs=N_JOBS)(delayed(single_run)(D, func, func_nums[id], run_id, model) for run_id in range(N_RUNS))
            end = time.perf_counter()
            print(f"Finished D={D} and func_num={func_nums[id]} for model {model} in {end - start} seconds.")
            file_name = f"out/{model}_{func.__name__}_{D}.txt"
            df = []
            for i in range(17):
                row = []
                for j, _ in enumerate(data):
                    row.append(data[j][i])
                df.append(row)
            df = pd.DataFrame(df)
            df.to_csv(file_name, sep=" ", header=False, index=False)
            scores = df.iloc[-2, :].values.flatten().tolist()
            best = np.min(scores)
            worst = np.max(scores)
            median = np.median(scores)
            mean = np.mean(scores)
            std = np.std(scores)
            results.append([best, worst, median, mean, std])
        results = pd.DataFrame(results, columns=columns, index=func_nums)
        results.to_string(results_file_name, float_format="{:.8f}".format)
        results.to_latex(results_file_name_tex, float_format="{:.8f}".format)
    
    start_t0 = time.perf_counter()
    x = 0.55
    for _ in range(200000):
        x = x + x
        x = x / 2
        x = x * x
        x = np.sqrt(x)
        x = np.log(x)
        x = np.exp(x)
        x = x / (x + 2)
    end_t0 = time.perf_counter()
    t0 = end_t0 - start_t0

    times_file_name = f"out/{model}_times.txt"
    times_file_name_tex = f"out/{model}_times.tex"
    times = []
    columns = ['T0', "T1", "T2", "(T2 - T1) / T0"]
    index = Ds
    for D in Ds:
        de = DifferentialEvolution(D, funcs[0](ndim=D), POPULATION_SIZE, F, CR, MUTATION_TYPE, CROSSOVER_TYPE)
        start_t1 = time.perf_counter()
        for _ in range(int(200000 / POPULATION_SIZE)):
            de.evaluate_population()
        end_t1 = time.perf_counter()
        t1 = end_t1 - start_t1

        t2s = []
        for _ in range(5):
            de = get_de(D, funcs[0](ndim=D), 200000, model)
            start_t2 = time.perf_counter()
            de.train(200000)
            end_t2 = time.perf_counter()
            t2s.append(end_t2 - start_t2)
        t2 = np.mean(t2s)
        times.append([t0, t1, t2, (t2 - t1) / t0])
    times = pd.DataFrame(times, columns=columns, index=index)
    times.to_string(times_file_name, float_format="{:.2f}".format)
    times.to_latex(times_file_name_tex, float_format="{:.2f}".format)

    end_total = time.perf_counter()
    print(f"Finished for model {model}", "\n\n", f"Total time: {end_total - start_total} seconds")
            
if __name__ == '__main__':
    Ds = [10, 20]
    cec = "2022"
    funcs = opfunu.get_functions_based_classname(cec)
    funcs.sort(key=lambda x: int(x.__name__[1:-4]))
    train(Ds, funcs, 'de')
    train(Ds, funcs, 'cde')
    train(Ds, funcs, 'jade')
    train(Ds, funcs, 'shade')