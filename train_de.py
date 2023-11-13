import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from SO_BO.CEC2022 import cec2022_func
from de import DifferentialEvolution
from cde import CDE
from jade import JADE
from shade import SHADE

MAX_FES_10 = 200000
MAX_FES_20 = 1000000
MAX_FES_OTHER = 10000
MIN_ERROR = 10 ** (-8)
N_RUNS = 30

OPTIMAL_SCORES = [300, 400, 600, 800, 900, 1800, 2000, 2200, 2300, 2400, 2600, 2700]

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

N_JOBS = -1

def get_seed(problem_size, func_num, runs, run_id):
    with open("SO_BO/input_data/Rand_Seeds.txt") as f:
        lines = f.read().splitlines()
    seed_id = (problem_size // 10 * func_num * runs + run_id) - runs
    seed_id = seed_id % 1000
    return int(float(lines[seed_id].strip()))

def count_fes(D, k, max_fes):
    return D ** (k/5 - 3) * max_fes

def get_de(D, func_num, model='de'):
    if model == 'de':
        de = DifferentialEvolution(D, func_num, POPULATION_SIZE, F, CR, MUTATION_TYPE, CROSSOVER_TYPE)
    elif model == 'cde':
        de = CDE(D, func_num, POPULATION_SIZE, STRAT_CONSTANT, DELTA)
    elif model == 'jade':
        de = JADE(D, func_num, POPULATION_SIZE, ARCHIVE_SIZE, P, C)
    elif model == 'shade':
        de = SHADE(D, func_num, POPULATION_SIZE, MEMORY_SIZE, ARCHIVE_SIZE)
    return de

def single_run(D, func_num, run_id, model='de'):
    seed = get_seed(D, func_num, N_RUNS, run_id)
    np.random.seed(seed)

    de = get_de(D, func_num, model)

    if de.D == 10:
        max_fes = MAX_FES_10
    elif de.D == 20:
        max_fes = MAX_FES_20
    else:
        max_fes = MAX_FES_OTHER

    k = 0
    fes = count_fes(de.D, k, max_fes)
    scores = []
    for _ in range(int(max_fes / de.population_size)):
        if de.best_score - OPTIMAL_SCORES[de.func_num - 1] <= MIN_ERROR:
            while fes <= max_fes:
                scores.append(MIN_ERROR)
                k += 1
                fes = count_fes(de.D, k, max_fes)
            break
        if de.func_evals + de.population_size > fes:
            scores.append(de.best_score - OPTIMAL_SCORES[de.func_num - 1])
            k += 1
            fes = count_fes(de.D, k, max_fes)
        if de.func_evals + de.population_size > max_fes:
            break
        de.step()

    scores.append(de.func_evals)
    return scores

def train(Ds, func_nums, model='de'):
    start_total = time.perf_counter()
    for D in Ds:
        results_file_name = f"out/{model}_{D}.txt"
        results_file_name_tex = f"out/{model}_{D}.tex"
        results = []
        columns = ["Best", "Worst", "Median", "Mean", "Std"]
        index = func_nums
        for func_num in func_nums:
            start = time.perf_counter()
            data = Parallel(n_jobs=N_JOBS)(delayed(single_run)(D, func_num, run_id, model) for run_id in range(N_RUNS))
            end = time.perf_counter()
            print(f"Finished D={D} and func_num={func_num} for model {model} in {end - start} seconds.")
            file_name = f"out/{model}_{func_num}_{D}.txt"
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
        results = pd.DataFrame(results, columns=columns, index=index)
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
        de = DifferentialEvolution(D, 1, POPULATION_SIZE, F, CR, MUTATION_TYPE, CROSSOVER_TYPE)
        start_t1 = time.perf_counter()
        for _ in range(int(200000 / POPULATION_SIZE)):
            de.evaluate_population()
        end_t1 = time.perf_counter()
        t1 = end_t1 - start_t1

        t2s = []
        for _ in range(5):
            de = get_de(D, 1, model)
            start_t2 = time.perf_counter()
            for _ in range(int(200000 / POPULATION_SIZE)):
                de.step()
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
    func_nums = list(range(1, 13))
    train(Ds, func_nums, 'cde')