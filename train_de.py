import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pathlib import Path
from models.de import DifferentialEvolution
from models.cde import CDE
from models.jade import JADE
from models.shade import SHADE
from models.l_shade import L_SHADE
from models.l_shade_rsp import L_SHADE_RSP
from models.qde import QDE
from models.rl_hpsde import RL_HPSDE
from models.rl_hpsde_n_walks import RL_HPSDE_N_WALKS
from models.rl_shade_rsp import RL_SHADE_RSP
from config import get_model_parameters, get_cec_funcs

MAX_FES_10 = 200000
MAX_FES_20 = 1000000
MAX_FES_OTHER = 10000
MIN_ERROR = 10 ** (-8)
N_RUNS = 60

TRAIN_FILE_QDE = "qtable_qde.txt"
TRAIN_FILE = "qtable.txt"
TRAIN_FILE_N_WALKS = "qtable_n_walks.txt"
TRAIN_FILE_RL_SHADE_RSP = "qtable_rl_shade_rsp.txt"

N_JOBS = -1

def get_seed(problem_size, func_num, runs, run_id):
    with open("Rand_Seeds.txt") as f:
        lines = f.read().splitlines()
    seed_id = (problem_size // 10 * func_num * runs + run_id) - runs
    seed_id = seed_id % 1000
    return int(float(lines[seed_id].strip()))

def count_fes(D, k, max_fes):
    return D ** (k/5 - 3) * max_fes

def get_de(D, func, max_fes, model='de'):
    params = get_model_parameters(model, D, func)
    if model == 'de':
        de = DifferentialEvolution(**params)
    elif model == 'cde':
        de = CDE(**params)
    elif model == 'jade':
        de = JADE(**params)
    elif model == 'shade':
        de = SHADE(**params)
    elif model == 'l-shade':
        de = L_SHADE(**params)
    elif model == 'l-shade-rsp':
        de = L_SHADE_RSP(**params)
    elif model == 'qde':
        de = QDE(**params)
    elif model == 'qde-train':
        de = QDE(**params)
        de.qlearning.load_qtable(TRAIN_FILE_QDE)
    elif model == 'qde-test':
        de = QDE(**params)
        de.qlearning.load_qtable(TRAIN_FILE_QDE)
    elif model == 'rl-hpsde':
        de = RL_HPSDE(**params)
    elif model == 'rl-hpsde-train':
        de = RL_HPSDE(**params)
        de.qlearning.load_qtable(TRAIN_FILE)
    elif model == 'rl-hpsde-test':
        de = RL_HPSDE(**params)
        de.qlearning.load_qtable(TRAIN_FILE)
    elif model == 'rl-hpsde-n-walks':
        de = RL_HPSDE_N_WALKS(**params)
    elif model == 'rl-hpsde-n-walks-train':
        de = RL_HPSDE_N_WALKS(**params)
        de.qlearning.load_qtable(TRAIN_FILE_N_WALKS)
    elif model == 'rl-hpsde-n-walks-test':
        de = RL_HPSDE_N_WALKS(**params)
        de.qlearning.load_qtable(TRAIN_FILE_N_WALKS)
    elif model == 'rl-shade-rsp':
        de = RL_SHADE_RSP(**params)
    elif model == 'rl-shade-rsp-train':
        de = RL_SHADE_RSP(**params)
        de.qlearning.load_qtable(TRAIN_FILE_RL_SHADE_RSP)
    elif model == 'rl-shade-rsp-test':
        de = RL_SHADE_RSP(**params)
        de.qlearning.load_qtable(TRAIN_FILE_RL_SHADE_RSP)
    return de

def single_run(D, func_name, func_num, run_id, model='de'):
    seed = get_seed(D, func_num, N_RUNS, run_id)
    np.random.seed(seed)

    if D == 10:
        max_fes = MAX_FES_10
    elif D == 20:
        max_fes = MAX_FES_20
    else:
        max_fes = MAX_FES_OTHER

    de = get_de(D, func_name, max_fes, model)

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

    if model == 'rl-hpsde-train':
        de.qlearning.save_qtable(TRAIN_FILE)
    elif model == 'rl-hpsde-n-walks-train':
        de.qlearning.save_qtable(TRAIN_FILE_N_WALKS)
    elif model == 'rl-shade-rsp-train':
        de.qlearning.save_qtable(TRAIN_FILE_RL_SHADE_RSP)
    elif model == 'qde-train':
        de.qlearning.save_qtable(TRAIN_FILE_QDE)
    scores.append(de.func_evals)
    return scores

def evolve(Ds, funcs_names, model='de'):
    model_path = f"out/{model}"
    Path(model_path).mkdir(parents=True, exist_ok=True)
    start_total = time.perf_counter()
    for D in Ds:
        results_file_name = f"{model_path}/{model}_{D}.txt"
        results_file_name_tex = f"{model_path}/{model}_{D}.tex"
        results = []
        columns = ["Best", "Worst", "Median", "Mean", "Std"]
        for id, func_name in enumerate(funcs_names):
            start = time.perf_counter()
            data = Parallel(n_jobs=N_JOBS)(delayed(single_run)(D, func_name, id + 1, run_id, model) for run_id in range(N_RUNS))
            end = time.perf_counter()
            print(f"Finished D={D} and func_num={id + 1} for model {model} in {end - start} seconds.")
            file_name = f"{model_path}/{model}_{func_name}_{D}.txt"
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
        results = pd.DataFrame(results, columns=columns, index=range(1, len(funcs_names) + 1))
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

    times_file_name = f"{model_path}/{model}_times.txt"
    times_file_name_tex = f"{model_path}/{model}_times.tex"
    times = []
    columns = ['T0', "T1", "T2", "(T2 - T1) / T0"]
    index = Ds
    for D in Ds:
        max_fes = MAX_FES_10
        params = get_model_parameters("de", D, funcs_names[0])
        de = DifferentialEvolution(**params)
        start_t1 = time.perf_counter()
        for _ in range(int(max_fes / de.population_size)):
            de.evaluate_population()
        end_t1 = time.perf_counter()
        t1 = end_t1 - start_t1

        t2s = []
        for _ in range(5):
            de = get_de(D, funcs_names[0], 200000, model)
            start_t2 = time.perf_counter()
            de.evolve(200000)
            end_t2 = time.perf_counter()
            t2s.append(end_t2 - start_t2)
        t2 = np.mean(t2s)
        times.append([t0, t1, t2, (t2 - t1) / t0])
    times = pd.DataFrame(times, columns=columns, index=index)
    times.to_string(times_file_name, float_format="{:.2f}".format)
    times.to_latex(times_file_name_tex, float_format="{:.2f}".format)

    end_total = time.perf_counter()
    print(f"Finished for model {model}", "\n\n", f"Total time: {end_total - start_total} seconds")

def train_with_rl(Ds, funcs_names, model):
    for D in Ds:
        for func_num, func_name in enumerate(funcs_names):
            start = time.perf_counter()
            for run in range(N_RUNS):
                single_run(D, func_name, func_num, run, model) 
            end = time.perf_counter()
            print(f"Finished D={D} and func={func_name} in {end - start} seconds.")


if __name__ == '__main__':
    Ds = [10, 20]
    cec = "2021"
    funcs_names = get_cec_funcs(cec)
    train_with_rl(Ds, funcs_names, 'rl-hpsde-n-walks-train')
    train_with_rl(Ds, funcs_names, 'rl-shade-rsp-train')
    train_with_rl(Ds, funcs_names, 'rl-hpsde-train')
    train_with_rl(Ds, funcs_names, 'qde-train')

    N_RUNS = 30
    Ds = [10, 20]
    cec = "2022"
    funcs_names = get_cec_funcs(cec)
    evolve(Ds, funcs_names, 'rl-hpsde-n-walks-test')
    evolve(Ds, funcs_names, 'rl-hpsde-test')
    evolve(Ds, funcs_names, 'rl-shade-rsp-test')
    evolve(Ds, funcs_names, 'qde-test')
