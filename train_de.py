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
from models.rl_shade import RL_SHADE
from config import get_model_parameters, get_cec_funcs

MAX_FES_10 = 200000
MAX_FES_20 = 1000000
MAX_FES_OTHER = 10000
MIN_ERROR = 10 ** (-8)
N_RUNS = 30

N_JOBS = -1

def get_seed(problem_size, func_num, runs, run_id):
    with open("Rand_Seeds.txt") as f:
        lines = f.read().splitlines()
    seed_id = (problem_size // 10 * func_num * runs + run_id) - runs
    seed_id = seed_id % 1000
    return int(float(lines[seed_id].strip()))

def count_fes(D, k, max_fes):
    return D ** (k/5 - 3) * max_fes

def get_de(D, func, max_fes, model='de', qtable_path=""):
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
    elif model == 'qde' or model == 'qde-train' or model == 'qde-test' or model == 'qde-test-greedy' or model == 'qde-epsilon-greedy' or model == 'qde-actions-qlde':
        de = QDE(**params)
    elif model == 'rl-hpsde' or model == 'rl-hpsde-train' or model == 'rl-hpsde-test' or model == 'rl-hpsde-test-boltzmann':
        de = RL_HPSDE(**params)
    elif model == 'rl-hpsde-n-walks' or model == 'rl-hpsde-n-walks-train' or model == 'rl-hpsde-n-walks-test' or model == 'rl-hpsde-n-walks-test-boltzmann' or model == 'rl-hpsde-n-walks-scalar-1' or model == 'rl-hpsde-n-walks-scalar-2' or model == 'rl-hpsde-n-walks-scalar-5':
        de = RL_HPSDE_N_WALKS(**params)
    elif model == 'rl-shade-rsp' or model == 'rl-shade-rsp-train' or model == 'rl-shade-rsp-test' or model == 'rl-shade-rsp-test-greedy' or model == 'rl-shade-rsp-states-5' or model == 'rl-shade-rsp-states-10':
        de = RL_SHADE_RSP(**params)
    elif model == 'rl-shade' or model == 'rl-shade-train' or model == 'rl-shade-test' or model == 'rl-shade-test-greedy' or model == 'rl-shade-interval-02' or model == 'rl-shade-interval-02-train' or model == 'rl-shade-interval-02-test' or model == 'rl-shade-interval-005' or model == 'rl-shade-interval-005-train' or model == 'rl-shade-interval-005-test':
        de = RL_SHADE(**params)
    
    if qtable_path != "":
        de.qlearning.load_qtable(qtable_path)
    return de

def single_run(D, func_name, func_num, run_id, model='de', qtable_in_path="", qtable_out_path=""):
    seed = get_seed(D, func_num, N_RUNS, run_id)
    np.random.seed(seed)

    if D == 10:
        max_fes = MAX_FES_10
    elif D == 20:
        max_fes = MAX_FES_20
    else:
        max_fes = MAX_FES_OTHER

    de = get_de(D, func_name, max_fes, model, qtable_in_path)

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
    
    if qtable_out_path != "":
        de.qlearning.save_qtable(qtable_out_path)
    scores.append(de.func_evals)
    return scores

def evolve(Ds, funcs_names, model='de', qtable_in_number=0):
    if qtable_in_number > 0:
        model_name = f"{model}-{qtable_in_number}"
    else:
        model_name = model
    model_out_path = f"out/{model_name}"
    Path(model_out_path).mkdir(parents=True, exist_ok=True)
    qtable_in_path = ""
    if qtable_in_number > 0:
        test_index = model_name.index('-test')
        model_name_short = model_name[:test_index]
        model_in_path = f"in/{model_name_short}"
        Path(model_in_path).mkdir(parents=True, exist_ok=True)
        qtable_in_path = f"{model_in_path}/{model_name_short}_qtable_{qtable_in_number}.txt"
    start_total = time.perf_counter()
    for D in Ds:

        results_file_name = f"{model_out_path}/{model_name}_{D}.txt"
        results = []
        columns = ["Best", "Worst", "Median", "Mean", "Std"]
        for id, func_name in enumerate(funcs_names):
            start = time.perf_counter()
            data = Parallel(n_jobs=N_JOBS)(delayed(single_run)(D, func_name, id + 1, run_id, model, qtable_in_path) for run_id in range(N_RUNS))
            end = time.perf_counter()
            print(f"Finished D={D} and func_num={id + 1} for model {model_name} in {end - start} seconds.")
            file_name = f"{model_out_path}/{model_name}_{func_name}_{D}.txt"
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

    times_file_name = f"{model_out_path}/{model_name}_times.txt"
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

    end_total = time.perf_counter()
    print(f"Finished for model {model_name}", "\n\n", f"Total time: {end_total - start_total} seconds")

def train_model(Ds, funcs_names, runs, model):
    model_name = model.replace('-train', '')
    model_path = f"in/{model_name}"
    Path(model_path).mkdir(parents=True, exist_ok=True)
    if runs[0] == 0:
        de = get_de(Ds[0], funcs_names[0], 200000, model)
        qtable_0_path = f"{model_path}/{model_name}_qtable_0.txt"
        de.qlearning.save_qtable(qtable_0_path)
    start_all_runs = time.perf_counter()
    for run in runs:
        qtable_in_path = f"{model_path}/{model_name}_qtable_{run}.txt"
        qtable_out_path = f"{model_path}/{model_name}_qtable_{run + 1}.txt"
        for D in Ds:
            start = time.perf_counter()
            for func_num, func_name in enumerate(funcs_names):
                single_run(D, func_name, func_num, run, model, qtable_in_path, qtable_out_path)
                qtable_in_path = qtable_out_path
            end = time.perf_counter()
            print(f"Finished D={D} and run={run} for model={model} in {end - start} seconds.")
    end_all_runs = time.perf_counter()
    print(f"Finished runs from {runs[0]} to {runs[-1]} for model={model} in {end_all_runs - start_all_runs} seconds.")

def train(Ds, funcs_names, runs, models):
    start = time.perf_counter()
    Parallel(n_jobs=len(models))(delayed(train_model)(Ds, funcs_names, runs, model) for model in models)
    end = time.perf_counter()
    print(f"Finished training in {end - start} seconds.")


if __name__ == '__main__':
    # N_RUNS = 60
    # Ds = [10, 20]
    # cec = "2021"
    # funcs_names = get_cec_funcs(cec)
    # runs = list(range(0, 30))
    # models = ['rl-shade-train', 'rl-shade-interval-02-train', 'rl-shade-interval-005-train', 'qde-train']
    # models = ['qde-train', 'rl-hpsde-train', 'rl-hpsde-n-walks-train', 'rl-shade-rsp-train', 'rl-shade-train']
    # train(Ds, funcs_names, runs, models)

    N_RUNS = 30
    Ds = [10, 20]
    cec = "2022"
    funcs_names = get_cec_funcs(cec)
    # evolve(Ds, funcs_names, 'de')
    # evolve(Ds, funcs_names, 'cde')
    # evolve(Ds, funcs_names, 'jade')
    # evolve(Ds, funcs_names, 'shade')
    # evolve(Ds, funcs_names, 'l-shade')
    # evolve(Ds, funcs_names, 'l-shade-rsp')
    # evolve(Ds, funcs_names, 'qde')
    # evolve(Ds, funcs_names, 'qde-actions-qlde')
    # evolve(Ds, funcs_names, 'qde-epsilon-greedy')
    # evolve(Ds, funcs_names, 'rl-hpsde')
    # evolve(Ds, funcs_names, 'rl-hpsde-n-walks')
    # evolve(Ds, funcs_names, 'rl-hpsde-n-walks-scalar-1')
    # evolve(Ds, funcs_names, 'rl-hpsde-n-walks-scalar-2')
    # evolve(Ds, funcs_names, 'rl-hpsde-n-walks-scalar-5')
    # evolve(Ds, funcs_names, 'rl-shade-rsp')
    # evolve(Ds, funcs_names, 'rl-shade-rsp-states-5')
    # evolve(Ds, funcs_names, 'rl-shade-rsp-states-10')
    # evolve(Ds, funcs_names, 'rl-shade')
    # evolve(Ds, funcs_names, 'rl-shade-interval-02')
    # evolve(Ds, funcs_names, 'rl-shade-interval-005')
    # evolve(Ds, funcs_names, 'qde-test', qtable_in_number=30)
    # evolve(Ds, funcs_names, 'rl-hpsde-test', qtable_in_number=30)
    # evolve(Ds, funcs_names, 'rl-hpsde-n-walks-test', qtable_in_number=30)
    # evolve(Ds, funcs_names, 'rl-shade-rsp-test', qtable_in_number=30)
    # evolve(Ds, funcs_names, 'rl-shade-test', qtable_in_number=30)
    # evolve(Ds, funcs_names, 'rl-shade-interval-02-test', qtable_in_number=30)
    # evolve(Ds, funcs_names, 'rl-shade-interval-005-test', qtable_in_number=30)
    # evolve(Ds, funcs_names, 'qde-test-greedy', qtable_in_number=30)
    # evolve(Ds, funcs_names, 'rl-hpsde-test-boltzmann', qtable_in_number=30)
    # evolve(Ds, funcs_names, 'rl-hpsde-n-walks-test-boltzmann', qtable_in_number=30)
    # evolve(Ds, funcs_names, 'rl-shade-rsp-test-greedy', qtable_in_number=30)
    # evolve(Ds, funcs_names, 'rl-shade-test-greedy', qtable_in_number=30)
    # evolve(Ds, funcs_names, 'qde-test', qtable_in_number=20)
    # evolve(Ds, funcs_names, 'rl-hpsde-test', qtable_in_number=20)
    # evolve(Ds, funcs_names, 'rl-hpsde-n-walks-test', qtable_in_number=20)
    # evolve(Ds, funcs_names, 'rl-shade-rsp-test', qtable_in_number=20)
    # evolve(Ds, funcs_names, 'rl-shade-test', qtable_in_number=20)
    # evolve(Ds, funcs_names, 'qde-test', qtable_in_number=10)
    # evolve(Ds, funcs_names, 'rl-hpsde-test', qtable_in_number=10)
    # evolve(Ds, funcs_names, 'rl-hpsde-n-walks-test', qtable_in_number=10)
    # evolve(Ds, funcs_names, 'rl-shade-rsp-test', qtable_in_number=10)
    # evolve(Ds, funcs_names, 'rl-shade-test', qtable_in_number=10)
    evolve(Ds, funcs_names, 'qde-test', qtable_in_number=40)
    evolve(Ds, funcs_names, 'rl-hpsde-test', qtable_in_number=40)
    evolve(Ds, funcs_names, 'rl-hpsde-n-walks-test', qtable_in_number=40)
    evolve(Ds, funcs_names, 'rl-shade-rsp-test', qtable_in_number=40)
    evolve(Ds, funcs_names, 'rl-shade-test', qtable_in_number=40)
    evolve(Ds, funcs_names, 'qde-test', qtable_in_number=50)
    evolve(Ds, funcs_names, 'rl-hpsde-test', qtable_in_number=50)
    evolve(Ds, funcs_names, 'rl-hpsde-n-walks-test', qtable_in_number=50)
    evolve(Ds, funcs_names, 'rl-shade-rsp-test', qtable_in_number=50)
    evolve(Ds, funcs_names, 'rl-shade-test', qtable_in_number=50)
    evolve(Ds, funcs_names, 'qde-test', qtable_in_number=50)
    evolve(Ds, funcs_names, 'rl-hpsde-test', qtable_in_number=60)
    evolve(Ds, funcs_names, 'rl-hpsde-n-walks-test', qtable_in_number=60)
    evolve(Ds, funcs_names, 'rl-shade-rsp-test', qtable_in_number=60)
    evolve(Ds, funcs_names, 'rl-shade-test', qtable_in_number=60)