import numpy as np
from scipy.stats import wilcoxon

COLUMNS = [1, 2, 3, 4, 5]
COLUMN_NAMES = ["Best", "Worst", "Median", "Mean", "Std"]
N_FUNCS = 12
N_RUNS = 30
OPTIMUM = 1.00e-08

def read_table(table_path):
    table = np.loadtxt(table_path, skiprows=1, usecols=COLUMNS)
    return table

def make_comparison(models, D, columns):
    table = np.empty((N_FUNCS, len(models) * len(columns)))
    column_indexes = []
    for i, column in enumerate(columns):
        column_indexes.append(COLUMN_NAMES.index(column))
    for i, model in enumerate(models):
        model_path = f"out/{model}/{model}_{D}.txt"
        model_table = read_table(model_path)
        for func_num in range(N_FUNCS):
            for j in range(len(columns)):
                x = model_table[func_num][column_indexes[j]]
                x = np.format_float_scientific(x, precision=2, exp_digits=2)
                table[func_num][i * len(columns) + j] = x
    return table

def comparisons_to_latex(comparisons, bold_best=False, bold_optimum=False):
    text = ""
    n_columns = 1
    for comparison in comparisons:
        n_columns += len(comparison[0])
    for func_num in range(N_FUNCS):
        line = f"{func_num + 1} & "
        for comparison in comparisons:
            if bold_best:
                mini = np.min(comparison[func_num])
            for value in comparison[func_num]:
                if type(value) == np.str_:
                    x = value
                else:
                    x = "{:.2e}".format(value)
                if bold_best and value == mini:
                    line += f"\\textbf{{{x}}} & "
                elif bold_optimum and value == OPTIMUM:
                    line += f"\\textbf{{{x}}} & "
                else:
                    line += x + " & "
        line = line[:-3]
        line += f" \\\ \cline{{1-{n_columns}}}\n"
        text += line
    text = text.replace('.', ',')
    return text

def wilcoxon_test(model1, model2, D):
    table = np.empty((N_FUNCS, 1), dtype=np.str_)
    for func in range(1, N_FUNCS + 1):
        func_scores = np.empty((2, N_RUNS))
        for i, model in enumerate((model1, model2)):
            func_path = f"out/{model}/{model}_F{func}2022_{D}.txt"
            func_table = np.loadtxt(func_path)
            func_scores[i] = func_table[-2]
        
        res = wilcoxon(func_scores[0], func_scores[1], zero_method='zsplit', method='approx', alternative='two-sided')
        pvalue = res.pvalue
        mean1 = np.mean(func_scores[0])
        mean2 = np.mean(func_scores[1])
        if pvalue > 0.05:
            symbol = '='
        else:
            if mean1 > mean2:
                symbol = '+'
            else:
                symbol = '-'
        table[func - 1] = symbol
    return table

def count_wins_loses(models):
    symbols_sum_10 = np.zeros(len(models))
    symbols_sum_20 = np.zeros(len(models))
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            symbols_10 = wilcoxon_test(models[i], models[j], 10).flatten()
            symbols_20 = wilcoxon_test(models[i], models[j], 20).flatten()
            for symbol in symbols_10:
                if symbol == '+':
                    symbols_sum_10[i] -= 1
                    symbols_sum_10[j] += 1
                elif symbol == '-':
                    symbols_sum_10[i] += 1
                    symbols_sum_10[j] -= 1
            for symbol in symbols_20:
                if symbol == '+':
                    symbols_sum_20[i] -= 1
                    symbols_sum_20[j] += 1
                elif symbol == '-':
                    symbols_sum_20[i] += 1
                    symbols_sum_20[j] -= 1
    return symbols_sum_10, symbols_sum_20

if __name__ == '__main__':
    # Porównanie różnych akcji dla qde

    # models = ["qde-actions-qlde", "qde"]
    # columns = ["Best", "Mean", "Std"]
    # comp_10 = make_comparison(models, 10, columns)
    # symbols_10 = wilcoxon_test(models[0], models[1], 10)
    # comp_20 = make_comparison(models, 20, columns)
    # symbols_20 = wilcoxon_test(models[0], models[1], 20)
    # text = comparisons_to_latex([comp_10, symbols_10], bold_optimum=True)
    # print(text)
    # text = comparisons_to_latex([comp_20, symbols_20], bold_optimum=True)
    # print(text)


    # Porównanie różnych strategii wyboru akcji dla qde

    # models = ["qde-epsilon-greedy", "qde"]
    # columns = ["Mean"]
    # comp_10 = make_comparison(models, 10, columns)
    # symbols_10 = wilcoxon_test(models[0], models[1], 10)
    # comp_20 = make_comparison(models, 20, columns)
    # symbols_20 = wilcoxon_test(models[0], models[1], 20)
    # text = comparisons_to_latex([comp_10, symbols_10, comp_20, symbols_20], bold_optimum=True)
    # print(text)


    # Porównanie różnej liczby błądzeń losowych dla RL-HPSDE-N-WALKS

    # models = ["rl-hpsde-n-walks", "rl-hpsde-n-walks-scalar-1", "rl-hpsde-n-walks-scalar-2", "rl-hpsde-n-walks-scalar-5"]
    # columns = ["Mean"]
    # comp_10 = make_comparison(models, 10, columns)
    # symbols_10_0_1 = wilcoxon_test(models[0], models[1], 10)
    # symbols_10_0_2 = wilcoxon_test(models[0], models[2], 10)
    # symbols_10_0_3 = wilcoxon_test(models[0], models[3], 10)
    # symbols_10_1_2 = wilcoxon_test(models[1], models[2], 10)
    # symbols_10_1_3 = wilcoxon_test(models[1], models[3], 10)
    # symbols_10_2_3 = wilcoxon_test(models[2], models[3], 10)
    # comp_20 = make_comparison(models, 20, columns)
    # symbols_20_0_1 = wilcoxon_test(models[0], models[1], 20)
    # symbols_20_0_2 = wilcoxon_test(models[0], models[2], 20)
    # symbols_20_0_3 = wilcoxon_test(models[0], models[3], 20)
    # symbols_20_1_2 = wilcoxon_test(models[1], models[2], 20)
    # symbols_20_1_3 = wilcoxon_test(models[1], models[3], 20)
    # symbols_20_2_3 = wilcoxon_test(models[2], models[3], 20)
    # text = comparisons_to_latex([comp_10], bold_best=True)
    # print(text)
    # text = comparisons_to_latex([comp_20], bold_best=True)
    # print(text)


    # Porównanie różnej liczby stanów dla RL-SHADE-RSP

    # models = ["rl-shade-rsp-states-3", "rl-shade-rsp-states-5", "rl-shade-rsp"]
    # columns = ["Mean"]
    # comp_10 = make_comparison(models, 10, columns)
    # symbols_10_0_1 = wilcoxon_test(models[0], models[1], 10)
    # symbols_10_0_2 = wilcoxon_test(models[0], models[2], 10)
    # symbols_10_1_2 = wilcoxon_test(models[1], models[2], 10)
    # comp_20 = make_comparison(models, 20, columns)
    # symbols_20_0_1 = wilcoxon_test(models[0], models[1], 20)
    # symbols_20_0_2 = wilcoxon_test(models[0], models[2], 20)
    # symbols_20_1_2 = wilcoxon_test(models[1], models[2], 20)
    # text = comparisons_to_latex([comp_10, comp_20], bold_best=True)
    # print(text)


    # Porównanie różnych zbiorów akcji dla RL-SHADE

    # models = ["rl-shade", "rl-shade-interval-01", "rl-shade-interval-005"]
    # columns = ["Mean"]
    # comp_10 = make_comparison(models, 10, columns)
    # symbols_10_0_1 = wilcoxon_test(models[0], models[1], 10)
    # symbols_10_0_2 = wilcoxon_test(models[0], models[2], 10)
    # symbols_10_1_2 = wilcoxon_test(models[1], models[2], 10)
    # comp_20 = make_comparison(models, 20, columns)
    # symbols_20_0_1 = wilcoxon_test(models[0], models[1], 20)
    # symbols_20_0_2 = wilcoxon_test(models[0], models[2], 20)
    # symbols_20_1_2 = wilcoxon_test(models[1], models[2], 20)
    # text = comparisons_to_latex([comp_10, comp_20], bold_best=True)
    # print(text)


    # Porównanie różnych zbiorów akcji dla nauczonego RL-SHADE

    # models = ["rl-shade-test-30", "rl-shade-interval-01-test-30", "rl-shade-interval-005"]
    # columns = ["Mean"]
    # comp_10 = make_comparison(models, 10, columns)
    # symbols_10_0_1 = wilcoxon_test(models[0], models[1], 10)
    # symbols_10_0_2 = wilcoxon_test(models[0], models[2], 10)
    # symbols_10_1_2 = wilcoxon_test(models[1], models[2], 10)
    # comp_20 = make_comparison(models, 20, columns)
    # symbols_20_0_1 = wilcoxon_test(models[0], models[1], 20)
    # symbols_20_0_2 = wilcoxon_test(models[0], models[2], 20)
    # symbols_20_1_2 = wilcoxon_test(models[1], models[2], 20)
    # text = comparisons_to_latex([comp_10, comp_20], bold_best=True)
    # print(text)


    # Porównanie douczanego i niedouczanego qDE

    # models = ["qde-test-greedy-30", "qde-test-30"]
    # columns = ["Mean"]
    # comp_10 = make_comparison(models, 10, columns)
    # symbols_10 = wilcoxon_test(models[0], models[1], 10)
    # comp_20 = make_comparison(models, 20, columns)
    # symbols_20 = wilcoxon_test(models[0], models[1], 20)
    # text = comparisons_to_latex([comp_10, symbols_10, comp_20, symbols_20], bold_optimum=True)
    # print(text)


    # Porównanie douczanego i niedouczanego RL-HPSDE

    # models = ["rl-hpsde-test-30", "rl-hpsde-test-boltzmann-30"]
    # columns = ["Mean"]
    # comp_10 = make_comparison(models, 10, columns)
    # symbols_10 = wilcoxon_test(models[0], models[1], 10)
    # comp_20 = make_comparison(models, 20, columns)
    # symbols_20 = wilcoxon_test(models[0], models[1], 20)
    # text = comparisons_to_latex([comp_10, symbols_10, comp_20, symbols_20], bold_optimum=True)
    # print(text)


    # Porównanie douczanego i niedouczanego RL-HPSDE-N-WALKS

    # models = ["rl-hpsde-n-walks-test-30", "rl-hpsde-n-walks-test-boltzmann-30"]
    # columns = ["Best", "Mean", "Std"]
    # comp_10 = make_comparison(models, 10, columns)
    # symbols_10 = wilcoxon_test(models[0], models[1], 10)
    # text = comparisons_to_latex([comp_10, symbols_10], bold_optimum=True)
    # print(text)
    # comp_20 = make_comparison(models, 20, columns)
    # symbols_20 = wilcoxon_test(models[0], models[1], 20)
    # text = comparisons_to_latex([comp_20, symbols_20], bold_optimum=True)
    # print(text)


    # Porównanie douczanego i niedouczanego RL-SHADE-RSP

    # models = ["rl-shade-rsp-test-greedy-30", "rl-shade-rsp-test-30"]
    # columns = ["Mean"]
    # comp_10 = make_comparison(models, 10, columns)
    # symbols_10 = wilcoxon_test(models[0], models[1], 10)
    # comp_20 = make_comparison(models, 20, columns)
    # symbols_20 = wilcoxon_test(models[0], models[1], 20)
    # text = comparisons_to_latex([comp_10, symbols_10, comp_20, symbols_20], bold_optimum=True)
    # print(text)


    # Porównanie douczanego i niedouczanego RL-SHADE

    # models = ["rl-shade-test-greedy-30", "rl-shade-test-30"]
    # columns = ["Best", "Mean", "Std"]
    # comp_10 = make_comparison(models, 10, columns)
    # symbols_10 = wilcoxon_test(models[0], models[1], 10)
    # text = comparisons_to_latex([comp_10, symbols_10], bold_optimum=True)
    # print(text)
    # comp_20 = make_comparison(models, 20, columns)
    # symbols_20 = wilcoxon_test(models[0], models[1], 20)
    # text = comparisons_to_latex([comp_20, symbols_20], bold_optimum=True)
    # print(text)


    # Długość uczenia qDE

    # models = ["qde", "qde-test-10", "qde-test-20", "qde-test-30", "qde-test-40", "qde-test-50", "qde-test-60"]
    # columns = ["Mean"]
    # comp_10 = make_comparison(models, 10, columns)
    # text = comparisons_to_latex([comp_10], bold_best=True)
    # print(text)
    # comp_20 = make_comparison(models, 20, columns)
    # text = comparisons_to_latex([comp_20], bold_best=True)
    # print(text)
    # print(count_wins_loses(models))


    # Długość uczenia RL-HPSDE
    
    # models = ["rl-hpsde", "rl-hpsde-test-10", "rl-hpsde-test-20", "rl-hpsde-test-30", "rl-hpsde-test-40", "rl-hpsde-test-50", "rl-hpsde-test-60"]
    # columns = ["Mean"]
    # comp_10 = make_comparison(models, 10, columns)
    # text = comparisons_to_latex([comp_10], bold_best=True)
    # print(text)
    # comp_20 = make_comparison(models, 20, columns)
    # text = comparisons_to_latex([comp_20], bold_best=True)
    # print(text)
    # print(count_wins_loses(models))


    # Długość uczenia RL-HPSDE-N-WALKS
    
    # models = ["rl-hpsde-n-walks", "rl-hpsde-n-walks-test-10", "rl-hpsde-n-walks-test-20", "rl-hpsde-n-walks-test-30", "rl-hpsde-n-walks-test-40", "rl-hpsde-n-walks-test-50", "rl-hpsde-n-walks-test-60"]
    # columns = ["Mean"]
    # comp_10 = make_comparison(models, 10, columns)
    # text = comparisons_to_latex([comp_10], bold_best=True)
    # print(text)
    # comp_20 = make_comparison(models, 20, columns)
    # text = comparisons_to_latex([comp_20], bold_best=True)
    # print(text)
    # print(count_wins_loses(models))


    # Długość uczenia RL-SHADE-RSP
    
    # models = ["rl-shade-rsp", "rl-shade-rsp-test-10", "rl-shade-rsp-test-20", "rl-shade-rsp-test-30", "rl-shade-rsp-test-40", "rl-shade-rsp-test-50", "rl-shade-rsp-test-60"]
    # columns = ["Mean"]
    # comp_10 = make_comparison(models, 10, columns)
    # text = comparisons_to_latex([comp_10], bold_best=True)
    # print(text)
    # comp_20 = make_comparison(models, 20, columns)
    # text = comparisons_to_latex([comp_20], bold_best=True)
    # print(text)
    # print(count_wins_loses(models))


    # Długość uczenia RL-SHADE
    
    # models = ["rl-shade", "rl-shade-test-10", "rl-shade-test-20", "rl-shade-test-30", "rl-shade-test-40", "rl-shade-test-50", "rl-shade-test-60"]
    # columns = ["Mean"]
    # comp_10 = make_comparison(models, 10, columns)
    # text = comparisons_to_latex([comp_10], bold_best=True)
    # print(text)
    # comp_20 = make_comparison(models, 20, columns)
    # text = comparisons_to_latex([comp_20], bold_best=True)
    # print(text)
    # print(count_wins_loses(models))


    # Różne stany dla qDE

    # models = ["qde", "qde-states-rl-shade", "qde-states-rl-hpsde"]
    # columns = ["Mean"]
    # comp_10 = make_comparison(models, 10, columns)
    # symbols_10_0_1 = wilcoxon_test(models[0], models[1], 10)
    # symbols_10_0_2 = wilcoxon_test(models[0], models[2], 10)
    # symbols_10_1_2 = wilcoxon_test(models[1], models[2], 10)
    # comp_20 = make_comparison(models, 20, columns)
    # symbols_20_0_1 = wilcoxon_test(models[0], models[1], 20)
    # symbols_20_0_2 = wilcoxon_test(models[0], models[2], 20)
    # symbols_20_1_2 = wilcoxon_test(models[1], models[2], 20)
    # text = comparisons_to_latex([comp_10, comp_20], bold_best=True)
    # print(text)


    # Różne stany dla nauczonego qDE

    # models = ["qde-test-30", "qde-states-rl-shade-test-30", "qde-states-rl-hpsde-test-30"]
    # columns = ["Mean"]
    # comp_10 = make_comparison(models, 10, columns)
    # symbols_10_0_1 = wilcoxon_test(models[0], models[1], 10)
    # symbols_10_0_2 = wilcoxon_test(models[0], models[2], 10)
    # symbols_10_1_2 = wilcoxon_test(models[1], models[2], 10)
    # comp_20 = make_comparison(models, 20, columns)
    # symbols_20_0_1 = wilcoxon_test(models[0], models[1], 20)
    # symbols_20_0_2 = wilcoxon_test(models[0], models[2], 20)
    # symbols_20_1_2 = wilcoxon_test(models[1], models[2], 20)
    # text = comparisons_to_latex([comp_10, comp_20], bold_best=True)
    # print(text)


    # Różne stany dla RL-SHADE

    # models = ["rl-shade", "rl-shade-states-rl-hpsde"]
    # columns = ["Mean"]
    # comp_10 = make_comparison(models, 10, columns)
    # symbols_10 = wilcoxon_test(models[0], models[1], 10)
    # comp_20 = make_comparison(models, 20, columns)
    # symbols_20 = wilcoxon_test(models[0], models[1], 20)
    # text = comparisons_to_latex([comp_10, symbols_10, comp_20, symbols_20], bold_optimum=True)
    # print(text)


    # Różne stany dla nauczonego RL-SHADE

    # models = ["rl-shade-test-30", "rl-shade-states-rl-hpsde-test-30"]
    # columns = ["Mean"]
    # comp_10 = make_comparison(models, 10, columns)
    # symbols_10 = wilcoxon_test(models[0], models[1], 10)
    # comp_20 = make_comparison(models, 20, columns)
    # symbols_20 = wilcoxon_test(models[0], models[1], 20)
    # text = comparisons_to_latex([comp_10, symbols_10, comp_20, symbols_20], bold_optimum=True)
    # print(text)


    # Porównanie RL-HPSDE oraz RL-HPSDE-N-WALKS

    # models = ["rl-hpsde-test-30", "rl-hpsde-n-walks-test-30"]
    # columns = ["Best", "Mean", "Std"]
    # comp_10 = make_comparison(models, 10, columns)
    # symbols_10 = wilcoxon_test(models[0], models[1], 10)
    # text = comparisons_to_latex([comp_10, symbols_10], bold_optimum=True)
    # print(text)
    # comp_20 = make_comparison(models, 20, columns)
    # symbols_20 = wilcoxon_test(models[0], models[1], 20)
    # text = comparisons_to_latex([comp_20, symbols_20], bold_optimum=True)
    # print(text)


    # Porównanie cDE oraz qDE

    # models = ["cde", "qde"]
    # columns = ["Best", "Mean", "Std"]
    # comp_10 = make_comparison(models, 10, columns)
    # symbols_10 = wilcoxon_test(models[0], models[1], 10)
    # text = comparisons_to_latex([comp_10, symbols_10], bold_optimum=True)
    # print(text)
    # comp_20 = make_comparison(models, 20, columns)
    # symbols_20 = wilcoxon_test(models[0], models[1], 20)
    # text = comparisons_to_latex([comp_20, symbols_20], bold_optimum=True)
    # print(text)


    # Porównanie L-SHADE oraz RL-HPSDE-N-WALKS

    # models = ["l-shade", "rl-hpsde-n-walks"]
    # columns = ["Best", "Mean", "Std"]
    # comp_10 = make_comparison(models, 10, columns)
    # symbols_10 = wilcoxon_test(models[0], models[1], 10)
    # text = comparisons_to_latex([comp_10, symbols_10], bold_optimum=True)
    # print(text)
    # comp_20 = make_comparison(models, 20, columns)
    # symbols_20 = wilcoxon_test(models[0], models[1], 20)
    # text = comparisons_to_latex([comp_20, symbols_20], bold_optimum=True)
    # print(text)


    # Porównanie L-SHADE-RSP oraz RL-SHADE-RSP

    # models = ["l-shade-rsp", "rl-shade-rsp"]
    # columns = ["Best", "Mean", "Std"]
    # comp_10 = make_comparison(models, 10, columns)
    # symbols_10 = wilcoxon_test(models[0], models[1], 10)
    # text = comparisons_to_latex([comp_10, symbols_10], bold_optimum=True)
    # print(text)
    # comp_20 = make_comparison(models, 20, columns)
    # symbols_20 = wilcoxon_test(models[0], models[1], 20)
    # text = comparisons_to_latex([comp_20, symbols_20], bold_optimum=True)
    # print(text)


    # Porównanie L-SHADE oraz RL-SHADE

    # models = ["l-shade", "rl-shade"]
    # columns = ["Best", "Mean", "Std"]
    # comp_10 = make_comparison(models, 10, columns)
    # symbols_10 = wilcoxon_test(models[0], models[1], 10)
    # text = comparisons_to_latex([comp_10, symbols_10], bold_optimum=True)
    # print(text)
    # comp_20 = make_comparison(models, 20, columns)
    # symbols_20 = wilcoxon_test(models[0], models[1], 20)
    # text = comparisons_to_latex([comp_20, symbols_20], bold_optimum=True)
    # print(text)


    # Porównanie algorytmów wykorzystujących uczenie się ze wzmocnieniem
    
    # models = ["qde-test-50", "rl-hpsde-test-30", "rl-hpsde-n-walks-test-30", "rl-shade-rsp", "rl-shade-test-50"]
    # columns = ["Mean"]
    # comp_10 = make_comparison(models, 10, columns)
    # text = comparisons_to_latex([comp_10], bold_best=True)
    # print(text)
    # comp_20 = make_comparison(models, 20, columns)
    # text = comparisons_to_latex([comp_20], bold_best=True)
    # print(text)
    # print(count_wins_loses(models))


    # Porównanie RL-SHADE z adaptacyjnymi ewolucjami różnicowymi

    models = ["cde", "jade", "shade", "l-shade", "l-shade-rsp", "rl-shade"]
    columns = ["Mean"]
    comp_10 = make_comparison(models, 10, columns)
    text = comparisons_to_latex([comp_10], bold_best=True)
    print(text)
    comp_20 = make_comparison(models, 20, columns)
    text = comparisons_to_latex([comp_20], bold_best=True)
    print(text)
    print(count_wins_loses(models))
