import numpy as np
from scipy.stats import wilcoxon

COLUMNS = [1, 2, 3, 4, 5]
COLUMN_NAMES = ["Best", "Worst", "Median", "Mean", "Std"]
N_FUNCS = 12
N_RUNS = 30

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
                table[func_num][i * len(models) + j] = x
    return table

def comparisons_to_latex(comparisons, bold_best=False):
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

if __name__ == '__main__':
    models = ["qde-actions-qlde", "qde"]
    columns = ["Mean", "Std"]

    comp_10 = make_comparison(models, 10, columns)
    symbols_10 = wilcoxon_test(models[0], models[1], 10)
    comp_20 = make_comparison(models, 20, columns)
    symbols_20 = wilcoxon_test(models[0], models[1], 20)
    text = comparisons_to_latex([comp_20, symbols_20])
    print(text)
