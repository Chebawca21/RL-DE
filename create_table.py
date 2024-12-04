import numpy as np

COLUMNS = [1, 2, 3, 4, 5]
COLUMN_NAMES = ["Best", "Worst", "Median", "Mean", "Std"]
N_FUNCS = 12

def read_table(table_path):
    table = np.loadtxt(table_path, skiprows=1, usecols=COLUMNS)
    return table

def make_comparison(models, D, column):
    table = np.empty((N_FUNCS, len(models)))
    column_index = COLUMN_NAMES.index(column)
    for i, model in enumerate(models):
        model_path = f"out/{model}/{model}_{D}.txt"
        model_table = read_table(model_path)
        for func_num in range(N_FUNCS):
            x = model_table[func_num][column_index]
            x = np.format_float_scientific(x, precision=2, exp_digits=2)
            table[func_num][i] = x
    return table

def comparisons_to_latex(comparisons, bold_best=False):
    text = ""
    n_columns = 1
    for comparison in comparisons:
        n_columns += len(comparison[0])
    for func_num in range(N_FUNCS):
        line = f"{func_num + 1} & "
        for comparison in comparisons:
            mini = np.min(comparison[func_num])
            for value in comparison[func_num]:
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

if __name__ == '__main__':
    models = ["rl-shade-test", "rl-shade-test-boltzmann"]
    column = "Median"

    comp_10 = make_comparison(models, 10, column)
    comp_20 = make_comparison(models, 20, column)
    text = comparisons_to_latex([comp_10, comp_20], bold_best=True)
    print(text)