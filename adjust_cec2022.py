import opfunu
import numpy as np
from opfunu.utils import operator
import opfunu.utils
import opfunu.utils.operator

# F1
def zakharov_func(x):
    x = np.array(x).ravel()
    temp = np.sum(0.5 * x * list(range(1, len(x) + 1)))
    return np.sum(x ** 2) + temp ** 2 + temp ** 4

# F3
def f3_evaluate(self, x, *args):
    self.n_fe += 1
    self.check_solution(x, self.dim_max, self.dim_supported)
    y = x - self.f_shift
    return operator.rotated_expanded_schaffer_func(y) + self.f_bias

def rotated_expanded_schaffer_func(x):
    x = np.asarray(x).ravel()
    x_pairs = np.column_stack((x, np.roll(x, -1)))
    sum_norm = np.sqrt(x_pairs[:, 0] ** 2 + x_pairs[:, 1] ** 2)
    tmp = np.sin(50 * pow(sum_norm, 0.2))
    # Calculate the Schaffer function for all pairs simultaneously
    schaffer_values = np.sqrt(sum_norm) + np.sqrt(sum_norm) * tmp * tmp
    f = np.sum(schaffer_values[:-1])
    f = f * f / (len(x) - 1) / (len(x) - 1)
    return f

# F4
def f4_evaluate(self, x, *args):
    self.n_fe += 1
    self.check_solution(x, self.dim_max, self.dim_supported)
    z = np.dot(self.f_matrix, x - self.f_shift)
    return operator.non_continuous_rastrigin_func(z) + self.f_bias

def non_continuous_rastrigin_func(x):
    x = np.array(x).ravel()
    results = rastrigin_func(x)
    return np.sum(results)

# F5
def f5_evaluate(self, x, *args):
    self.n_fe += 1
    self.check_solution(x, self.dim_max, self.dim_supported)
    z = np.dot(self.f_matrix, x - self.f_shift)
    return operator.levy_func(z, shift=1.0) + self.f_bias

# F6
def f6_evaluate(self, x, *args):
    self.n_fe += 1
    self.check_solution(x, self.dim_max, self.dim_supported)
    z = x - self.f_shift
    mz = np.dot(self.f_matrix, z)
    mz1 = np.concatenate((mz[self.idx1], mz[self.idx2], mz[self.idx3]))
    return (operator.bent_cigar_func(mz1[:self.n1]) +
            operator.hgbat_func(mz1[self.n1:self.n2], shift=-1.0) +
            operator.rastrigin_func(mz1[self.n2:]) + self.f_bias)

def hgbat_func(x, shift=0.0):
    x = np.array(x).ravel()
    ndim = len(x)
    z = 5.0 * x / 100.0 + shift
    t1 = np.sum(z)
    t2 = np.sum(z ** 2)
    return np.abs(t2 ** 2 - t1 ** 2) ** 0.5 + (0.5 * t2 + t1) / ndim + 0.5

def rastrigin_func(x):
    x = np.array(x).ravel()
    z = 5.12 * x / 100.0
    return np.sum(z ** 2 - 10 * np.cos(2 * np.pi * z) + 10)

# F7
def f7_evaluate(self, x, *args):
    self.n_fe += 1
    self.check_solution(x, self.dim_max, self.dim_supported)
    z = x - self.f_shift
    mz = np.dot(self.f_matrix, z)
    mz1 = np.concatenate((mz[self.idx1], mz[self.idx2], mz[self.idx3], mz[self.idx4], mz[self.idx5], mz[self.idx6]))
    return (operator.hgbat_func(mz1[:self.n1], shift=-1.0) +
            operator.katsuura_func(mz1[self.n1:self.n2]) +
            operator.ackley_func(mz1[self.n2:self.n3]) +
            operator.rastrigin_func(mz1[self.n3:self.n4]) +
            operator.modified_schwefel_func(mz1[self.n4:self.n5]) +
            operator.rotated_expanded_schaffer_func(mz1[:int(np.ceil(0.2 * self.ndim))]) + self.f_bias)

def katsuura_func(x):
    x = np.array(x).ravel()
    ndim = len(x)
    z = 5.0 * x / 100.0
    result = 1.0
    for idx in range(0, ndim):
        temp = np.sum([np.abs(2 ** j * z[idx] - np.round(2 ** j * z[idx])) / 2 ** j for j in range(1, 33)])
        result *= (1 + (idx + 1) * temp) ** (10.0 / ndim ** 1.2)
    return (result - 1) * 10 / ndim ** 2

def modified_schwefel_func(x):
    z = 1000.0 * x / 100.0
    z = np.array(z).ravel() + 4.209687462275036e+002
    nx = len(z)

    fx = np.zeros(nx)
    for i in range(nx):
        if z[i] > 500:
            fx[i] -= ((500.0 - np.fmod(z[i], 500)) * np.sin(np.sqrt(500.0 - np.fmod(z[i], 500))) -
                     ((z[i] - 500.0) / 100.) ** 2 / nx)
        elif z[i] < -500:
            fx[i] -= (-500.0 + np.fmod(np.abs(z[i]), 500)) * np.sin(np.sqrt(500.0 - np.fmod(np.abs(z[i]), 500))) - (
                     (z[i] + 500.0) / 100.) ** 2 / nx
        else:
            fx[i] -= z[i] * np.sin(np.sqrt(np.abs(z[i])))

    return np.sum(fx) + 4.189828872724338e+002 * nx

# F8
def f8_evaluate(self, x, *args):
    self.n_fe += 1
    self.check_solution(x, self.dim_max, self.dim_supported)
    z = x - self.f_shift
    mz = np.dot(self.f_matrix, z)
    mz1 = np.concatenate((mz[self.idx1], mz[self.idx2], mz[self.idx3], mz[self.idx4], mz[self.idx5]))
    return (operator.katsuura_func(mz1[:self.n1]) +
            operator.happy_cat_func(mz1[self.n1:self.n2], shift=-1.0) +
            operator.grie_rosen_cec_func(mz1[self.n2:self.n3]) +
            operator.modified_schwefel_func(mz1[self.n3:self.n4]) +
            operator.ackley_func(mz1[self.n4:self.ndim]) + self.f_bias)

def happy_cat_func(x, shift=0.0):
    z = 5.0 * x / 100.0
    z = np.array(z).ravel() + shift
    ndim = len(z)
    t1 = np.sum(z)
    t2 = np.sum(z ** 2)
    return np.abs(t2 - ndim) ** 0.25 + (0.5 * t2 + t1) / ndim + 0.5

def grie_rosen_cec_func(x):
    """This is based on the CEC version which unrolls the griewank and rosenbrock functions for better performance"""
    z = 5.0 * x / 100.0
    z = np.array(z).ravel()
    z += 1.0  # This centers the optimal solution of rosenbrock to 0

    tmp1 = (z[:-1] * z[:-1] - z[1:]) ** 2
    tmp2 = (z[:-1] - 1.0) ** 2
    temp = 100.0 * tmp1 + tmp2
    f = np.sum(temp ** 2 / 4000.0 - np.cos(temp) + 1.0)
    # Last calculation
    tmp1 = (z[-1] * z[-1] - z[0]) ** 2
    tmp2 = (z[-1] - 1.0) ** 2
    temp = 100.0 * tmp1 + tmp2
    f += (temp ** 2) / 4000.0 - np.cos(temp) + 1.0

    return f

# F9
def f9_evaluate(self, x, *args):
    self.lamdas = [1, 1e-6, 1e-26, 1e-6, 1e-6]

    self.n_fe += 1
    self.check_solution(x, self.dim_max, self.dim_supported)

    # 1. Rotated Rosenbrock’s Function f2
    z0 = np.dot(self.f_matrix[:self.ndim, :], 2.048*(x - self.f_shift[0])/100) + 1
    g0 = self.lamdas[0] * self.g0(z0) + self.bias[0]
    w0 = operator.calculate_weight(x - self.f_shift[0], self.xichmas[0])

    # 2. High Conditioned Elliptic Function f8
    z1 = np.dot(self.f_matrix[self.ndim:2*self.ndim, :], x - self.f_shift[1])
    g1 = self.lamdas[1] * self.g1(z1) + self.bias[1]
    w1 = operator.calculate_weight(x - self.f_shift[1], self.xichmas[1])

    # 3. Rotated Bent Cigar Function f6
    z2 = np.dot(self.f_matrix[2*self.ndim:3*self.ndim, :], x - self.f_shift[2])
    g2 = self.lamdas[2] * self.g2(z2) + self.bias[2]
    w2 = operator.calculate_weight(x - self.f_shift[2], self.xichmas[2])

    # 4. Rotated Discus Function f14
    z3 = np.dot(self.f_matrix[3 * self.ndim:4 * self.ndim, :], x - self.f_shift[3])
    g3 = self.lamdas[3] * self.g3(z3) + self.bias[3]
    w3 = operator.calculate_weight(x - self.f_shift[3], self.xichmas[3])

    # 5. High Conditioned Elliptic Function f8
    z4 = x - self.f_shift[4]
    g4 = self.lamdas[4] * self.g4(z4) + self.bias[4]
    w4 = operator.calculate_weight(x - self.f_shift[4], self.xichmas[4])

    ws = np.array([w0, w1, w2, w3, w4])
    ws = ws / np.sum(ws)
    gs = np.array([g0, g1, g2, g3, g4])
    return np.dot(ws, gs) + self.f_bias

# F10
def f10_evaluate(self, x, *args):
    self.n_fe += 1
    self.check_solution(x, self.dim_max, self.dim_supported)

    # 1. Rotated Schwefel's Function f12
    z0 = x - self.f_shift[0]
    g0 = self.lamdas[0] * operator.modified_schwefel_func(z0) + self.bias[0]
    w0 = operator.calculate_weight(x - self.f_shift[0], self.xichmas[0])

    # 2. Rotated Rastrigin’s Function f4
    z1 = np.dot(self.f_matrix[self.ndim:2*self.ndim, :], (5.12/100)*(x - self.f_shift[1]))
    g1 = self.lamdas[1] * operator.rastrigin_func(z1) + self.bias[1]
    w1 = operator.calculate_weight(x - self.f_shift[1], self.xichmas[1])

    # 3. HGBat Function f7
    z2 = np.dot(self.f_matrix[2*self.ndim:3*self.ndim, :], x - self.f_shift[2])
    g2 = self.lamdas[2] * operator.hgbat_func(z2, shift=-1.0) + self.bias[2]
    w2 = operator.calculate_weight(x - self.f_shift[2], self.xichmas[2])

    ws = np.array([w0, w1, w2])
    ws = ws / np.sum(ws)
    gs = np.array([g0, g1, g2])
    return np.dot(ws, gs) + self.f_bias

# F11
def escaffer6_func(x):
    x = np.asarray(x).ravel()
    x_pairs = np.column_stack((x, np.roll(x, -1)))
    sum_sq = x_pairs[:, 0] ** 2 + x_pairs[:, 1] ** 2
    # Calculate the Schaffer function for all pairs simultaneously
    schaffer_values = (0.5 + (np.sin(np.sqrt(sum_sq)) ** 2 - 0.5) /
                       (1 + 0.001 * sum_sq) ** 2)
    return np.sum(schaffer_values)

def f11_evaluate(self, x, *args):
    self.lamdas = [5e-4, 1, 10, 1, 10]
    self.g0 = escaffer6_func

    self.n_fe += 1
    self.check_solution(x, self.dim_max, self.dim_supported)

    # 1. Expanded Schaffer’s F6 Function f3
    z0 = np.dot(self.f_matrix[:self.ndim, :], (x - self.f_shift[0]))
    g0 = self.lamdas[0] * self.g0(z0) + self.bias[0]
    w0 = operator.calculate_weight(x - self.f_shift[0], self.xichmas[0])

    # 2. Modified Schwefel's Function f12
    z1 = np.dot(self.f_matrix[self.ndim:2*self.ndim, :], x - self.f_shift[1])
    g1 = self.lamdas[1] * self.g1(z1) + self.bias[1]
    w1 = operator.calculate_weight(x - self.f_shift[1], self.xichmas[1])

    # 3. Griewank’s Function f15
    z2 = np.dot(self.f_matrix[2*self.ndim:3*self.ndim, :], 600.*(x - self.f_shift[2])/100)
    g2 = self.lamdas[2] * self.g2(z2) + self.bias[2]
    w2 = operator.calculate_weight(x - self.f_shift[2], self.xichmas[2])

    # 4. Rosenbrock’s Function f2
    z3 = np.dot(self.f_matrix[3 * self.ndim:4 * self.ndim, :], 2.048*(x - self.f_shift[3])/100) + 1.0
    g3 = self.lamdas[3] * self.g3(z3) + self.bias[3]
    w3 = operator.calculate_weight(x - self.f_shift[3], self.xichmas[3])

    # 5. Rastrigin’s Function f4
    z4 = np.dot(self.f_matrix[4 * self.ndim:5 * self.ndim, :], x - self.f_shift[4])
    g4 = self.lamdas[4] * self.g4(z4) + self.bias[4]
    w4 = operator.calculate_weight(x - self.f_shift[4], self.xichmas[4])

    ws = np.array([w0, w1, w2, w3, w4])
    ws = ws / np.sum(ws)
    gs = np.array([g0, g1, g2, g3, g4])
    return np.dot(ws, gs) + self.f_bias

# F12
def f12_evaluate(self, x, *args):
    self.n_fe += 1
    self.check_solution(x, self.dim_max, self.dim_supported)

    # 1. HGBat Function f7
    z0 = np.dot(self.f_matrix[:self.ndim, :], x - self.f_shift[0])
    g0 = self.lamdas[0] * operator.hgbat_func(z0, shift=-1.0) + self.bias[0]
    w0 = operator.calculate_weight(x - self.f_shift[0], self.xichmas[0])

    # 2. Rastrigin’s Function f4
    z1 = np.dot(self.f_matrix[self.ndim:2*self.ndim, :], x - self.f_shift[1])
    g1 = self.lamdas[1] * operator.rastrigin_func(z1) + self.bias[1]
    w1 = operator.calculate_weight(x - self.f_shift[1], self.xichmas[1])

    # 3. Modified Schwefel's Function f12
    z2 = np.dot(self.f_matrix[2*self.ndim:3*self.ndim, :], x - self.f_shift[2])
    g2 = self.lamdas[2] * operator.modified_schwefel_func(z2) + self.bias[2]
    w2 = operator.calculate_weight(x - self.f_shift[2], self.xichmas[2])

    # 4. Bent Cigar Function f6
    z3 = np.dot(self.f_matrix[3 * self.ndim:4 * self.ndim, :], x - self.f_shift[3])
    g3 = self.lamdas[3] * operator.bent_cigar_func(z3) + self.bias[3]
    w3 = operator.calculate_weight(x - self.f_shift[3], self.xichmas[3])

    # 5.  High Conditioned Elliptic Function f8
    z4 = np.dot(self.f_matrix[4 * self.ndim:5 * self.ndim, :], x - self.f_shift[4])
    g4 = self.lamdas[4] * operator.elliptic_func(z4) + self.bias[4]
    w4 = operator.calculate_weight(x - self.f_shift[4], self.xichmas[4])

    # 6.  Expanded Schaffer’s F6 Function f3
    z5 = np.dot(self.f_matrix[5 * self.ndim:6 * self.ndim, :], x - self.f_shift[5])
    g5 = self.lamdas[5] * escaffer6_func(z5) + self.bias[5]
    w5 = operator.calculate_weight(x - self.f_shift[5], self.xichmas[5])

    ws = np.array([w0, w1, w2, w3, w4, w5])
    ws = ws / np.sum(ws)
    gs = np.array([g0, g1, g2, g3, g4, g5])
    return np.dot(ws, gs) + self.f_bias

def adjust_cec_functions():
    # F1
    opfunu.utils.operator.zakharov_func = zakharov_func

    # F3
    opfunu.cec_based.cec2022.F32022.evaluate = f3_evaluate
    opfunu.utils.operator.rotated_expanded_schaffer_func = rotated_expanded_schaffer_func

    # F4
    opfunu.cec_based.cec2022.F42022.evaluate = f4_evaluate
    opfunu.utils.operator.non_continuous_rastrigin_func = non_continuous_rastrigin_func

    # F5
    opfunu.cec_based.cec2022.F52022.evaluate = f5_evaluate

    # F6
    opfunu.cec_based.cec2022.F62022.evaluate = f6_evaluate
    opfunu.utils.operator.hgbat_func = hgbat_func
    opfunu.utils.operator.rastrigin_func = rastrigin_func

    # F7
    opfunu.cec_based.cec2022.F72022.evaluate = f7_evaluate
    opfunu.utils.operator.katsuura_func = katsuura_func
    opfunu.utils.operator.modified_schwefel_func = modified_schwefel_func

    # F8
    opfunu.cec_based.cec2022.F82022.evaluate = f8_evaluate
    opfunu.utils.operator.happy_cat_func = happy_cat_func
    opfunu.utils.operator.grie_rosen_cec_func = grie_rosen_cec_func

    # F9
    opfunu.cec_based.cec2022.F92022.evaluate = f9_evaluate

    # F10
    opfunu.cec_based.cec2022.F102022.evaluate = f10_evaluate

    # F11
    opfunu.cec_based.cec2022.F112022.evaluate = f11_evaluate

    # F12
    opfunu.cec_based.cec2022.F122022.evaluate = f12_evaluate
