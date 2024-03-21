import numpy as np
from de import DifferentialEvolution
from qlearning import QLearning
from SO_BO.CEC2022 import cec2022_func
from scipy.stats import levy
import matplotlib.pyplot as plt

class RL_HPSDE(DifferentialEvolution):
    def __init__(self, dimension, func_num, population_size, memory_size, num_steps=200, step_size=10, p=0.1, archive_size=None):
        self.D = dimension
        self.func_num = func_num
        self.cec = cec2022_func(self.func_num)
        self.population_size = population_size
        self.memory_size = memory_size
        self.memory_F = np.full((self.memory_size, 1), 0.5)
        self.memory_cr = np.full((self.memory_size, 1), 0.5)
        self.k = 0
        self.num_steps = num_steps
        self.step_size = step_size
        actions = [*range(1, 5)]
        states = [*range(1, 5)]
        self.qlearning = QLearning(states, actions, selection_strategy='boltzmann')
        self.p = int(p * population_size)
        if archive_size is None:
            self.archive_size = population_size
        else:
            self.archive_size = archive_size
        self.func_evals = 0
        self.best_individual = None
        self.best_score = np.inf
        self.population = self.initializePopulation()
        self.archive = []
        self.evaluate_population()
        self.state = self.get_state()
    
    def progressive_random_walk(self):
        x_min = -100
        x_max = 100
        starting_zone = np.random.randint(2, size=self.D)
        walk = np.zeros((self.num_steps, self.D))
        for i in range(self.D):
            r = np.random.rand()
            r *= (x_max - x_min) / 2
            if starting_zone[i] == 1:
                walk[0][i] = x_max - r
            else:
                walk[0][i] = x_min + r
        r_D = np.random.randint(0, self.D)
        if starting_zone[r_D] == 1:
            walk[0][r_D] = x_max
        else:
            walk[0][r_D] = x_min
        for s in range(self.num_steps):
            for i in range(self.D):
                r = np.random.rand()
                r *= self.step_size
                if starting_zone[i] == 1:
                    r = -r
                walk[s][i] = walk[s - 1][i] + r
                if walk[s][i] > x_max:
                    walk[s][i] = x_max - (walk[s][i] - x_max)
                    starting_zone[i] = 1 - starting_zone[i]
                elif walk[s][i] < x_min:
                    walk[s][i] = x_min + (x_min - walk[s][i])
                    starting_zone[i] = 1 - starting_zone[i]
        return walk

    def draw_walk(self, walk):
        x = walk[:, 0]
        y = walk[:, 1]
        plt.plot(x, y, marker='s', linewidth=1, markersize=3)
        plt.show()

    def dfdc(self, walk, f):
        self.func_evals += self.num_steps

        d = np.linalg.norm(walk - self.best_individual, ord=2, axis=-1)

        f_average = np.average(f)
        d_average = np.average(d)
        f_std = np.std(f)
        d_std = np.std(d)

        dfdc = np.sum((f - f_average) * (d - d_average)) * (1 / (self.num_steps * f_std * d_std))
        return dfdc

    def drie(self, f):
        diff = f[1:] - f[:-1]
        e_star = np.max(np.abs(diff))
        all_e = np.array([1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1]) * e_star
        information_entropies = []
        for e in all_e:
            symbols = np.zeros_like(diff)
            for i in range(len(symbols)):
                if diff[i] < -e:
                    symbols[i] = -1
                elif np.abs(diff[i]) <= e:
                    symbols[i] = 0
                elif diff[i] > e:
                    symbols[i] = 1
            first_symbols = symbols[:-1]
            second_symbols = symbols[1:]
            probabilities = np.zeros(6)
            for i in range(len(first_symbols)):
                if first_symbols[i] == -1 and second_symbols[i] == 0:
                    probabilities[0] += 1
                if first_symbols[i] == -1 and second_symbols[i] == 1:
                    probabilities[1] += 1
                if first_symbols[i] == 0 and second_symbols[i] == -1:
                    probabilities[2] += 1
                if first_symbols[i] == 0 and second_symbols[i] == 1:
                    probabilities[3] += 1
                if first_symbols[i] == 1 and second_symbols[i] == -1:
                    probabilities[4] += 1
                if first_symbols[i] == 1 and second_symbols[i] == 0:
                    probabilities[5] += 1
            probabilities /= len(symbols)
            probabilities[probabilities < 1e-15] = 1e-15
            H = -np.sum(probabilities * np.emath.logn(6, probabilities))
            information_entropies.append(H)
        return np.max(information_entropies)

    def get_state(self):
        self.func_evals = self.func_evals + self.num_steps
        walk = self.progressive_random_walk()
        f = self.cec.values(walk)
        dfdc = self.dfdc(walk, f)
        drie = self.drie(f)
        if drie < 0.5 and dfdc <= 0.15:
            return 1
        elif drie < 0.5 and dfdc > 0.15:
            return 2
        elif drie >= 0.5 and dfdc <= 0.15:
            return 3
        elif drie >= 0.5 and dfdc > 0.15:
            return 4
    
    def generate_F_cauchy(self):
        F = -1
        r = np.random.randint(0, self.memory_size)
        while F <= 0:
            F = np.random.standard_cauchy()
            F = self.memory_F[r] + (F * 0.1)
        if F > 1:
            F = 1
        return F

    def generate_F_levy(self):
        F = -1
        r = np.random.randint(0, self.memory_size)
        while F <= 0:
            F = levy.rvs(loc=self.memory_F[r], scale=0.1, size=1)[0]
        if F > 1:
            F = 1
        return F

    def generate_cr(self):
        cr = -1
        r = np.random.randint(0, self.memory_size)
        while cr <= 0:
            cr = np.random.normal(self.memory_cr[r], 0.1)
        if cr > 1:
            cr = 1
        return cr

    def weighted_lehmer_mean(self, list, diffs):
        n = len(list)
        mean = 0.0
        squared_mean = 0.0
        for value, diff in zip(list, diffs):
            weight = diff / (n * diff)
            mean += weight * value
            squared_mean += weight * (value ** 2)
        return squared_mean / mean

    def weighted_mean(self, list, diffs):
        n = len(list)
        mean = 0.0
        for value, diff in zip(list, diffs):
            weight = diff / (n * diff)
            mean += weight * value
        return mean

    def step(self):
        new_population = []
        new_scores = []
        S_F, S_cr = [], []
        diffs = []

        action = self.qlearning.get_action(self.state)
        for i in range(self.population_size):
            cr = self.generate_cr()
            if action == 1:
                F = self.generate_F_cauchy()
                mutant = self.mutation(F, 'current-to-rand', i)
            elif action == 2:
                F = self.generate_F_cauchy()
                mutant = self.mutation(F, 'current-to-best', i)
            elif action == 3:
                F = self.generate_F_levy()
                mutant = self.mutation(F, 'current-to-rand', i)
            elif action == 4:
                F = self.generate_F_levy()
                mutant = self.mutation(F, 'current-to-best', i)

            candidate = self.binary_crossover(mutant, self.population[i], cr)
            candidate_score = self.evaluate(candidate)
            if candidate_score <= self.scores[i]:
                new_population.append(candidate)
                new_scores.append(candidate_score)
            else:
                # self.qlearning.update_qtable(self.state, self.state, (F, cr), -0.5)
                new_population.append(self.population[i])
                new_scores.append(self.scores[i])
            if candidate_score < self.scores[i]:
                self.archive.append(self.population[i])
                diffs.append(self.scores[i] - candidate_score)
                S_F.append(F)
                S_cr.append(cr)

        if len(self.archive) > self.archive_size:
            np.random.shuffle(self.archive)
            self.archive = self.archive[:self.archive_size]
        if S_F and S_cr:
            self.memory_F[self.k] = self.weighted_lehmer_mean(S_F, diffs)
            self.memory_cr[self.k] = self.weighted_mean(S_cr, diffs)
            self.k += 1
            if self.k >= self.memory_size:
                self.k = 0
        self.population = new_population
        self.scores = np.array(new_scores)
        self.update_best_score()
        reward = len(S_F) / self.population_size
        next_state = self.get_state()
        self.qlearning.update_qtable(self.state, next_state, action, reward)

    def next_func_evals(self):
        return self.func_evals + self.population_size + self.num_steps

# rl_hpsde = RL_HPSDE(2, 10, 50, 5)
# walk = rl_hpsde.progressive_random_walk()
# rl_hpsde.draw_walk(walk)