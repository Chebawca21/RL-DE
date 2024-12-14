import numpy as np
from itertools import product
from models.de import DifferentialEvolution
from qlearning import QLearning

class QDE(DifferentialEvolution):
    def __init__(self, dimension, func, population_size, max_fes, mutation_type='rand', p=0.1, selection_strategy='boltzmann', actions='cde', states='qde', archive_size=None):
        self.D = dimension
        self.func = func
        self.population_size = population_size
        self.rank_greediness_factor = 3
        if actions == 'qlde':
            actions = [(0.4, 0.7), (0.6, 0.7), (0.8, 0.7), (0.4, 0.9), (0.6, 0.9), (0.8, 0.9), (0.9, 0.9)]
        else:
            self.F_pool = [0.5, 0.8, 1.0]
            self.cr_pool = [0.0, 0.5, 1.0]
            actions = list(product(self.F_pool, self.cr_pool))
        self.states = states
        if self.states == 'rl-hpsde-n-walks':
            self.num_walks = 0.5 * self.D
            self.num_steps = 200
            self.step_size = 10
            self.walks = []
            self.f_walks = []
            for _ in range(self.num_walks):
                walk = self.progressive_random_walk()
                f = np.zeros(self.num_steps)
                for i, point in enumerate(walk):
                    f[i] = self.evaluate(point)
                self.walks.append(walk)
                self.f_walks.append(f)
                states = [*range(1, 5)]
        elif self.states == 'rl-shade':
            self.n_states = 10
            states = [*range(1, self.n_states + 1)]
        else:
            states = [0]
        self.qlearning = QLearning(states, actions, selection_strategy=selection_strategy)
        self.mutation_type = mutation_type
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

    def dfdc(self, walk, f):
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
        if self.states == 'rl-hpsde-n-walks':
            r = np.random.randint(0, self.num_walks)
            walk = self.walks[r]
            f = self.f_walks[r]
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
        elif self.states == 'rl-shade':
            state = self.func_evals / (self.max_fes / self.n_states)
            return np.ceil(state)
        else:
            return 0

    def step(self):
        new_population = np.zeros_like(self.population)
        new_scores = np.zeros(self.population_size)
        prs = self.get_rank_probabilities()

        for i in range(self.population_size):
            F, cr = self.qlearning.get_action(self.state)
            mutant = self.mutation(F, self.mutation_type, current=i, p=self.p, prs=prs)
            candidate = self.binary_crossover(mutant, self.population[i], cr)
            candidate_score = self.evaluate(candidate)
            next_state = self.get_state()
            if candidate_score <= self.scores[i]:
                self.qlearning.update_qtable(self.state, next_state, (F, cr), 1)
                new_population[i] = candidate
                new_scores[i] = candidate_score
                self.archive.append(self.population[i])
            else:
                self.qlearning.update_qtable(self.state, next_state, (F, cr), 0)
                new_population[i] = self.population[i]
                new_scores[i] = self.scores[i]
            self.state = next_state

        self.resize_archive()
        self.population = new_population
        self.scores = new_scores
        self.update_best_score()