import numpy as np
from models.rl_hpsde import RL_HPSDE
from qlearning import QLearning

class RL_HPSDE_N_WALKS(RL_HPSDE):
    def __init__(self, dimension, func, max_population_scalar, min_population_size, max_fes, memory_size, num_walks=5, num_steps=200, step_size=10, p=0.1, selection_strategy='boltzmann', archive_size=None):
        self.D = dimension
        self.func = func
        self.population_size =  max_population_scalar * self.D
        self.max_population_size = max_population_scalar * self.D
        self.min_population_size = min_population_size
        self.max_fes = max_fes
        self.rank_greediness_factor = 3
        self.memory_size = memory_size
        self.memory_F = np.full((self.memory_size, 1), 0.5)
        self.memory_cr = np.full((self.memory_size, 1), 0.5)
        self.k = 0
        self.num_walks = num_walks
        self.num_steps = num_steps
        self.step_size = step_size
        self.walks = []
        self.f_walks = []
        self.func_evals = 0
        for _ in range(self.num_walks):
            walk = self.progressive_random_walk()
            f = np.zeros(self.num_steps)
            for i, point in enumerate(walk):
                f[i] = self.evaluate(point)
            self.walks.append(walk)
            self.f_walks.append(f)
        actions = [*range(1, 5)]
        states = [*range(1, 5)]
        self.qlearning = QLearning(states, actions, selection_strategy=selection_strategy)
        self.p = int(p * self.population_size)
        if archive_size is None:
            self.archive_size = self.population_size
        else:
            self.archive_size = archive_size
        self.best_individual = None
        self.best_score = np.inf
        self.population = self.initializePopulation()
        self.archive = []
        self.evaluate_population()
        self.state = self.get_state()

    def get_state(self):
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

    def step(self):
        new_population = np.zeros_like(self.population)
        new_scores = np.zeros(self.population_size)
        S_F, S_cr = [], []
        diffs = []
        prs = self.get_rank_probabilities()

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
                new_population[i] = candidate
                new_scores[i] = candidate_score
            else:
                new_population[i] = self.population[i]
                new_scores[i] = self.scores[i]
            if candidate_score < self.scores[i]:
                self.archive.append(self.population[i])
                diffs.append(self.scores[i] - candidate_score)
                S_F.append(F)
                S_cr.append(cr)

        self.resize_archive()
        if S_F and S_cr:
            self.memory_F[self.k] = self.weighted_lehmer_mean(S_F, diffs)
            self.memory_cr[self.k] = self.weighted_mean(S_cr, diffs)
            self.k += 1
            if self.k >= self.memory_size:
                self.k = 0
        reward = len(S_F) / self.population_size
        next_state = self.get_state()
        self.qlearning.update_qtable(self.state, next_state, action, reward)
        self.state = next_state
        self.population_size, self.population, self.scores = self.adjust_population_size(new_population, new_scores)
        self.update_best_score()

    def next_func_evals(self):
        return self.func_evals + self.population_size