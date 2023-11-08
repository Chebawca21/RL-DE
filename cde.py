import numpy as np
from itertools import product
from de import DifferentialEvolution
from SO_BO.CEC2022 import cec2022_func


class CDE(DifferentialEvolution):
    def __init__(self, dimension, FuncNum, population_size, strat_constant, delta):
        self.D = dimension
        self.cec = cec2022_func(FuncNum)
        self.population_size = population_size
        self.n_strategies = 9
        self.F_pool = [0.5, 0.8, 1.0]
        self.cr_pool = [0.0, 0.5, 1.0]
        self.strategies = list(product(self.F_pool, self.cr_pool))
        self.probabilities = np.full(self.n_strategies, 1/self.D)
        self.probabilities /= self.probabilities.sum()
        self.strat_succ = np.zeros(self.n_strategies)
        self.strat_constant = strat_constant
        self.delta = delta
        self.func_evals = 0
        self.best_individual = None
        self.best_score = np.inf
        self.population = self.initializePopulation()
        self.archive = []
        self.evaluate_population()

    def mutation(self, F):
        idxs = np.random.randint(0, self.population_size, 3)
        best_id = np.argmin([self.scores[idxs[0]], self.scores[idxs[1]], self.scores[idxs[2]]])
        mutant = self.difference(self.population[idxs[best_id]], self.population[idxs[1]], self.population[idxs[2]], F)
        return mutant

    def binary_crossover(self, a, b, cr):
        c = np.zeros(self.D)
        l = np.random.randint(0, self.D)
        for i in range(self.D):
            r = np.random.rand()
            if r <= cr or i == l:
                c[i] = b[i]
            else:
                c[i] = a[i]

        return c

    def update_strategies(self):
        strat_sum = sum(self.strat_succ)

        for i in range(self.n_strategies):
            probability = (self.strat_succ[i] + self.strat_constant) / (strat_sum + self.n_strategies * self.strat_constant)
            if probability < self.delta:
                self.probabilities = np.full(self.n_strategies, 1/self.D)
                break
            else:
                self.probabilities[i] = probability
        self.probabilities /= self.probabilities.sum()

    def get_strategy(self):
        return np.random.choice(range(self.n_strategies), p=self.probabilities)

    def step(self):
        new_population = []
        new_scores = []

        for i in range(self.population_size):
            strat_id = self.get_strategy()
            F, cr = self.strategies[strat_id]
            mutant = self.mutation(F)
            candidate = self.binary_crossover(mutant, self.population[i], cr)
            candidate_score = self.evaluate(candidate)
            if candidate_score <= self.scores[i]:
                self.strat_succ[strat_id] += 1
                new_population.append(candidate)
                new_scores.append(candidate_score)
            else:
                new_population.append(self.population[i])
                new_scores.append(self.scores[i])

        self.update_strategies
        self.population = new_population
        self.scores = np.array(new_scores)
        self.update_best_score()
