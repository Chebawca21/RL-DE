import numpy as np
from itertools import product
from de import DifferentialEvolution


class CDE(DifferentialEvolution):
    def __init__(self, dimension, func, population_size, strat_constant, delta, mutation_type='randrl', p=0.1, archive_size=None):
        self.D = dimension
        self.func = func
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
            if self.mutation_type == 'current-to-best' or self.mutation_type == 'current-to-rand':
                mutant = self.mutation(F, self.mutation_type, i)
            elif self.mutation_type == 'current-to-pbest':
                mutant = self.mutation(F, self.mutation_type, self.p)
            else:
                mutant = self.mutation(F, self.mutation_type)
            candidate = self.binary_crossover(mutant, self.population[i], cr)
            candidate_score = self.evaluate(candidate)
            if candidate_score <= self.scores[i]:
                self.strat_succ[strat_id] += 1
                new_population.append(candidate)
                new_scores.append(candidate_score)
                self.archive.append(self.population[i])
            else:
                new_population.append(self.population[i])
                new_scores.append(self.scores[i])

        if len(self.archive) > self.archive_size:
            np.random.shuffle(self.archive)
            self.archive = self.archive[:self.archive_size]
        self.update_strategies
        self.population = new_population
        self.scores = np.array(new_scores)
        self.update_best_score()
