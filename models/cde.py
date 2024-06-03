import numpy as np
from itertools import product
from models.de import DifferentialEvolution


class CDE(DifferentialEvolution):
    def __init__(self, dimension, func, population_size, strat_constant, inverse_delta, mutation_type='randrl', p=0.1, archive_size=None):
        self.D = dimension
        self.func = func
        self.population_size = population_size
        self.rank_greediness_factor = 3
        self.n_strategies = 9
        self.F_pool = [0.5, 0.8, 1.0]
        self.cr_pool = [0.0, 0.5, 1.0]
        self.strategies = list(product(self.F_pool, self.cr_pool))
        self.probabilities = np.full(self.n_strategies, 1/self.D)
        self.probabilities /= self.probabilities.sum()
        self.strat_succ = np.zeros(self.n_strategies)
        self.strat_constant = strat_constant
        self.delta = 1 / inverse_delta
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
        new_population = np.zeros_like(self.population)
        new_scores = np.zeros(self.population_size)
        prs = self.get_rank_probabilities()

        for i in range(self.population_size):
            strat_id = self.get_strategy()
            F, cr = self.strategies[strat_id]
            mutant = self.mutation(F, self.mutation_type, current=i, p=self.p, prs=prs)
            candidate = self.binary_crossover(mutant, self.population[i], cr)
            candidate_score = self.evaluate(candidate)
            if candidate_score <= self.scores[i]:
                self.strat_succ[strat_id] += 1
                new_population[i] = candidate
                new_scores[i] = candidate_score
                self.archive.append(self.population[i])
            else:
                new_population[i] = self.population[i]
                new_scores[i] = self.scores[i]

        self.resize_archive()
        self.update_strategies
        self.population = new_population
        self.scores = new_scores
        self.update_best_score()
