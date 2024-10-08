import numpy as np
from models.jade import JADE


class SHADE(JADE):
    def __init__(self, dimension, func, population_size, memory_size, archive_size, mutation_type='current-to-pbest'):
        self.D = dimension
        self.func = func
        self.population_size = population_size
        self.rank_greediness_factor = 3
        self.memory_size = memory_size
        self.memory_F = np.full((self.memory_size, 1), 0.5)
        self.memory_cr = np.full((self.memory_size, 1), 0.5)
        self.k = 0
        self.archive_size = archive_size
        self.mutation_type = mutation_type
        self.func_evals = 0
        self.best_individual = None
        self.best_score = np.inf
        self.population = self.initializePopulation()
        self.archive = []
        self.evaluate_population()

    def generate_F(self):
        F = -1
        r = np.random.randint(0, self.memory_size)
        while F <= 0:
            F = np.random.standard_cauchy()
            F = self.memory_F[r] + (F * 0.1)
        if F > 1:
            F = 1
        return F

    def generate_cr(self):
        r = np.random.randint(0, self.memory_size)
        cr = np.random.normal(self.memory_cr[r], 0.1)
        if cr < 0:
            cr = 0
        elif cr > 1:
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
        new_population = np.zeros_like(self.population)
        new_scores = np.zeros(self.population_size)
        S_F, S_cr = [], []
        diffs = []
        prs = self.get_rank_probabilities()

        for i in range(self.population_size):
            F = self.generate_F()
            cr = self.generate_cr()

            p = np.random.randint(2, int(0.2 * self.population_size))
            mutant = self.mutation(F, self.mutation_type, current=i, p=p, prs=prs)
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
        self.population = new_population
        self.scores = new_scores
        self.update_best_score()
