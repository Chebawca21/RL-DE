import numpy as np
from models.l_shade import L_SHADE


class L_SHADE_RSP(L_SHADE):
    def __init__(self, dimension, func, max_population_size, min_population_size, max_fes, memory_size, archive_size, mutation_type='current-to-pbest-r'):
        self.D = dimension
        self.func = func
        self.population_size = max_population_size
        self.max_population_size = max_population_size
        self.min_population_size = min_population_size
        self.max_fes = max_fes
        self.rank_greediness_factor = 3
        self.memory_size = memory_size
        self.memory_F = np.full((self.memory_size, 1), 0.3)
        self.memory_cr = np.full((self.memory_size, 1), 0.8)
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
        r = np.random.randint(0, self.memory_size + 1)
        if r == self.memory_size:
            mean = 0.9
        else:
            mean = self.memory_F[r]

        while F <= 0:
            F = np.random.standard_cauchy()
            F = mean + (F * 0.1)
        if F > 1:
            F = 1
        return F

    def generate_cr(self):
        cr = -1
        r = np.random.randint(0, self.memory_size + 1)
        if r == self.memory_size:
            mean = 0.9
        else:
            mean = self.memory_cr[r]

        while cr <= 0:
            cr = np.random.normal(mean, 0.1)
        if cr > 1:
            cr = 1

        if self.func_evals < 0.25 * self.max_fes:
            cr = max(cr, 0.7)
        elif self.func_evals < 0.5 * self.max_fes:
            cr = max(cr, 0.6)
        return cr

    def step(self):
        new_population = np.zeros_like(self.population)
        new_scores = np.zeros(self.population_size)
        S_F, S_cr = [], []
        diffs = []
        prs = self.get_rank_probabilities()

        for i in range(self.population_size):
            F = self.generate_F()
            cr = self.generate_cr()

            p_high_bound = max(2, int(0.2 * self.population_size))
            if p_high_bound <= 2:
                p = 2
            else:
                p = np.random.randint(2, p_high_bound)
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
        self.population_size, self.population, self.scores = self.adjust_population_size(new_population, new_scores)
        self.archive_size = self.population_size
        self.resize_archive()
        self.update_best_score()
