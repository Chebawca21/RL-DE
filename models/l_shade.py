import numpy as np
from models.shade import SHADE


class L_SHADE(SHADE):
    def __init__(self, dimension, func, r_n_init, min_population_size, max_fes, memory_size, r_arc, p, mutation_type='current-to-pbest'):
        self.D = dimension
        self.func = func
        self.population_size = r_n_init * self.D
        self.max_population_size = r_n_init * self.D
        self.min_population_size = min_population_size
        self.max_fes = max_fes
        self.rank_greediness_factor = 3
        self.memory_size = memory_size
        self.memory_F = np.full((self.memory_size, 1), 0.5)
        self.memory_cr = np.full((self.memory_size, 1), 0.5)
        self.k = 0
        self.r_arc = r_arc
        self.archive_size = int(r_arc * self.population_size)
        self.p = p
        self.mutation_type = mutation_type
        self.func_evals = 0
        self.best_individual = None
        self.best_score = np.inf
        self.population = self.initializePopulation()
        self.archive = []
        self.evaluate_population()

    def adjust_population_size(self, new_population, new_scores):
        new_population_size = np.round(((self.min_population_size - self.max_population_size) / self.max_fes * self.func_evals) + self.max_population_size)
        new_population_size = max(int(new_population_size), 1)
        optimal = sorted(zip(new_scores, new_population), key=lambda x: x[0])[:new_population_size]
        new_scores, new_population = zip(*optimal)
        new_population = np.array(new_population)
        new_scores = np.array(new_scores)
        return new_population_size, new_population, new_scores

    def step(self):
        new_population = np.zeros_like(self.population)
        new_scores = np.zeros(self.population_size)
        S_F, S_cr = [], []
        diffs = []
        prs = self.get_rank_probabilities()

        for i in range(self.population_size):
            F = self.generate_F()
            cr = self.generate_cr()

            p = max(2, int(self.p * self.population_size))
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
        self.archive_size = int(self.r_arc * self.population_size)
        self.resize_archive()
        self.update_best_score()
