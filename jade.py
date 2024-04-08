import numpy as np
from de import DifferentialEvolution


class JADE(DifferentialEvolution):
    def __init__(self, dimension, func, population_size, archive_size, p, c, mutation_type='current-to-pbest'):
        self.D = dimension
        self.func = func
        self.population_size = population_size
        self.archive_size = archive_size
        self.mean_F = 0.5
        self.mean_cr = 0.5
        self.p = int(p * population_size)
        self.c = c
        self.mutation_type = mutation_type
        self.func_evals = 0
        self.best_individual = None
        self.best_score = np.inf
        self.population = self.initializePopulation()
        self.archive = []
        self.evaluate_population()

    def generate_F(self):
        F = -1
        while F <= 0:
            F = np.random.standard_cauchy()
            F = self.mean_F + (F * 0.1)
        if F > 1:
            F = 1
        return F

    def generate_cr(self):
        cr = -1
        while cr <= 0:
            cr = np.random.normal(self.mean_cr, 0.1)
        if cr > 1:
            cr = 1
        return cr

    def lehmer_mean(self, list):
        if not list:
            return 0
        return sum([x ** 2 for x in list]) / sum(list) 

    def mean(self, list):
        if not list:
            return 0
        return sum(list) / len(list)

    def step(self):
        new_population = []
        new_scores = []
        S_F, S_cr = [], []

        for i in range(self.population_size):
            F = self.generate_F()
            cr = self.generate_cr()

            mutant = self.mutation(F, self.mutation_type, current=i, p=self.p)
            candidate = self.binary_crossover(mutant, self.population[i], cr)
            candidate_score = self.evaluate(candidate)
            if candidate_score <= self.scores[i]:
                new_population.append(candidate)
                new_scores.append(candidate_score)
                self.archive.append(self.population[i])
                S_F.append(F)
                S_cr.append(cr)
            else:
                new_population.append(self.population[i])
                new_scores.append(self.scores[i])
        
        self.resize_archive()
        self.mean_F = ((1 - self.c) * self.mean_F) + (self.c * self.lehmer_mean(S_F))
        self.mean_cr = ((1 - self.c) * self.mean_cr) + (self.c * self.mean(S_cr))
        self.population = new_population
        self.scores = np.array(new_scores)
        self.update_best_score()
