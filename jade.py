import numpy as np
from scipy.stats import cauchy
from de import DifferentialEvolution
from SO_BO.CEC2022 import cec2022_func


class JADE(DifferentialEvolution):
    def __init__(self, dimension, FuncNum, population_size, archive_size, p, c):
        self.D = dimension
        self.cec = cec2022_func(FuncNum)
        self.population_size = population_size
        self.archive_size = archive_size
        self.mean_F = 0.5
        self.mean_cr = 0.5
        self.p = int(p * population_size)
        self.c = c
        self.func_evals = 0
        self.best_individual = None
        self.best_score = np.inf
        self.population = self.initializePopulation()
        self.archive = []
        self.evaluate_population()

    def mutation(self, current, F, p):
        idx0 = np.random.randint(0, self.population_size)
        if not self.archive:
            pop_and_arch = np.concatenate((self.population, np.reshape([], (0, self.D))))
        else:
            pop_and_arch = np.concatenate((self.population, self.archive))
        idx1 = np.random.randint(0, len(pop_and_arch))
        p_best_idxs = self.scores.argsort()[:p]
        p_best = np.random.randint(0, len(p_best_idxs))
        mutant = self.difference(self.population[current], self.population[p_best_idxs[p_best]], self.population[current], F)
        mutant = self.difference(mutant, self.population[idx0], pop_and_arch[idx1], F)
        return mutant

    def generate_F(self):
        F = -1
        while F <= 0:
            F = cauchy.rvs(self.mean_F, 0.1)
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
        S_F, S_cr = [], []

        for i in range(self.population_size):
            F = self.generate_F()
            cr = self.generate_cr()

            mutant = self.mutation(i, F, self.p)
            candidate = self.binary_crossover(mutant, self.population[i], cr)
            if self.evaluate(candidate) <= self.scores[i]:
                new_population.append(candidate)
                self.archive.append(self.population[i])
                S_F.append(F)
                S_cr.append(cr)
            else:
                new_population.append(self.population[i])
        
        if len(self.archive) > self.archive_size:
            np.random.shuffle(self.archive)
            self.archive = self.archive[:self.archive_size]
        self.mean_F = ((1 - self.c) * self.mean_F) + (self.c * self.lehmer_mean(S_F))
        self.mean_cr = ((1 - self.c) * self.mean_cr) + (self.c * self.mean(S_cr))
        self.population = new_population
        self.evaluate_population()
