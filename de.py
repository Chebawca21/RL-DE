import numpy as np
from SO_BO.CEC2022 import cec2022_func


class DifferentialEvolution:
    def __init__(self, dimension, FuncNum, population_size, F, cr, mutation_type='random', crossover_type='bin'):
        self.D = dimension
        self.cec = cec2022_func(FuncNum)
        self.population_size = population_size
        self.F = F
        self.cr = cr
        self.mutation_type = mutation_type
        self.crossover_type = crossover_type
        self.func_evals = 0
        self.best_individual = None
        self.best_score = np.inf
        self.population = self.initializePopulation()
        self.evaluate_population()

    def initializePopulation(self):
        return 200.0 * np.random.rand(self.population_size, self.D) - 100.0

    def mutation(self, type='best', current=None):
        if type == 'random':
            idxs = np.random.randint(0, self.population_size, 3)
            mutant = self.difference(self.population[idxs[0]], self.population[idxs[1]], self.population[idxs[2]])
        if type == 'best':
            idxs = np.random.randint(0, self.population_size, 2)
            curr_best = np.argmin(self.scores)
            mutant = self.difference(self.population[curr_best], self.population[idxs[0]], self.population[idxs[1]])
        if type == 'current-to-best':
            idxs = np.random.randint(0, self.population_size, 2)
            curr_best = np.argmin(self.scores)
            # mutant = self.difference(self.population[current], self.population[idxs[0]], self.population[idxs[1]])
            # mutant = self.difference(mutant, self.population[curr_best], self.population[current])
            mutant = self.difference(self.difference(self.population[current], self.population[curr_best], self.population[current]), self.population[idxs[0]], self.population[idxs[1]])
        return mutant

    def difference(self, a, b, c):
        return [a[i] + self.F * (b[i] - c[i]) for i in range(self.D)]

    def crossover(self, a, b, type='bin'):
        if type == 'bin':
            return self.binary_crossover(a, b)
        if type == 'exp':
            return self.exponential_crossover(a, b)

    def binary_crossover(self, a, b):
        c = np.zeros(self.D)
        for i in range(self.D):
            r = np.random.rand()
            if r < self.cr:
                c[i] = b[i]
            else:
                c[i] = a[i]

        return c

    def exponential_crossover(self, a, b):
        i = np.random.randint(0, self.D)
        copied = 0
        c = a.copy()
        while copied < self.D:
            r = np.random.rand()
            print(r)
            if r < self.cr:
                c[i] = b[i]
                copied = copied + 1
                i = i + 1
                if i == self.D:
                    i = 0
            else:
                break

        return c

    def evaluate(self, value):
        self.func_evals = self.func_evals + 1
        return self.cec.value(value)

    def evaluate_population(self):
        self.func_evals = self.func_evals + self.population_size
        self.scores = self.cec.values(self.population)

        for i in range(self.population_size):
            if self.scores[i] < self.best_score:
                self.best_score = self.scores[i]
                self.best_individual = self.population[i]

    def step(self):
        new_population = []

        for i in range(self.population_size):
            if self.mutation_type == 'current-to-best':
                mutant = self.mutation(self.mutation_type, i)
            else:
                mutant = self.mutation(self.mutation_type)
            candidate = self.crossover(mutant, self.population[i])
            if self.evaluate(candidate) <= self.scores[i]:
                new_population.append(candidate)
            else:
                new_population.append(self.population[i])

        self.population = new_population
        self.evaluate_population()

