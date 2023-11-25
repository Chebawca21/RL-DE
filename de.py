import numpy as np
from SO_BO.CEC2022 import cec2022_func


class DifferentialEvolution:
    def __init__(self, dimension, func_num, population_size, F, cr, mutation_type='rand', crossover_type='bin', p=0.1, archive_size=None):
        self.D = dimension
        self.func_num = func_num
        self.cec = cec2022_func(self.func_num)
        self.population_size = population_size
        self.F = F
        self.cr = cr
        self.mutation_type = mutation_type
        self.crossover_type = crossover_type
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

    def initializePopulation(self):
        return 200.0 * np.random.rand(self.population_size, self.D) - 100.0

    def mutation(self, F, type='best', current=None, p=None):
        if type == 'rand':
            idxs = np.random.randint(0, self.population_size, 3)
            mutant = self.difference(self.population[idxs[0]], self.population[idxs[1]], self.population[idxs[2]], F)
        if type == 'best':
            idxs = np.random.randint(0, self.population_size, 2)
            curr_best = np.argmin(self.scores)
            mutant = self.difference(self.population[curr_best], self.population[idxs[0]], self.population[idxs[1]], F)
        if type == 'current-to-best':
            idxs = np.random.randint(0, self.population_size, 2)
            curr_best = np.argmin(self.scores)
            mutant = self.difference(self.population[current], self.population[idxs[0]], self.population[idxs[1]], F)
            mutant = self.difference(mutant, self.population[curr_best], self.population[current], F)
        if type == 'randrl':
            idxs = np.random.randint(0, self.population_size, 3)
            best_id = np.argmin([self.scores[idxs[0]], self.scores[idxs[1]], self.scores[idxs[2]]])
            mutant = self.difference(self.population[idxs[best_id]], self.population[idxs[1]], self.population[idxs[2]], F)
        if type == 'current-to-pbest':
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

    def difference(self, a, b, c, F):
        return np.clip(a + F * (b - c), -100, 100)

    def crossover(self, a, b, cr, type='bin'):
        if type == 'bin':
            return self.binary_crossover(a, b, cr)
        if type == 'exp':
            return self.exponential_crossover(a, b, cr)

    def binary_crossover(self, a, b, cr):
        c = np.zeros(self.D)
        for i in range(self.D):
            r = np.random.rand()
            if r < cr:
                c[i] = b[i]
            else:
                c[i] = a[i]

        return c

    def exponential_crossover(self, a, b, cr):
        i = np.random.randint(0, self.D)
        copied = 0
        c = a.copy()
        while copied < self.D:
            r = np.random.rand()
            print(r)
            if r < cr:
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
        self.update_best_score()

    def update_best_score(self):
        for i in range(self.population_size):
            if self.scores[i] < self.best_score:
                self.best_score = self.scores[i]
                self.best_individual = self.population[i]

    def step(self):
        new_population = []
        new_scores = []

        for i in range(self.population_size):
            if self.mutation_type == 'current-to-best':
                mutant = self.mutation(self.F, self.mutation_type, i)
            elif self.mutation_type == 'current-to-pbest':
                mutant = self.mutation(self.F, self.mutation_type, self.p)
            else:
                mutant = self.mutation(self.F, self.mutation_type)
            candidate = self.crossover(mutant, self.population[i], self.cr, self.crossover_type)
            candidate_score = self.evaluate(candidate)
            if candidate_score <= self.scores[i]:
                new_population.append(candidate)
                new_scores.append(candidate_score)
                self.archive.append(self.population[i])
            else:
                new_population.append(self.population[i])
                new_scores.append(self.scores[i])

        if len(self.archive) > self.archive_size:
            np.random.shuffle(self.archive)
            self.archive = self.archive[:self.archive_size]
        self.population = new_population
        self.scores = np.array(new_scores)
        self.update_best_score()
