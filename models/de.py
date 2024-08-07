import numpy as np

class DifferentialEvolution:
    def __init__(self, dimension, func, population_size, F, cr, mutation_type='rand', crossover_type='bin', p=0.1, archive_size=None):
        self.D = dimension
        self.func = func
        self.population_size = population_size
        self.rank_greediness_factor = 3
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

    def get_rank_probabilities(self):
        best_idxs = (-self.scores).argsort()
        ranks = np.zeros(self.population_size)
        for i, idx in enumerate(best_idxs):
            ranks[idx] = self.rank_greediness_factor * (self.population_size - i - 1) + 1
        prs = np.zeros(self.population_size)
        ranks_sum = sum(ranks)
        for i in range(self.population_size):
            prs[i] = ranks[i] / ranks_sum
        return prs

    def mutation(self, F, type='best', current=None, p=None, prs=None, Fw=1):
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
        if type == 'current-to-rand':
            idxs = np.random.randint(0, self.population_size, 3)
            mutant = self.difference(self.population[current], self.population[current], self.population[idxs[0]], F)
            mutant = self.difference(mutant, self.population[idxs[1]], self.population[idxs[2]], F)
        if type == 'randrl':
            idxs = np.random.randint(0, self.population_size, 3)
            best_id = np.argmin([self.scores[idxs[0]], self.scores[idxs[1]], self.scores[idxs[2]]])
            mutant = self.difference(self.population[idxs[best_id]], self.population[idxs[1]], self.population[idxs[2]], F)
        if type == 'current-to-pbest':
            idx0 = np.random.randint(0, self.population_size)
            random0 = self.population[idx0]
            r = np.random.rand()
            if r < len(self.archive) / (len(self.archive) + self.population_size):
                idx1 = np.random.randint(0, len(self.archive))
                random1 = self.archive[idx1]
            else:
                idx1 = np.random.randint(0, self.population_size)
                random1 = self.population[idx1]
            p_best_idxs = self.scores.argsort()[:p]
            p_best = np.random.randint(0, len(p_best_idxs))
            mutant = self.difference(self.population[current], self.population[p_best_idxs[p_best]], self.population[current], F)
            mutant = self.difference(mutant, random0, random1, F)
        if type == 'current-to-pbest-r':
            idx0 = np.random.choice(self.population_size, p=prs)
            random0 = self.population[idx0]
            r = np.random.rand()
            if r < len(self.archive) / (len(self.archive) + self.population_size):
                idx1 = np.random.randint(0, len(self.archive))
                random1 = self.archive[idx1]
            else:
                idx1 = np.random.choice(self.population_size, p=prs)
                random1 = self.population[idx1]
            p_best_idxs = self.scores.argsort()[:p]
            p_best = np.random.randint(0, len(p_best_idxs))
            mutant = self.difference(self.population[current], self.population[p_best_idxs[p_best]], self.population[current], Fw)
            mutant = self.difference(mutant, random0, random1, F)
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
        l = np.random.randint(0, self.D)
        for i in range(self.D):
            r = np.random.rand()
            if r <= cr or i == l:
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
        return self.func.evaluate(value)

    def evaluate_population(self):
        self.func_evals = self.func_evals + self.population_size
        self.scores = np.zeros(self.population_size)
        for i, individual in enumerate(self.population):
            self.scores[i] = self.func.evaluate(individual)
        self.update_best_score()

    def resize_archive(self):
        if len(self.archive) > self.archive_size:
            np.random.shuffle(self.archive)
            self.archive = self.archive[:self.archive_size]

    def update_best_score(self):
        for i in range(self.population_size):
            if self.scores[i] < self.best_score:
                self.best_score = self.scores[i]
                self.best_individual = self.population[i]

    def step(self):
        new_population = np.zeros_like(self.population)
        new_scores = np.zeros(self.population_size)
        prs = self.get_rank_probabilities()

        for i in range(self.population_size):
            mutant = self.mutation(self.F, self.mutation_type, current=i, p=self.p, prs=prs)
            candidate = self.crossover(mutant, self.population[i], self.cr, self.crossover_type)
            candidate_score = self.evaluate(candidate)
            if candidate_score <= self.scores[i]:
                new_population[i] = candidate
                new_scores[i] = candidate_score
                self.archive.append(self.population[i])
            else:
                new_population[i] = self.population[i]
                new_scores[i] = self.scores[i]

        self.resize_archive()
        self.population = new_population
        self.scores = new_scores
        self.update_best_score()

    def next_func_evals(self):
        return self.func_evals + self.population_size

    def evolve(self, max_fes):
        while self.func_evals <= max_fes:
            if self.next_func_evals() > max_fes:
                break
            self.step()
        return self.best_score
