import numpy as np
from itertools import product
from models.l_shade import L_SHADE
from qlearning import QLearning


class RL_SHADE(L_SHADE):
    def __init__(self, dimension, func, r_n_init, min_population_size, max_fes, r_arc, p, n_states=5, actions_interval=0.1, mutation_type='current-to-pbest', selection_strategy='boltzmann'):
        self.D = dimension
        self.func = func
        self.population_size = r_n_init * self.D
        self.max_population_size = r_n_init * self.D
        self.min_population_size = min_population_size
        self.max_fes = max_fes
        self.rank_greediness_factor = 3
        self.r_arc = r_arc
        self.archive_size = int(r_arc * self.population_size)
        self.p = p
        self.mutation_type = mutation_type
        self.n_states = n_states
        self.actions_interval = actions_interval
        self.F_pool = np.arange(0, 1 + actions_interval, actions_interval)
        self.cr_pool = np.arange(0, 1 + actions_interval, actions_interval)
        actions = list(product(self.F_pool, self.cr_pool))
        states = [*range(1, n_states + 1)]
        self.qlearning = QLearning(states, actions, selection_strategy=selection_strategy)
        self.func_evals = 0
        self.best_individual = None
        self.best_score = np.inf
        self.population = self.initializePopulation()
        self.archive = []
        self.evaluate_population()
        self.state = self.get_state()
    
    def generate_F(self, m_F):
        F = -1
        while F <= 0:
            F = np.random.standard_cauchy()
            F = m_F + (F * 0.1)
        if F > 1:
            F = 1
        return F

    def generate_cr(self, m_cr):
        cr = np.random.normal(m_cr, 0.1)
        if cr < 0:
            cr = 0
        elif cr > 1:
            cr = 1
        return cr
    
    def get_state(self):
        state = self.func_evals / (self.max_fes / self.n_states)
        return np.ceil(state)

    def step(self):
        new_population = np.zeros_like(self.population)
        new_scores = np.zeros(self.population_size)
        diffs = []
        prs = self.get_rank_probabilities()

        for i in range(self.population_size):
            m_F, m_cr = self.qlearning.get_action(self.state)
            F = self.generate_F(m_F)
            cr = self.generate_cr(m_cr)

            p = max(2, int(self.p * self.population_size))
            mutant = self.mutation(F, self.mutation_type, current=i, p=p, prs=prs)
            candidate = self.binary_crossover(mutant, self.population[i], cr)
            candidate_score = self.evaluate(candidate)
            next_state = self.get_state()
            if candidate_score < self.scores[i]:
                self.qlearning.update_qtable(self.state, next_state, (m_F, m_cr), 1)
                new_population[i] = candidate
                new_scores[i] = candidate_score
                self.archive.append(self.population[i])
                diffs.append(self.scores[i] - candidate_score)
            else:
                self.qlearning.update_qtable(self.state, next_state, (m_F, m_cr), 0)
                new_population[i] = self.population[i]
                new_scores[i] = self.scores[i]
            self.state = next_state
        
        self.resize_archive()
        self.population_size, self.population, self.scores = self.adjust_population_size(new_population, new_scores)
        self.archive_size = int(self.r_arc * self.population_size)
        self.resize_archive()
        self.update_best_score()