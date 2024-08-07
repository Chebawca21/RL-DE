import numpy as np
import matplotlib.pyplot as plt
from de import DifferentialEvolution
import time

class DrawDE():
    def __init__(self, func_num, population_size, F, cr, x_min, x_max, x_step, max_fes, fes_step):
        self.de = DifferentialEvolution(2, func_num, population_size, F, cr)
        self.x_min = x_min
        self.x_max = x_max
        self.x_step = x_step
        self.max_fes = max_fes
        self.fes_step = fes_step

        self.initialize_function()

        plt.ion()
        self.fig = plt.figure()
        self.draw(0, self.de.best_score)
        time.sleep(10)

    def initialize_function(self):
        self.x1, self.x2 = np.meshgrid(np.arange(self.x_min, self.x_max + self.x_step, self.x_step), np.arange(self.x_min, self.x_max + self.x_step, self.x_step))
        self.y = np.zeros_like(self.x1)
        for idx, _ in np.ndenumerate(self.x1):
            self.y[idx[0]][idx[1]] = self.de.cec.value([self.x1[idx[0]][idx[1]], self.x2[idx[0]][idx[1]]])

    def draw_function(self):
        plt.contourf(self.x1, self.x2, self.y)
        plt.colorbar()
        plt.xlabel('x1')
        plt.ylabel('x2')

    def draw_population(self):
        plt.scatter(self.de.population[:, 0], self.de.population[:, 1], marker='o', c='red')
    
    def draw(self, generation, best_score):
        plt.clf()
        self.draw_function()
        self.draw_population()
        plt.title(f"Generacja: {generation}, Najlepszy wynik: {best_score:.3f}")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.show()

    def stop(self):
        plt.ioff()

    def evolve(self):
        for i in range(int(self.max_fes / self.de.population_size)):
            if self.de.func_evals + self.de.population_size > self.max_fes:
                break
            self.de.step()
            if self.de.func_evals % self.fes_step == 0:
                self.draw(i, self.de.best_score)
        self.stop()


draw_de = DrawDE(10, 50, 0.8, 0.8, -100, 100, 2, 10000, 50)
draw_de.evolve()