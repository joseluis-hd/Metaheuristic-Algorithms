"""
José Luis Haro Díaz
Genetic Algorithm 
Metaheuristic Algorithms
"""

import numpy as np
import random as rd
import copy
import matplotlib.pyplot as plt

# --------- Semillas ---------
rd.seed(1)
np.random.seed(1)

# --------- Hiperparámetros GA ---------
D = 2          
N = 40         
G = 100        
pm = 0.4       

# --------- Funciones ---------
def f1(x):  # dominio: [-2,2]^2
    return x[0] * np.exp(-(x[0]**2 + x[1]**2))

def f2(x):  # dominio: [-5.12,5.12]^2
    return x[0]**2 + x[1]**2

def clip_bounds(x, lower, upper):
    return np.minimum(np.maximum(x, lower), upper)

class Individuo:
    def __init__(self, lower, upper):
        self.sol = lower + (upper - lower) * np.random.rand(D)
        self.fx = None
        self.fit = None

    def eval(self, func):
        self.fx = func(self.sol)
        self.fit = (1.0 / (1.0 + self.fx)) if self.fx >= 0 else (1.0 + abs(self.fx))

def ruleta(pobl):
    total = sum(p.fit for p in pobl)
    r, acc = rd.random(), 0.0
    for p in pobl:
        acc += p.fit / total
        if acc >= r:
            return copy.deepcopy(p)
    return copy.deepcopy(pobl[-1])

def crossover_1p(p1, p2):
    pc = rd.randint(1, D-1) if D > 1 else 1
    c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
    c1.sol[pc:], c2.sol[pc:] = p2.sol[pc:].copy(), p1.sol[pc:].copy()
    return c1, c2

def mutar_random_reset(ind, lower, upper, pm):
    for j in range(D):
        if rd.random() < pm:
            ind.sol[j] = lower[j] + (upper[j] - lower[j]) * rd.random()

# --------- GA principal (minimiza func) ---------
def run_ga(func, lower, upper, gens=G, pop_size=N, p_mut=pm):
    # inicialización
    pobl = [Individuo(lower, upper) for _ in range(pop_size)]
    best = None

    for _ in range(gens):
        #evaluar y actualizar mejor
        for ind in pobl:
            ind.sol = clip_bounds(ind.sol, lower, upper)
            ind.eval(func)
        best_gen = min(pobl, key=lambda z: z.fx)
        if best is None or best_gen.fx < best.fx:
            best = copy.deepcopy(best_gen)

        #nueva población (selección + cruza + mutación)
        nuevos = []
        while len(nuevos) < pop_size:
            p1 = ruleta(pobl)
            p2 = ruleta(pobl)
            while np.allclose(p1.sol, p2.sol):
                p2 = ruleta(pobl)
            h1, h2 = crossover_1p(p1, p2)
            mutar_random_reset(h1, lower, upper, p_mut)
            mutar_random_reset(h2, lower, upper, p_mut)
            h1.sol = clip_bounds(h1.sol, lower, upper)
            h2.sol = clip_bounds(h2.sol, lower, upper)
            nuevos.extend([h1, h2])
        pobl = nuevos[:pop_size]

    #evaluación final y retorno
    for ind in pobl:
        ind.eval(func)
    best_final = min(pobl, key=lambda z: z.fx)
    if best_final.fx < best.fx:
        best = best_final

    best.pobl_final = pobl
    return best

# --------- Gráfica 3D  ---------
def grafica_superficie(func, lower, upper, best, title=None):
    x0 = np.linspace(lower[0], upper[0], 120)
    x1 = np.linspace(lower[1], upper[1], 120)
    X0, X1 = np.meshgrid(x0, x1)
    Z = func(np.stack((X0, X1)))

    fig = plt.figure(figsize=(7.4, 5.4))
    ax = plt.axes(projection='3d')
    ax.plot_wireframe(X0, X1, Z, rstride=6, cstride=6)

    P  = np.array([ind.sol for ind in best.pobl_final])
    Zp = np.array([func(p) for p in P])
    ax.scatter(P[:,0], P[:,1], Zp, s=28, c='g', alpha=0.9)

    #mejor en rojo
    ax.scatter(best.sol[0], best.sol[1], best.fx, s=60, c='r')

    ax.set_xlabel("x0"); ax.set_ylabel("x1"); ax.set_zlabel("f(x)")
    if title: ax.set_title(title)
    plt.tight_layout()
    return fig

# --------- Ejecución para f1 y f2 ---------
if __name__ == "__main__":
    #f1
    lower1 = np.array([-2.0, -2.0]); upper1 = np.array([2.0, 2.0])
    best1 = run_ga(f1, lower1, upper1)
    print(f"f1 -> best x: {best1.sol}, f: {best1.fx:.6f}")
    grafica_superficie(f1, lower1, upper1, best1, title=None)

    #f2
    lower2 = np.array([-5.12, -5.12]); upper2 = np.array([5.12, 5.12])
    best2 = run_ga(f2, lower2, upper2)
    print(f"f2 -> best x: {best2.sol}, f: {best2.fx:.6f}")
    grafica_superficie(f2, lower2, upper2, best2, title=None)

    plt.ioff()
    plt.show()
