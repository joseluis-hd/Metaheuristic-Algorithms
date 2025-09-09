"""
José Luis Haro Díaz
Newton Method
Metaheuristic Algorithms
"""


import numpy as np
import random as rd
import matplotlib.pyplot as plt

# ---------- Semillas reproducibles ----------
rd.seed(1)
np.random.seed(1)

# ---------- Parámetros ES ----------
G = 100         # generaciones
D = 2           # dimensión
MU = 20         # μ padres
LAMBDA = 40     # λ hijos por generación

# ---------- Funciones ----------
def f1(x):  # dominio: [-2,2]^2
    return x[0] * np.exp(-(x[0]**2 + x[1]**2))

def f2(x):  # dominio: [-5.12,5.12]^2
    return x[0]**2 + x[1]**2

# ---------- Operadores ----------
def recomb_discreta(v1, v2):
    mask = np.random.rand(*v1.shape) < 0.5
    return np.where(mask, v1, v2)

def clip_bounds(x, lower, upper):
    return np.minimum(np.maximum(x, lower), upper)

def es_mu_lambda_discreta(func, lower, upper, g=G, mu=MU, lam=LAMBDA):
    # Inicializar μ padres con solución y sigmas > 0
    pobl = []
    for _ in range(mu):
        sol = lower + (upper - lower) * np.random.rand(D)
        sigma = np.random.rand(D) * 0.5 + 0.05
        pobl.append({'sol': sol, 'sigma': sigma, 'fit': func(sol)})

    trayectoria = []  # guarda el mejor por generación (para graficar)
    for _ in range(g):
        hijos = []
        for _ in range(lam):
            i = np.random.randint(0, mu)
            j = i
            while j == i:
                j = np.random.randint(0, mu)
            # recombinación discreta en solución y sigma
            y  = recomb_discreta(pobl[i]['sol'],   pobl[j]['sol'])
            sg = recomb_discreta(pobl[i]['sigma'], pobl[j]['sigma'])
            # mutación gaussiana y recorte a límites
            y = clip_bounds(y + np.random.normal(0.0, sg, D), lower, upper)
            hijos.append({'sol': y, 'sigma': sg, 'fit': func(y)})

        # (μ,λ): padres NO sobreviven
        hijos.sort(key=lambda d: d['fit'])  # minimización
        pobl = hijos[:mu]
        trayectoria.append(pobl[0]['sol'].copy())

    best = pobl[0]
    return best, np.array(trayectoria)

def grafica_funcion(func, lower, upper, titulo, best, trayectoria):
    # Malla para la superficie
    x0 = np.linspace(lower[0], upper[0], 120)
    x1 = np.linspace(lower[1], upper[1], 120)
    X0, X1 = np.meshgrid(x0, x1)
    grid = np.stack((X0, X1))
    Z = func(grid)

    plt.ion()
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # wireframe de la función
    ax.plot_wireframe(X0, X1, Z, rstride=6, cstride=6, cmap='viridis')
    # trayectoria del mejor (en 2D, levantamos con f)
    if len(trayectoria) > 1:
        trajZ = np.array([func(p) for p in trayectoria])
        ax.plot(trayectoria[:,0], trayectoria[:,1], trajZ, linewidth=2)
    # mejor solución final
    ax.scatter(best['sol'][0], best['sol'][1], best['fit'], facecolor='red', s=60)
    # ejes/estilo
    ax.set_xlim3d(lower[0], upper[0]); ax.set_xlabel('x0')
    ax.set_ylim3d(upper[1], lower[1]); ax.set_ylabel('x1')  # igual que los ejemplos del profe
    ax.set_zlabel('f(x)')
    ax.set_title(titulo)
    fig.canvas.draw(); fig.canvas.flush_events()
    return fig, ax

if __name__ == "__main__":
    #-------- f1 --------
    lower1 = np.array([-2.0, -2.0])
    upper1 = np.array([ 2.0,  2.0])
    best1, traj1 = es_mu_lambda_discreta(f1, lower1, upper1)
    grafica_funcion(
        f1, lower1, upper1,
        "Práctica 5 — f1(x)=x0·e^(-(x0^2+x1^2))  (μ,λ)-ES + recomb. discreta\n"
        f"Mejor: x≈{best1['sol']}, f≈{best1['fit']:.6f}",
        best1, traj1
    )

    #-------- f2 --------
    lower2 = np.array([-5.12, -5.12])
    upper2 = np.array([ 5.12,  5.12])
    best2, traj2 = es_mu_lambda_discreta(f2, lower2, upper2)
    grafica_funcion(
        f2, lower2, upper2,
        "Práctica 5 — f2(x)=x0^2+x1^2  (μ,λ)-ES + recomb. discreta\n"
        f"Mejor: x≈{best2['sol']}, f≈{best2['fit']:.6f}",
        best2, traj2
    )

    #Mostrar ventanas
    plt.ioff()
    plt.show()
    