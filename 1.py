# practica5_es_discreta.py
# Práctica 5 — Estrategias Evolutivas (μ,λ) con recombinación sexual discreta
# Genera dos gráficas 3D (una por función) y marca el mejor punto encontrado.

import numpy as np
import random as rd
import matplotlib.pyplot as plt

# -------------------- Configuración --------------------
rd.seed(2)
np.random.seed(2)

G = 120        # generaciones
D = 2          # dimensión
MU = 30        # μ padres
LAMBDA = 90    # λ hijos por generación

# -------------------- Funciones objetivo --------------------
# f1(x) = x0 * e^{-(x0^2 + x1^2)} ; dominio [-2, 2]^2
def f1(x: np.ndarray) -> float:
    return x[0] * np.exp(-(x[0]**2 + x[1]**2))

# f2(x) = x0^2 + x1^2 ; dominio [-5.12, 5.12]^2
def f2(x: np.ndarray) -> float:
    return x[0]**2 + x[1]**2

# -------------------- Operadores ES --------------------
# Recombinación sexual discreta por-gen (también aplicada a σ)
def recomb_discrete(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    mask = np.random.rand(*v1.shape) < 0.5
    return np.where(mask, v1, v2)

def clip_bounds(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, lower), upper)

# (μ,λ)-ES con recombinación discreta
def es_mu_lambda_discrete(func, lower, upper, g=G, mu=MU, lam=LAMBDA):
    # Inicializar μ padres con solución y σ (vector de desviaciones > 0)
    pop = []
    for _ in range(mu):
        sol = lower + (upper - lower) * np.random.rand(D)
        sigma = np.random.rand(D) * 0.5 + 0.05
        pop.append({"sol": sol, "sigma": sigma, "fit": func(sol)})

    for _ in range(g):
        kids = []
        for _ in range(lam):
            # Selección aleatoria de dos padres distintos
            i = np.random.randint(0, mu)
            j = i
            while j == i:
                j = np.random.randint(0, mu)

            # Recombinación discreta solución y σ
            child_sol = recomb_discrete(pop[i]["sol"],   pop[j]["sol"])
            child_sig = recomb_discrete(pop[i]["sigma"], pop[j]["sigma"])

            # Mutación gaussiana por-gen con desviaciones child_sig
            r = np.random.normal(0.0, child_sig, D)
            child_sol = clip_bounds(child_sol + r, lower, upper)

            kids.append({"sol": child_sol, "sigma": child_sig, "fit": func(child_sol)})

        # (μ,λ): solo los hijos compiten, los padres no sobreviven
        kids.sort(key=lambda d: d["fit"])  # minimización
        pop = kids[:mu]

    return pop[0]  # mejor individuo

# -------------------- Utilidad para graficar --------------------
def make_surface_plot(func, lower, upper, best, title):
    x0 = np.linspace(lower[0], upper[0], 120)
    x1 = np.linspace(lower[1], upper[1], 120)
    X0, X1 = np.meshgrid(x0, x1)
    grid = np.stack((X0, X1))
    Z = func(grid)

    fig = plt.figure(figsize=(7.2, 5.6))
    ax = plt.axes(projection="3d")
    ax.plot_wireframe(X0, X1, Z, rstride=6, cstride=6)
    ax.scatter(best["sol"][0], best["sol"][1], best["fit"], s=60)
    ax.set_title(title)
    ax.set_xlabel("x0"); ax.set_ylabel("x1"); ax.set_zlabel("f(x)")
    plt.tight_layout()
    return fig

# -------------------- Ejecución: f1 y f2 --------------------
if __name__ == "__main__":
    # f1
    lower1 = np.array([-2.0, -2.0])
    upper1 = np.array([ 2.0,  2.0])
    best1 = es_mu_lambda_discrete(f1, lower1, upper1)
    fig1 = make_surface_plot(
        f1, lower1, upper1, best1,
        "Práctica 5 — f1(x)=x0·e^(-(x0^2+x1^2))  (μ,λ)-ES + recomb. discreta\n"
        f"Mejor: x≈{best1['sol']}, f≈{best1['fit']:.6f}"
    )

    # f2
    lower2 = np.array([-5.12, -5.12])
    upper2 = np.array([ 5.12,  5.12])
    best2 = es_mu_lambda_discrete(f2, lower2, upper2)
    fig2 = make_surface_plot(
        f2, lower2, upper2, best2,
        "Práctica 5 — f2(x)=x0^2+x1^2  (μ,λ)-ES + recomb. discreta\n"
        f"Mejor: x≈{best2['sol']}, f≈{best2['fit']:.6f}"
    )

    # Mostrar ambas ventanas
    plt.show()

    # (Opcional) Guardar archivos si quieres entregar las imágenes:
    # fig1.savefig("practica5_f1.png", dpi=160)
    # fig2.savefig("practica5_f2.png", dpi=160)
