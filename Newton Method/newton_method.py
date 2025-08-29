"""
José Luis Haro Díaz
Newton Method
Metaheuristic Algorithms
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

#EPS
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"]  = 42

#Newton method to find roots of f'(x) = 0 using f''(x)
def newton_fprime_root(fp, fpp, x0, tol = 1e-10, maxiter = 100) -> None:
    x = float(x0)
    for _ in range(maxiter):
        g, h = fp(x), fpp(x)
        if not np.isfinite(h) or abs(h) < 1e-12:
            return None
        x_new = x - g/h
        if abs(x_new - x) < tol:
            x = x_new
            break
        x = x_new
    return x if abs(fp(x)) < 1e-6 else None

# Remove duplicates and sort
def unique_sorted(vals, tol = 1e-6) -> list:
    out = []
    for v in vals:
        if v is None or not np.isfinite(v): continue
        if not out or abs(v - out[-1]) >= tol:
            out.append(v)
    return out

#Find all roots of f'(x) = 0 in [a,b] using Newton method with n_points initial points
def newton_roots(fp, fpp, a, b, n_points = 601) -> list: #601 is a safe choice for 600 intervals
    points = np.linspace(a, b, n_points)
    roots = [newton_fprime_root(fp, fpp, s) for s in points]
    roots = sorted(r for r in roots if r is not None and a-1e-6 <= r <= b+1e-6)
    return unique_sorted(roots, tol=1e-6)

#Plotting function
def plot_case(funcion, funcionPrima, funcionPrima2, a, b, title, legend_labels, basename) -> None:
    x = np.linspace(a, b, 2000)

    plt.figure(figsize=(8,6))
    plt.plot(x, funcion(x),       'm',  label=legend_labels[0]) #Magenta
    plt.plot(x, funcionPrima(x),  'g-', label=legend_labels[1]) #Green
    plt.plot(x, funcionPrima2(x), 'b--',label=legend_labels[2]) #Blue
    plt.grid(True)
    roots = newton_roots(funcionPrima, funcionPrima2, a, b) #Red dots for roots
    if roots:
        plt.scatter(roots, np.zeros(len(roots)), s=80, c='r', label="Raices")

    plt.title(title)
    plt.legend(loc="lower center", ncol=2, framealpha=1.0)
    plt.tight_layout()

    out = Path("results"); out.mkdir(exist_ok = True)
    plt.savefig(out / f"{basename}.png", dpi=150, bbox_inches = "tight", transparent = False)
    plt.savefig(out / f"{basename}.eps",            bbox_inches = "tight", transparent = False)
    plt.show()

#=== FUNCTIONS AND DERIVATIVES ===#

#Case1: f(x) = sin(2x)
def func_1(x) -> float:
    y = np.sin(2*x)
    return y
def func_prima_1(x) -> float:
    y = 2*np.cos(2*x)
    return y
def func_prima_p_1(x) -> float:
    y = -4*np.sin(2*x)
    return y

#Case2: f(x) = sin(x) + x cos(x)
def func_2(x) -> float:
    y = np.sin(x) + x*np.cos(x)
    return y
def func_prima_2(x) -> float:
    y = 2*np.cos(x) - x*np.sin(x)
    return y
def func_prima_p_2(x) -> float:
    y = -3*np.sin(x) - x*np.cos(x)
    return y

if __name__ == "__main__":
    #Plot for 1
    plot_case(funcion = func_1, funcionPrima = func_prima_1, funcionPrima2 = func_prima_p_1, a = -4, b = 4, title = "f(x) = sin(2x) , x ∈ [-4,4]",
        legend_labels = ["f(x) = sin(2x)", "f'(x) = 2cos(2x)", "f''(x) = -4sin(2x)"], basename = "newton_method_1")

    #Plot for 2
    plot_case(funcion = func_2, funcionPrima = func_prima_2, funcionPrima2 = func_prima_p_2, a = -5, b = 5, title = "f(x) = sin(x) + x cos(x) , x ∈ [-5,5]",
        legend_labels = ["f(x) = sin(x) + x cos(x)", "f'(x) = 2 cos(x) - x sin(x)", "f''(x) = -3 sin(x) - x cos(x)"], basename = "newton_method_2")
