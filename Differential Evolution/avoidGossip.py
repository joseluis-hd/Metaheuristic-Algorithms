import numpy as np
import matplotlib.pyplot as plt

publico = np.array([[3, 4],[5, 6],[10, 10],[1, 9],[10, 0]])
nopublico = np.array([[7, 7],[6, 2],[9.5, 9.5]])

def avoidGossip(solucion, publico=publico, nopublico=nopublico):
    if len(solucion.shape) == 3:
        filas, columnas, profundidad = solucion.shape
        y = np.zeros((columnas, profundidad))
        for col in range(columnas):
            for prof in range(profundidad):
                suma = 0
                for person in publico:
                    suma += np.linalg.norm(solucion[:, col, prof] - person)
                for person in nopublico:
                    suma -= 1.2 * np.linalg.norm(solucion[:, col, prof] - person)
                y[col, prof] = suma
        return y

    else:
        suma = 0
        for person in publico:
            suma += np.linalg.norm(solucion - person)
        for person in nopublico:
            suma -= 1.2 * np.linalg.norm(solucion - person)
        return suma

if __name__ == "__main__":
    X0 = np.linspace(0, 12, 200)
    X1 = np.linspace(0, 12, 200)
    X0, X1 = np.meshgrid(X0, X1)
    x = np.stack((X0, X1))
    z = avoidGossip(x)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection="3d")
    ax.plot_wireframe(X0, X1, z, rstride=10, cstride=10, linewidth=0.8)

    for p in publico:
        ax.scatter(p[0], p[1], 10, marker="o", color="b", label="público" if p is publico[0] else "")
    for n in nopublico:
        ax.scatter(n[0], n[1], 10, marker="o", color="r", label="no público" if n is nopublico[0] else "")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Valor objetivo")
    ax.set_title("Superficie avoidGossip")
    plt.legend()
    plt.tight_layout()
    plt.show()
