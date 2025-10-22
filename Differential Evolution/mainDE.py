import DE as de
import avoidGossip as bn
import numpy as np

lower = [10, 10]
upper = [10, 10]
breakCriteria = 100
algoritmo = de.algorithmDE(dimention=2,
                           lower=lower,
                           upper=upper,
                           function=bn.avoidGossip,
                           populationSize=30,
                           maxIter=100,
                           breakCriteria=breakCriteria,
                           F=1,
                           C=1,
                           seed=9001,)

solution, fitness, = algoritmo.run()
print("Mejor solución encontrada:")
print(solution)
print(fitness)