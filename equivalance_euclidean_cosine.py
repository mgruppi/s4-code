import numpy as np
from scipy.spatial.distance import euclidean, cosine
from itertools import combinations

x = np.random.random((100, 100))

for i in range(len(x)):
    x[i] = x[i]/np.linalg.norm(x[i])


total_abs_diff = 0
for u, v in combinations(x, 2):
    e = euclidean(u, v)
    dcos = cosine(u, v)
    eq = np.sqrt(2)*np.sqrt(dcos)
    eq2 = (e**2)/2

    print(e, dcos, eq, eq2)
    total_abs_diff += np.fabs(e-eq)

print("Total diff", total_abs_diff)