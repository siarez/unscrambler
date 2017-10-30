import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt



def compute_pos(dist, dim=2):
    """
    This function computes the positions of points in space from their distance matrix.
    :param dist: is a square distance matrix
    :param dim: dimention of the coord system
    :return: returns points coordinates
    """
    d1j2 = np.expand_dims(np.square(dist[0, :]), 0)
    di12 = np.expand_dims(np.square(dist[:, 0]), 1)
    dij2 = np.square(dist)
    M = (d1j2 + di12 - dij2)/2
    S, U = np.linalg.eig(M)
    return U * np.sqrt(S)



"""Test position calculator"""
points = np.arange(0, 20).reshape((10, 2))
points += np.random.randint(0, 5, (10, 2))
points = np.array([[0, 0], [0, 20], [10, 0], [10, 20], [20, 0], [20, 20]])
#points = np.random.randint(0, 20, (10, 2))
test_dists = squareform(pdist(points))
positions = np.real(compute_pos(test_dists))
#remove nan columns
positions = positions[:,~np.all(np.isnan(positions), axis=0)]
std_in_pos = np.nanstd(positions, 0)
sort_index = np.argsort(std_in_pos)[-2:]

fig = plt.figure()
plt.subplot(211)
plt.scatter(positions[:, sort_index[0]], positions[:, sort_index[1]])
#plt.xlim(-20, 20)
#plt.ylim(-20, 20)
plt.subplot(212)
plt.scatter(points[:, 0], points[:, 1])
#plt.xlim(-20, 20)
#plt.ylim(-20, 20)
plt.show()


print("done")