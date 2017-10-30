import numpy as np
from scipy.spatial.distance import pdist, squareform
#from sklearn.datasets import fetch_mldata
from sklearn import datasets
from fastcluster import linkage

import matplotlib.pyplot as plt

mnist = datasets.fetch_mldata('MNIST original', data_home='./')
images = mnist.data
# pick to random images
indices = np.random.choice(len(images), 5000, replace=False)
picked_images = np.array([images[i] for i in indices]).astype(np.double)/255
test_image = np.array(images[20000]).astype(np.float)/255
image_noise = np.random.randn(*picked_images.shape) / 100
picked_images += image_noise

iris = datasets.load_iris()


def calc_dist(data):
    #return squareform(pdist(data))
    return 1 - np.power(np.abs(np.corrcoef(data, rowvar=True)), 2)
    #return np.abs(np.cov(data, rowvar=True)) * 255


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

iris.data = picked_images.transpose()
print(iris.data.shape)
dist_mat = calc_dist(iris.data)
positions = compute_pos(dist_mat)
variation_in_pos = np.nanmax(positions, 0) - np.nanmin(positions, 0)
std_in_pos = np.nanstd(positions[:, 0:20], 0)
sort_index = np.argsort(std_in_pos)[-2:]

fig = plt.figure()
plt.subplot(211)
plt.imshow(test_image.reshape(28, 28))
plt.subplot(212)
plt.scatter(positions[:, sort_index[0]], positions[:, sort_index[1]], c=test_image)
#plt.scatter(positions[:, 0], positions[:, 1], c=test_image, cmap="Blues")
#plt.xlim([-0.74, -0.69])
plt.show()

N = len(iris.data)
plt.pcolormesh(dist_mat)
plt.colorbar()
plt.xlim([0, N])
plt.ylim([0, N])
#plt.show()


scrambling_order = np.random.permutation(N)
X = iris.data[scrambling_order, :]

dist_mat = calc_dist(X)

plt.pcolormesh(dist_mat)
plt.xlim([0,N])
plt.ylim([0,N])
#plt.show()



def seriation(Z, N, cur_index):
    '''
        input:
            - Z is a hierarchical tree (dendrogram)
            - N is the number of points given to the clustering process
            - cur_index is the position in the tree for the recursive traversal
        output:
            - order implied by the hierarchical tree Z

        seriation computes the order implied by a hierarchical tree (dendrogram)
    '''
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index - N, 0])
        right = int(Z[cur_index - N, 1])
        return (seriation(Z, N, left) + seriation(Z, N, right))


def compute_serial_matrix(dist_mat, method="ward"):
    '''
        input:
            - dist_mat is a distance matrix
            - method = ["ward","single","average","complete"]
        output:
            - seriated_dist is the input dist_mat,
              but with re-ordered rows and columns
              according to the seriation, i.e. the
              order implied by the hierarchical tree
            - res_order is the order implied by
              the hierarhical tree
            - res_linkage is the hierarhical tree (dendrogram)

        compute_serial_matrix transforms a distance matrix into
        a sorted distance matrix according to the order implied
        by the hierarchical tree (dendrogram)
    '''
    N = len(dist_mat)
    flat_dist_mat = squareform(dist_mat, checks=False)
    res_linkage = linkage(flat_dist_mat, method=method, preserve_input=True)
    res_order = seriation(res_linkage, N, N + N - 2)
    seriated_dist = np.zeros((N, N))
    a, b = np.triu_indices(N, k=1)
    seriated_dist[a, b] = dist_mat[[res_order[i] for i in a], [res_order[j] for j in b]]
    seriated_dist[b, a] = seriated_dist[a, b]

    return seriated_dist, res_order, res_linkage


methods = ["ward", "single", "average", "complete"]
for method in methods:
    print("Method:\t", method)

    ordered_dist_mat, res_order, res_linkage = compute_serial_matrix(dist_mat, method)

    plt.pcolormesh(ordered_dist_mat)
    plt.xlim([0, N])
    plt.ylim([0, N])
    #plt.show()


fig = plt.figure()
plt.subplot(311)
plt.imshow(test_image.reshape(28, 28))
scrambled_img = test_image[scrambling_order]
plt.subplot(312)
plt.imshow(scrambled_img.reshape(28, 28))
unscrambled_img = scrambled_img[res_order]
plt.subplot(313)
plt.imshow(unscrambled_img.reshape(28, 28))

plt.show()