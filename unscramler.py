import numpy as np
from scipy.spatial.distance import pdist, squareform
#from sklearn.datasets import fetch_mldata
from sklearn import datasets
from fastcluster import linkage
import matplotlib.pyplot as plt
import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

"""Get CIFAR-10"""
file_path = "./mldata/cifar-10-batches-py/data_batch_1"
data = unpickle(file_path)
images = np.array(data[b'data'])
channels = np.array(np.split(images, 3, 1))
cifar_grey = np.mean(channels, 0)
"""Get MNIST"""
mnist = datasets.fetch_mldata('MNIST original', data_home='./')

#images = mnist.data
images = cifar_grey
image_dim = int(np.sqrt(len(images[0])))

# pick to random images
indices = np.random.choice(len(images), 5000, replace=False)
picked_images = np.array([images[i] for i in indices]).astype(np.double)/255

def calc_dist(data):
    return 1 - np.power(np.abs(np.corrcoef(data, rowvar=True)), 2)
    #return np.power(1 - np.abs(np.corrcoef(data, rowvar=True)), 2)
    #return 1 - np.abs(np.corrcoef(data, rowvar=True))
    #return 1 - np.maximum(np.corrcoef(data, rowvar=True), 0)
    #return 1 - np.exp(1 - np.abs(np.corrcoef(data, rowvar=True)))


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

picked_images = picked_images.transpose()
dist_mat = calc_dist(picked_images)
positions = np.real(compute_pos(dist_mat))
std_in_pos = np.nanstd(positions[:, 0:4], 0)
sort_index = np.argsort(std_in_pos)[-2:]


"""
#
N = len(iris.data)
plt.pcolormesh(dist_mat)
plt.colorbar()
plt.xlim([0, N])
plt.ylim([0, N])
#plt.show()
"""

test_image = np.arange(0, 255, 255/image_dim**2).astype(np.float)/255
diag_line = np.ones((image_dim, image_dim))
np.fill_diagonal(diag_line, 0)
test_image *= diag_line.flat
scrambling_order = np.random.permutation(image_dim**2)
picked_images_scrambled = picked_images[scrambling_order, :]
test_image_scrambled = test_image[scrambling_order]

dist_mat_scrambled = calc_dist(picked_images_scrambled)
positions_scrambled = np.real(compute_pos(dist_mat_scrambled))
std_in_pos_scrambled = np.nanstd(positions[:, 0:4], 0)
sort_index_scrambled = np.argsort(std_in_pos)[-2:]

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey=False)
ax1.imshow(test_image.reshape(image_dim, image_dim))
ax2.scatter(positions[:, sort_index[0]], positions[:, sort_index[1]], s=8, c=test_image)
ax3.imshow(test_image_scrambled.reshape(image_dim, image_dim))
ax4.scatter(positions_scrambled[:, sort_index[0]], positions[:, sort_index[1]], s=8, c=test_image_scrambled)
plt.show()
