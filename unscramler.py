import numpy as np
from sklearn import datasets
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
num_of_images = 5000

# pick to random images
indices = np.random.choice(len(images), num_of_images, replace=False)
picked_images = np.array([images[i] for i in indices]).astype(np.double)/255

def calc_dist(data):
    #return 1 - np.power(np.abs(np.corrcoef(data, rowvar=True)), 2)
    #return np.power(1 - np.abs(np.corrcoef(data, rowvar=True)), 2)
    return 1 - np.abs(np.corrcoef(data, rowvar=True))
    #return 1 - np.corrcoef(data, rowvar=True)
    #return 1 - np.maximum(np.corrcoef(data, rowvar=True), 0)
    #return np.exp(1 - np.abs(np.corrcoef(data, rowvar=True))) - 1
    #return np.exp(1 - np.power(np.abs(np.corrcoef(data, rowvar=True)), 2)) - 1

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

def create_disk_noise(a, b, n, num):
    """
    :param a: position
    :param b: position
    :param n: size
    :param num: number of images
    :return: 2D square array/image
    """
    y, x = np.ogrid[-a:n - a, -b:n - b]
    noise_coef = x * x + y * y
    noise = np.expand_dims(noise_coef, 2) * np.random.randn(n, n, num) / (n**2 / 2)
    return noise

def create_disk_image(a, b, n, r):
    """
    :param a: position
    :param b: position
    :param n: size
    :param r: radius
    :return: 2D square array/image
    """
    y, x = np.ogrid[-a:n - a, -b:n - b]
    mask = x * x + y * y <= r * r
    array = np.ones((n, n))
    array[mask] = 0
    return array

def create_test_image(dim):
    """
    Creates a square image that is useful for visual inspection of unscrambler performance.

    :param dim: dimension of image
    :return: 2D array of size (dim dim)
    """
    image = np.arange(0, 1, 1 / dim ** 2).astype(np.float)
    diag_line = np.ones((dim, dim))
    np.fill_diagonal(diag_line, 0)
    image *= diag_line.flat
    image *= np.flipud(np.fliplr(np.tri(dim, k=18))).flat
    image *= create_disk_image(16, 16, dim, 5).flat
    return image

picked_images = picked_images.transpose()
#picked_images += np.reshape(create_disk_noise(16, 16, 32, num_of_images), (image_dim**2, -1))

dist_mat = calc_dist(picked_images)
positions = np.real(compute_pos(dist_mat))
positions = positions[:,~np.all(np.isnan(positions), axis=0)]
std_in_pos = np.nanstd(positions[:, 0:100], 0)
sort_index = np.argsort(std_in_pos)[-2:]

plt.hist(dist_mat.flat, 51)
plt.show()

# create test image
test_image = create_test_image(image_dim)

# create scrambling scheme and scrambling the data
scrambling_order = np.random.permutation(image_dim**2)
picked_images_scrm = picked_images[scrambling_order, :]
test_image_scrm = test_image[scrambling_order]

dist_mat_scrm = calc_dist(picked_images_scrm)
positions_scrm = np.real(compute_pos(dist_mat_scrm))
positions_scrm = positions_scrm[:,~np.all(np.isnan(positions_scrm), axis=0)]
std_in_pos_scrm = np.nanstd(positions_scrm[:, 0:100], 0)
sort_index_scrm = np.argsort(std_in_pos_scrm)[-2:]

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey=False)
ax1.imshow(test_image.reshape(image_dim, image_dim))
ax2.scatter(positions[:, sort_index[0]], positions[:, sort_index[1]], s=8, c=test_image)
ax3.imshow(test_image_scrm.reshape(image_dim, image_dim))
ax4.scatter(positions_scrm[:, sort_index_scrm[0]], positions_scrm[:, sort_index_scrm[1]], s=8, c=test_image_scrm)
plt.show()


"""
#
N = len(iris.data)
plt.pcolormesh(dist_mat)
plt.colorbar()
plt.xlim([0, N])
plt.ylim([0, N])
#plt.show()
"""
