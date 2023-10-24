import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import pickle
from sklearn.manifold import TSNE, LocallyLinearEmbedding
from scipy.spatial.distance import pdist, squareform
from scipy import signal
# import skvideo.io
from openTSNE import oTSNE

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
# """Get MNIST"""
# mnist = datasets.fetch_mldata('MNIST original', data_home='./')
# """Load video"""
# wgl = skvideo.io.vread("wiggling_1.3gp")
# wgl = np.squeeze(wgl[:, 0:90:2, 15:105:2, 1])
# wgl = np.reshape(wgl, (wgl.shape[0], wgl.shape[1] * wgl.shape[2])).astype(np.float)


#images = mnist.data
images = cifar_grey
#images = wgl
image_dim = int(np.sqrt(len(images[0])))
num_of_images = 1000

# making the random numbers predictable!
np.random.seed(40)

# pick to random images
indices = np.random.choice(len(images), num_of_images, replace=False)
picked_images = np.array([images[i] for i in indices]).astype(np.double)/(255)

def calc_dist(data):

    #dist = 1 - np.power(np.abs(np.corrcoef(data, rowvar=True)), 2)
    #dist = np.power(1 - np.abs(np.corrcoef(data, rowvar=True)), 2)
    dist = 1 - np.abs(np.corrcoef(data, rowvar=True))
    #dist = 1 - np.corrcoef(data, rowvar=True)
    #dist = 1 - np.maximum(np.corrcoef(data, rowvar=True), 0)
    #dist = np.exp(1 - np.abs(np.corrcoef(data, rowvar=True))) - 1
    #dist = np.exp(1 - np.power(np.abs(np.corrcoef(data, rowvar=True)), 2)) - 1

    pixel_var = np.var(data, 1)
    var_i = np.expand_dims(pixel_var, 0)
    var_j = np.expand_dims(pixel_var, 1)
    pixel_cov = np.cov(data, rowvar=True)
    #dist = np.sqrt( np.maximum(var_i + var_j - 2 * pixel_cov, 0))

    # eliminating pixels with nan distance. These are pixels who have not changed across all images. That is std = 0
    eleminated_pixel_idx = np.all(np.isnan(dist), axis=0)
    dist = dist[:, ~np.all(np.isnan(dist), axis=0)]
    dist = dist[~np.all(np.isnan(dist), axis=1), :]
    return dist, eleminated_pixel_idx

def compute_pos(dist, dim=2):
    """
    This function computes the positions of points in space from their distance matrix.
    https://math.stackexchange.com/questions/156161/finding-the-coordinates-of-points-from-distance-matrix
    http://www.galileoco.com/literature/OCRyoungHouseholder38.pdf
    :param dist: is a square distance matrix
    :param dim: dimention of the coord system
    :return: returns points coordinates
    """
    # ToDo: read on eigenvalue constraints and threasholding
    # http: // www.galileoco.com / literature / OCRyoungHouseholder38.pdf
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4303596/
    # https://dl.acm.org/citation.cfm?id=2398462
    # https://www.stat.berkeley.edu/~bickel/BL2008Aos-thresholding.pdf
    # use tSNE to project higher dimensions down to 2. http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html


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
    noise = np.expand_dims(noise_coef, 2) * np.random.randn(n, n, num) / (n**2 * 100)
    return noise

def create_noisy_img(n, num):
    """
    :param n: size
    :param num: number of images
    :return: 2D square array/image
    """
    # creating kernel
    kernel_dim = 9

    # creating a center heavy kernel
    y, x = np.ogrid[-kernel_dim/2:kernel_dim/2, -kernel_dim/2:kernel_dim/2]
    x += 0.5
    y += 0.5
    conv_kernel2 = x * x + y * y
    conv_kernel2 = 1 / (conv_kernel2 + 1)
    conv_kernel2 /= np.sum(conv_kernel2)

    # creating a uniform kernel
    conv_kernel = np.ones((kernel_dim, kernel_dim))/(kernel_dim**2)

    # creating noisy images
    noise = np.random.rand(n, n, num) * 4
    # running the kernel over each noisy image
    convolved_imgs = []
    for i in range(noise.shape[-1]):
        convolved_imgs.append(signal.convolve2d(noise[..., i], conv_kernel2, boundary='symm', mode='same'))
    convolved_imgs = np.array(convolved_imgs)
    convolved_imgs = np.rollaxis(convolved_imgs, 0, 3)
    return convolved_imgs

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
    image = np.arange(0, 1, 1 / (dim ** 2 - 0.5)).astype(np.float32)
    diag_line = np.ones((dim, dim))
    np.fill_diagonal(diag_line, 0)
    image *= diag_line.flat
    image *= np.flipud(np.fliplr(np.tri(dim, k=18))).flat
    image *= create_disk_image(dim/2, dim/2, dim, 5).flat
    return image

picked_images = picked_images.transpose()
# Corner occlusion: sets the corner of all images to zero. Causing the algorithm to cut the out.
picked_images *= np.reshape(np.flipud(np.tri(image_dim, k=18)), (image_dim**2, 1))

# make pixels zero in a checkerboard fashion
checkerboard = np.zeros((image_dim, image_dim),dtype=int)
checkerboard[1::2, ::2] = 1
checkerboard[::2, 1::2] = 1
# picked_images *= np.reshape(checkerboard, (image_dim**2, 1))

# disk occlusion:
# picked_images *= np.reshape(create_disk_image(image_dim/2, image_dim/2, image_dim, 4), (image_dim**2, 1))

# Adding noise
#picked_images += np.reshape(create_disk_noise(image_dim/2, image_dim/2, image_dim, num_of_images), (image_dim**2, -1))
#picked_images += np.reshape(create_noisy_img(image_dim, num_of_images), (image_dim**2, num_of_images))

dist_mat, bad_pixel_idx = calc_dist(picked_images)
positions = np.real(compute_pos(dist_mat))
positions = positions[:, ~np.all(np.isnan(positions), axis=0)]
std_in_pos = np.nanstd(positions[:, 0:100], 0)
sort_index = np.argsort(std_in_pos)[-20:]
positions_embd = TSNE(n_components=2, verbose=True, metric="precomputed", perplexity=24.0, n_iter=300, init="random").fit_transform(dist_mat)
embedding_dist = squareform(pdist(positions_embd))

#positions_embd = LocallyLinearEmbedding(n_neighbors=120, n_components=2, method="modified").fit_transform(positions[:, sort_index])

#plt.hist(dist_mat.flat, 51)
#plt.show()

# create test image
test_image = create_test_image(image_dim)

def plot_3d():
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax3D = fig.add_subplot(111, projection='3d')
    ax3D.scatter(positions_embd[:, 0], positions_embd[:, 1], positions_embd[:, 2], s=8, c=test_image[~bad_pixel_idx])
    plt.show()

# plot_3d()


# create scrambling scheme and scrambling the data

scrambling_order = np.random.permutation(image_dim**2)
picked_images_scrm = picked_images[scrambling_order, :]
test_image_scrm = test_image[scrambling_order]





dist_mat_scrm, bad_pixel_idx_scrm = calc_dist(picked_images_scrm)
positions_scrm = np.real(compute_pos(dist_mat_scrm))
positions_scrm = positions_scrm[:, ~np.all(np.isnan(positions_scrm), axis=0)]
std_in_pos_scrm = np.nanstd(positions_scrm[:, 0:100], 0)
sort_index_scrm = np.argsort(std_in_pos_scrm)[-20:]
#positions_scrm_embd = TSNE(n_components=2, verbose=True).fit_transform(positions_scrm[:, sort_index_scrm])
positions_scrm_embd = TSNE(n_components=2, verbose=True, metric="precomputed", perplexity=24.0, n_iter=300, init="random").fit_transform(dist_mat_scrm)



f, ((ax1, ax2, ax_tnse1), (ax3, ax4, ax_tnse2)) = plt.subplots(2, 3, sharey=False)
ax1.imshow(test_image.reshape(image_dim, image_dim))
ax2.scatter(positions[:, sort_index[-1:]], positions[:, sort_index[-2:-1]], s=8, c=test_image[~bad_pixel_idx])
ax_tnse1.scatter(positions_embd[:, 0], positions_embd[:, 1], s=8, c=test_image[~bad_pixel_idx])
ax3.imshow(test_image_scrm.reshape(image_dim, image_dim))
ax4.scatter(positions_scrm[:, sort_index_scrm[-1:]], positions_scrm[:, sort_index_scrm[-2:-1]], s=8, c=test_image_scrm[~bad_pixel_idx_scrm])
ax_tnse2.scatter(positions_scrm_embd[:, 0], positions_scrm_embd[:, 1], s=8, c=test_image_scrm[~bad_pixel_idx_scrm])
plt.show()
