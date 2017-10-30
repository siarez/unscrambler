import numpy as np
import matplotlib.pyplot as plt

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

file_path = "./mldata/cifar-10-batches-py/data_batch_1"
data = unpickle(file_path)
images = np.array(data[b'data'])
channels = np.array(np.split(images, 3, 1))
grey = np.mean(channels, 0)
fig = plt.figure()
plt.imshow(grey[30].reshape(32, 32), cmap="Greys")
plt.show()
print("yo!")