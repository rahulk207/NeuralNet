from NeuralNet import MyNeuralNetwork
import mnist
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pickle

def oneHot(y, n_classes):
    y_one_hot = np.zeros((y.shape[0], n_classes))
    y_one_hot[np.arange(y.size),y] = 1
    return y_one_hot

train_images = mnist.train_images()
train_labels = mnist.train_labels()
train_images = train_images.reshape((train_images.shape[0], train_images.shape[1] * train_images.shape[2])).astype('float64')

test_images = mnist.test_images()
test_labels = mnist.test_labels()
test_images = test_images.reshape((test_images.shape[0], test_images.shape[1] * test_images.shape[2])).astype('float64')

# scaler = StandardScaler()
scaler = MinMaxScaler()
train_images = scaler.fit_transform(train_images)
test_images = scaler.transform(test_images)

# m = np.mean(train_images, axis=0, keepdims = True)
# s = np.std(train_images, axis=0, keepdims = True) + 1e-8
#
# train_images -= m
# train_images /= s

#encode y into one-hot
n_classes = 10
train_labels = oneHot(train_labels, 10)
test_labels = oneHot(test_labels, 10)

# train_images, test_images = train_images/255, test_images/255
active_fn = 'tanh'
net = MyNeuralNetwork(5, [ 784, 256, 128, 64, 10], active_fn, 0.1, 'normal', 64, 100)
net.fit(train_images, train_labels, test_images, test_labels)
accuracy = net.score(test_images, test_labels)

f = open(active_fn + " model", "wb")
pickle.dump(net, f)

print(accuracy)
