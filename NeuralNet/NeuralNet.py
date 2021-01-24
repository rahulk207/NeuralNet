import numpy as np
import matplotlib.pyplot as plt

class MyNeuralNetwork():
	"""
	My implementation of a Neural Network Classifier.
	"""

	def __init__(self, n_layers, layer_sizes, activation, learning_rate, weight_init, batch_size, num_epochs):
		"""
		Initializing a new MyNeuralNetwork object

		Parameters
		----------
		n_layers : int value specifying the number of layers

		layer_sizes : integer array of size n_layers specifying the number of nodes in each layer

		activation : string specifying the activation function to be used
					 possible inputs: relu, sigmoid, linear, tanh

		learning_rate : float value specifying the learning rate to be used

		weight_init : string specifying the weight initialization function to be used
					  possible inputs: zero, random, normal

		batch_size : int value specifying the batch size to be used

		num_epochs : int value specifying the number of epochs to be used
		"""


		self.acti_fns = {'relu':self.relu, 'sigmoid':self.sigmoid, 'linear':self.linear, 'tanh':self.tanh, 'softmax':self.softmax}
		self.weight_inits = {'zero':self.zero_init, 'random':self.random_init, 'normal':self.normal_init}
		self.acti_fns_grad = {'relu':self.relu_grad, 'sigmoid':self.sigmoid_grad, 'linear':self.linear_grad, 'tanh':self.tanh_grad, 'softmax':self.softmax_grad}

		if activation not in self.acti_fns:
			raise Exception('Incorrect Activation Function')

		if weight_init not in self.weight_inits:
			raise Exception('Incorrect Weight Initialization Function')

		self.n_layers = n_layers
		self.layer_sizes = layer_sizes
		self.activation = activation
		self.learning_rate = learning_rate
		self.weight_init = weight_init
		self.batch_size = batch_size
		self.num_epochs = num_epochs

		self.weights = list(); self.biases = list()
		for i in range(self.n_layers-1):
			self.weights.append(self.weight_inits[self.weight_init]((self.layer_sizes[i], self.layer_sizes[i+1])))
			self.biases.append(np.array([0]*self.layer_sizes[i+1], dtype='float64'))

	def relu(self, X):
		"""
		Calculating the ReLU activation for a particular layer

		Parameters
		----------
		X : 1-dimentional numpy array

		Returns
		-------
		x_calc : 1-dimensional numpy array after calculating the necessary function over X
		"""
		return np.maximum(0, X)

	def relu_grad(self, X):
		"""
		Calculating the gradient of ReLU activation for a particular layer

		Parameters
		----------
		X : 1-dimentional numpy array

		Returns
		-------
		x_calc : 1-dimensional numpy array after calculating the necessary function over X
		"""
		return (X > 0) * 1

	def sigmoid(self, X):
		"""
		Calculating the Sigmoid activation for a particular layer

		Parameters
		----------
		X : 1-dimentional numpy array

		Returns
		-------
		x_calc : 1-dimensional numpy array after calculating the necessary function over X
		"""
		return 1/(1+np.exp(-X))

	def sigmoid_grad(self, X):
		"""
		Calculating the gradient of Sigmoid activation for a particular layer

		Parameters
		----------
		X : 1-dimentional numpy array

		Returns
		-------
		x_calc : 1-dimensional numpy array after calculating the necessary function over X
		"""
		return self.sigmoid(X) * (1 - self.sigmoid(X))

	def linear(self, X):
		"""
		Calculating the Linear activation for a particular layer

		Parameters
		----------
		X : 1-dimentional numpy array

		Returns
		-------
		x_calc : 1-dimensional numpy array after calculating the necessary function over X
		"""
		#see if need to multiply with some constant
		return X

	def linear_grad(self, X):
		"""
		Calculating the gradient of Linear activation for a particular layer

		Parameters
		----------
		X : 1-dimentional numpy array

		Returns
		-------
		x_calc : 1-dimensional numpy array after calculating the necessary function over X
		"""
		return 1

	def tanh(self, X):
		"""
		Calculating the Tanh activation for a particular layer

		Parameters
		----------
		X : 1-dimentional numpy array

		Returns
		-------
		x_calc : 1-dimensional numpy array after calculating the necessary function over X
		"""
		return np.tanh(X)

	def tanh_grad(self, X):
		"""
		Calculating the gradient of Tanh activation for a particular layer

		Parameters
		----------
		X : 1-dimentional numpy array

		Returns
		-------
		x_calc : 1-dimensional numpy array after calculating the necessary function over X
		"""
		return 1 - np.square(self.tanh(X))

	def softmax(self, X):
		"""
		Calculating the ReLU activation for a particular layer

		Parameters
		----------
		X : 1-dimentional numpy array

		Returns
		-------
		x_calc : 1-dimensional numpy array after calculating the necessary function over X
		"""
		#add trick to avoid overflow
		e = np.exp(X - np.max(X, axis=1, keepdims = True))
		return e/np.sum(e, axis = 1, keepdims = True)

	def softmax_grad(self, X):
		"""
		Calculating the gradient of Softmax activation for a particular layer

		Parameters
		----------
		X : 1-dimentional numpy array

		Returns
		-------
		x_calc : 1-dimensional numpy array after calculating the necessary function over X
		"""
		return self.softmax(X) * (1 - self.softmax(X))

	def zero_init(self, shape):
		"""
		Calculating the initial weights after Zero Activation for a particular layer

		Parameters
		----------
		shape : tuple specifying the shape of the layer for which weights have to be generated

		Returns
		-------
		weight : 2-dimensional numpy array which contains the initial weights for the requested layer
		"""
		return np.zeros(shape)

	def random_init(self, shape):
		"""
		Calculating the initial weights after Random Activation for a particular layer

		Parameters
		----------
		shape : tuple specifying the shape of the layer for which weights have to be generated

		Returns
		-------
		weight : 2-dimensional numpy array which contains the initial weights for the requested layer
		"""
		np.random.seed(1234)
		return 0.01*np.random.rand(shape[0], shape[1])

	def normal_init(self, shape):
		"""
		Calculating the initial weights after Normal(0,1) Activation for a particular layer

		Parameters
		----------
		shape : tuple specifying the shape of the layer for which weights have to be generated

		Returns
		-------
		weight : 2-dimensional numpy array which contains the initial weights for the requested layer
		"""
		np.random.seed(1234)
		return 0.01*np.random.randn(shape[0], shape[1])

	def crossEntropyLoss(self, y, y_prob):

		return -np.mean(y * np.log(y_prob + 1e-8))

	def forward(self, X):

		self.temp_values = list()
		self.temp_values.append([np.array(X), np.array(X)])
		X_temp_acti = np.array(X)
		for i in range(self.n_layers-2):
			# print(self.biases[i])
			X_temp = np.matmul(X_temp_acti, self.weights[i]) + self.biases[i]
			# print(X_temp)
			self.temp_values.append([np.array(X_temp)])
			X_temp_acti = (self.acti_fns[self.activation](X_temp))
			# print(X_temp_acti)
			self.temp_values[i+1].append(np.array(X_temp_acti))

		X_temp = np.matmul(X_temp_acti, self.weights[self.n_layers-2]) + self.biases[self.n_layers-2]
		self.temp_values.append([np.array(X_temp)])
		X_temp_acti = (self.softmax(X_temp))
		# print(X_temp_acti)
		self.temp_values[-1].append(np.array(X_temp_acti))

		return X_temp_acti

	def backward(self, y, n):

		w_gradients = list(); b_gradients = list()
		prev = self.temp_values[-1][1] - y

		w_gradients.append((np.matmul((prev.T), self.temp_values[-2][1]).T)/n)
		b_gradients.append(np.sum(prev, axis=0)/n)

		for i in range(self.n_layers-2, 0, -1):
			prev = np.matmul(prev, self.weights[i].T) * self.acti_fns_grad[self.activation](self.temp_values[i][0])
			w_gradients.append((np.matmul((prev.T), self.temp_values[i-1][1]).T)/n)
			b_gradients.append(np.sum(prev, axis=0)/n)

		return w_gradients, b_gradients

	def update_weights(self, w_gradients, b_gradients):

		for i in range(self.n_layers-1):
			# print(self.weights[i]); print("Hey")
			self.weights[i] -= self.learning_rate * w_gradients[-i-1]
			# print(self.weights[i])
			self.biases[i] -= self.learning_rate * b_gradients[-i-1]
			# print(str(i) + "th weights printed")

		return None

	def fit(self, X, y, X_test, y_test):
		"""
		Fitting (training) the linear model.

		Parameters
		----------
		X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as training data.

		y : 1-dimensional numpy array of shape (n_samples,) which acts as training labels.

		Returns
		-------
		self : an instance of self
		"""
		n_batches = len(y)//self.batch_size
		if(X.shape[0] % self.batch_size != 0):
			n_batches += 1

		train_losses, test_losses = [], []
		for epoch in range(self.num_epochs):
			for b in range(n_batches):
				start = b*self.batch_size
				end = (b+1)*self.batch_size
				batch_X = X[start:end]; batch_y = y[start:end]
				y_prob = self.forward(batch_X)
				# print(y_prob)
				loss = self.crossEntropyLoss(batch_y, y_prob)
				w_gradients, b_gradients = self.backward(batch_y, len(batch_y))
				# print(w_gradients); print("Hey")
				# print(b_gradients)
				self.update_weights(w_gradients, b_gradients)
			print("Epoch:", epoch)
			train_prob = self.forward(X); train_loss = self.crossEntropyLoss(y, train_prob)
			train_losses.append(train_loss)
			test_prob = self.forward(X_test); test_loss = self.crossEntropyLoss(y_test, test_prob)
			test_losses.append(test_loss)
			print("Train Accuracy:", self.score(X, y))
			print("Test Accuracy:", self.score(X_test, y_test))

		EPOCHS = [i for i in range(1, self.num_epochs+1)]
		plt.plot(EPOCHS, train_losses)
		plt.plot(EPOCHS, test_losses)
		plt.legend(["Train loss", "Test loss"])
		plt.savefig("Loss Plot "+ self.activation)
		plt.close()
		# fit function has to return an instance of itself or else it won't work with test.py
		return self

	def predict_proba(self, X):
		"""
		Predicting probabilities using the trained linear model.

		Parameters
		----------
		X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

		Returns
		-------
		y : 2-dimensional numpy array of shape (n_samples, n_classes) which contains the
			class wise prediction probabilities.
		"""

		# return the numpy array y which contains the predicted values
		y_prob = self.forward(X)
		# print(y_prob)
		return y_prob

	def predict(self, X):
		"""
		Predicting values using the trained linear model.

		Parameters
		----------
		X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

		Returns
		-------
		y : 1-dimensional numpy array of shape (n_samples,) which contains the predicted values.
		"""

		# return the numpy array y which contains the predicted values
		y_prob = self.predict_proba(X)
		y_pred = np.argmax(y_prob, axis = 1)

		return y_pred

	def score(self, X, y):
		"""
		Predicting values using the trained linear model.

		Parameters
		----------
		X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

		y : 2-dimensional numpy array of shape (n_samples, n_classes) which acts as testing labels.

		Returns
		-------
		acc : float value specifying the accuracy of the model on the provided testing set
		"""

		# return the numpy array y which contains the predicted values
		y_pred =self.predict(X); accuracy = 0
		for i in range(len(y)):
			if(np.argmax(y[i])==y_pred[i]): accuracy += 1
		accuracy = accuracy/len(y)
		# print(y_pred.tolist()); print(y)
		return accuracy
