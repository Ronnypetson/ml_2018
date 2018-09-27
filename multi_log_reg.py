import numpy as np

class multi_log_reg:
	# Initializer
	def __init__(self,learning_rate=0.1,train_iter=1000,mini_batch_len=100,num_classes=10):
		self.learning_rate = learning_rate
		self.train_iter = train_iter
		self.mini_batch_len = mini_batch_len
		self.num_classes = num_classes

	def softmax(self,y):
		return np.exp(y)/np.sum(np.exp(y))

	def cross_entropy_loss(self,y,y_):
		return -np.dot(y,np.log(y_))

	# Fit parameters theta by mini-batch gradient descent
	def fit(self,X_train,Y_train):
		# Initialize parameters with small random values
		self.theta = np.random.normal(0.0,1.0,size=(X_train.shape[1],self.num_classes))
		mean_losses = np.zeros(self.train_iter)
		# Update parameters iteratively
		for i in range(self.train_iter):
			grad = np.zeros(self.theta.shape)
			indices = np.random.choice(X_train.shape[0],self.mini_batch_len,replace=False)
			# Compute gradient
			for ind in indices:
				x = X_train[ind]
				y = np.zeros(self.num_classes)
				y[int(Y_train[ind])] = 1.0
				y_ = self.softmax(np.dot(x,self.theta))
				grad += np.outer(x,np.transpose(y_-y)) # changed here
				mean_losses[i] += self.cross_entropy_loss(y,y_) # changed here
			grad *= self.learning_rate/self.mini_batch_len
			mean_losses[i] /= self.mini_batch_len
			# Update parameters
			self.theta -= grad
		return mean_losses

	# Estimate Y from X
	def predict(self,X_test):
		return [np.argmax(self.softmax(np.dot(x,self.theta))) for x in X_test]

