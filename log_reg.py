import numpy as np

class log_reg:
	# Initializer
	def __init__(self,learning_rate=0.1,train_iter=1000,mini_batch_len=100,class_=0):
		self.learning_rate = learning_rate
		self.train_iter = train_iter
		self.mini_batch_len = mini_batch_len
		self.class_ = class_

	def h(self,x):
		return 1.0/(1.0+np.exp(-np.dot(x,self.theta)))

	def loss_(self,x,y):
		return -np.log(self.h(x)) if y == 1.0 else -np.log(1.0-self.h(x))

	# Fit parameters theta by mini-batch gradient descent
	def fit(self,X_train,Y_train):
		# Create column with 1
		#X_train = np.insert(X_train,0,1,1)
		# Initialize parameters with small random values
		self.theta = np.random.normal(0.0,1.0,size=X_train[0].shape)
		mean_losses = np.zeros(self.train_iter)
		# Update parameters iteratively
		for i in range(self.train_iter):
			grad = np.zeros(self.theta.shape)
			indices = np.random.choice(X_train.shape[0],self.mini_batch_len,replace=False)
			# Compute gradient
			for ind in indices:
				x = X_train[ind]
				y = 1.0 if (Y_train[ind] == self.class_) else 0.0
				grad += (self.h(x)-y)*x # changed here
				mean_losses[i] += self.loss_(x,y) # changed here
			grad *= self.learning_rate/self.mini_batch_len
			mean_losses[i] /= self.mini_batch_len
			# Update parameters
			self.theta -= grad
		return mean_losses

	# Estimate Y from X
	def predict(self,X_test):
		#X_test = np.insert(X_test,0,1,1)
		return 1.0/(1.0+np.exp(-np.dot(X_test,self.theta)))
		#return np.array([1.0 if y > 0.5 else 0.0 for y in Y_]) # changed here for logistic regression
