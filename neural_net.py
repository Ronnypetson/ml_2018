import numpy as np

# Ativações
def relu(z):
	return np.array([max(v,0.0) for v in z])

def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))

# Derivadas das ativações
def relu_(z):
	return np.array([1.0 if v > 0 else 0.0 for v in z])

def sigmoid_(z):
	g = sigmoid(z)
	return g*(1.0-g)

acts = {"sigmoid":sigmoid,"relu":relu}
acts_ = {"sigmoid":sigmoid_,"relu":relu_}

class layer:
	def __init__(self,input_len,output_len,activation):
		self.shape = (input_len,output_len)
		self.activation = activation
		self.theta = np.random.uniform(-1.0,1.0,size=self.shape)

class neural_net:
	# Initializer
	# O primeiro neuronio de cada camada é o bias
	def __init__(self,learning_rate=0.1,train_iter=1000,mini_batch_len=100,layers_dims=[(785,100),(100,10)],\
							 activation="sigmoid"):
		self.learning_rate = learning_rate
		self.train_iter = train_iter
		self.mini_batch_len = mini_batch_len
		self.num_classes = layers_dims[-1][1]
		self.activation = activation
		self.layers = []
		for d in layers_dims:
			self.layers.append(layer(d[0],d[1],activation))

	def softmax(self,y):
		return np.exp(y)/np.sum(np.exp(y))

	def cross_entropy_loss(self,y,y_):
		return -np.dot(y,np.log(y_))

	def loss(self,y,y_):
		return -np.add(np.dot(y,np.log(y_))+np.dot((1.0-y),(np.log(1.0-y_))))

	def forward(self,X):
		a = [X] # Atentar para os índices: a[i], theta[i] -> z[i] -> a[i+1]
		z = [ np.dot(a[0],self.layers[0].theta) ]
		for i in range(1,len(self.layers)):
			a.append(acts[self.activation](z[i-1]))
			a[i][0] = 1.0
			z.append(np.dot(a[i],self.layers[i].theta))
		a.append(acts[self.activation](z[-1])) # Tem que guardar a ativação do final
		g_ = [ acts_[self.activation](z_) for z_ in z ]
		return a, g_

	# Fit parameters theta by mini-batch gradient descent
	def fit(self,X_train,Y_train):
		# Initialize parameters with small random values
		mean_losses = np.zeros(self.train_iter)
		# Update parameters iteratively
		for i in range(self.train_iter):
			#grad = np.zeros(self.theta.shape)
			indices = np.random.choice(X_train.shape[0],self.mini_batch_len,replace=False)
			deltas = []
			for j in range(len(self.layers)):
				deltas.append(np.zeros(self.layers[j].shape))
			# Compute gradient
			for ind in indices:
				x = X_train[ind]
				y = np.zeros(self.num_classes)
				y[int(Y_train[ind])] = 1.0
				act,act_ = self.forward(x)
				erro = [act[-1]-y]
				for j in range(len(self.layers)-2,-1,-1):
					erro.append( np.multiply( np.dot(self.layers[j+1].theta,erro[-1]),act_[j] ) ) # act_[j+1]?
				for j in range(len(self.layers)):
					deltas[j] = np.add(deltas[j],np.outer(act[j],erro[len(self.layers)-j-1]))
				mean_losses[i] += self.cross_entropy_loss(y,act[-1]) # changed here
			# Compute gradients
			deltas = [d*self.learning_rate/self.mini_batch_len for d in deltas]
			# Update parameters
			for j in range(len(self.layers)):
				self.layers[j].theta -= deltas[j]
			mean_losses[i] /= self.mini_batch_len
		return mean_losses

	# Estimate Y from X
	def predict(self,X_test):
		#return [np.argmax(self.softmax(np.dot(x,self.theta))) for x in X_test]
		return forward(X_test)[0]

