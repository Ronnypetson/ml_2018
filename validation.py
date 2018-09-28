import numpy as np
from preprocess import *
from log_reg import *
from multi_log_reg import *
from neural_net import *
from matplotlib import pyplot as plt

def mse(Y,Y_):
	cost = np.zeros(Y.shape[0])
	for i in range(len(cost)):
		#print(Y[i],Y_[i])
		cost[i] = 0.0 if int(Y_[i]) == int(Y[i]) else 1.0
	return np.mean((cost)**2)/2.0

def mse_(Y,Y_):
	return np.mean((Y-Y_)**2)/2.0

#def crossValidationError(Y, Y_):
	#return (np.sum((Y_ - Y)**2))/(2*len(Y))


def log_reg_(train_X,train_Y,valid_X,valid_Y,learning_rate=0.1,max_iter=500,class_=0):
	## Descida de gradiente
	thetas = []
	Y_ = []
	for i in range (10):
		lr_model = log_reg(learning_rate=learning_rate,train_iter=max_iter,class_=i)
		cost = lr_model.fit(train_X,train_Y)
		#plt.xlabel('iterations')
		#plt.ylabel('Cost')
		#plt.plot(cost)
		#plt.show()
		Y_.append(lr_model.predict(valid_X))
		thetas.append(lr_model.theta)
	Y_ = np.transpose(Y_)
	Y_ = [np.argmax(np.array(y)) for y in Y_]
	#print(len(Y_),len(valid_X))
	return mse(valid_Y,Y_), np.transpose(thetas)

def multi_log_reg_(train_X,train_Y,valid_X,valid_Y,num_classes=10,learning_rate=0.1,max_iter=500):
	## Descida de gradiente
	lr_model = multi_log_reg(learning_rate=learning_rate,train_iter=max_iter,num_classes=num_classes)
	cost = lr_model.fit(train_X,train_Y)
	#plt.xlabel('iterations')
	#plt.ylabel('Cost')
	#plt.plot(cost)
	#plt.show()
	Y_ = lr_model.predict(valid_X)
	one_hot_Y = []
	for i in range(valid_Y.shape[0]):
		y = np.zeros(num_classes)
		y[int(valid_Y[i])] = 1.0
		one_hot_Y.append(y)
	return mse_(np.array(one_hot_Y),Y_), lr_model.theta

def neural_net_(train_X,train_Y,valid_X,valid_Y,learning_rate=0.1,max_iter=500):
	## Descida de gradiente
	lr_model = neural_net(learning_rate=learning_rate,train_iter=max_iter)
	cost = lr_model.fit(train_X,train_Y)
	plt.xlabel('iterations')
	plt.ylabel('Cost')
	plt.plot(cost)
	plt.show()
	#Y_ = lr_model.predict(valid_X)
	#one_hot_Y = []
	#for i in range(valid_Y.shape[0]):
	#	y = np.zeros(num_classes)
	#	y[int(valid_Y[i])] = 1.0
	#	one_hot_Y.append(y)
	#return mse_(np.array(one_hot_Y),Y_), lr_model.theta

## Read data from csv files
# Load train-validation
train_X,train_Y = getXY('fashion-mnist-dataset/fashion-mnist_train.csv')
# Load test
test_X,test_Y = getXY('fashion-mnist-dataset/fashion-mnist_test.csv')

# Aplicar validacao cruzada com k=5
num_samples = train_X.shape[0]
num_features = train_X.shape[1]
k = 2
chosen_params = {"RedeNeural":neural_net_,"RegressaoLogistica_OneVsAll":np.zeros((num_features,10)), "RegressaoLogistica_Multiclasse":np.zeros((num_features,10))}
block_len = int(num_samples/k)
methods = {"RedeNeural":neural_net_} # "RegressaoLogistica_OneVsAll":log_reg_, "RegressaoLogistica_Multiclasse":multi_log_reg_
for m in methods:
	print("Evaluating method "+m)
	mean_losses = np.zeros(k)
	#params = np.zeros((k,num_features,10))
	for i in range(k):
		_valid_X = train_X[i*block_len:(i+1)*block_len]
		_train_X = np.concatenate((train_X[0:i*block_len],train_X[(i+1)*block_len:]),axis=0)
		_valid_Y = train_Y[i*block_len:(i+1)*block_len]
		_train_Y = np.concatenate((train_Y[0:i*block_len],train_Y[(i+1)*block_len:]),axis=0)
		methods[m](_train_X,_train_Y,_valid_X,_valid_Y)
		#mean_losses[i], params[i] = methods[m](_train_X,_train_Y,_valid_X,_valid_Y) # params[i]
	#print("Validation mean loss score %s: %f"%(m,np.mean(mean_losses)))
	#chosen_params[m] = params[np.argmin(mean_losses)]

#for m in methods:
#	print("Test mean loss score: %f"%mse(test_Y, 1.0/(1.0+np.exp(-np.dot(test_X,chosen_params[m])))))

