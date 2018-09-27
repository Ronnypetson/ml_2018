import numpy as np
from preprocess import *
from log_reg import *
from multi_log_reg import *
from matplotlib import pyplot as plt

def mse(Y,Y_):
	return np.mean((Y-Y_)**2)

def log_reg_(train_X,train_Y,valid_X,valid_Y,learning_rate=0.1,max_iter=500,class_=0):
	## Descida de gradiente
	lr_model = log_reg(learning_rate=learning_rate,train_iter=max_iter,class_=class_)
	cost = lr_model.fit(train_X,train_Y)
	plt.xlabel('iterations')
	plt.ylabel('Cost')
	plt.plot(cost)
	plt.show()
	Y_ = lr_model.predict(valid_X)
	return mse(valid_Y,Y_), lr_model.theta

def multi_log_reg_(train_X,train_Y,valid_X,valid_Y,num_classes=10,learning_rate=0.1,max_iter=500):
	## Descida de gradiente
	lr_model = multi_log_reg(learning_rate=learning_rate,train_iter=max_iter,num_classes=num_classes)
	cost = lr_model.fit(train_X,train_Y)
	plt.xlabel('iterations')
	plt.ylabel('Cost')
	plt.plot(cost)
	plt.show()
	Y_ = lr_model.predict(valid_X)
	return mse(valid_Y,Y_), lr_model.theta

## Read data from csv files
# Load train-validation
train_X,train_Y = getXY('fashion-mnist-dataset/fashion-mnist_train.csv')
# Load test
test_X,test_Y = getXY('fashion-mnist-dataset/fashion-mnist_test.csv')

# Aplicar validacao cruzada com k=5
num_samples = train_X.shape[0]
num_features = train_X.shape[1]
k = 2
block_len = int(num_samples/k)
methods = {"multi_log_reg":multi_log_reg_} # "log_reg":log_reg_
for m in methods:
	print("Evaluating method "+m)
	mean_losses = np.zeros(k)
	params = np.zeros((k,num_features))
	for i in range(k):
		_valid_X = train_X[i*block_len:(i+1)*block_len]
		_train_X = np.concatenate((train_X[0:i*block_len],train_X[(i+1)*block_len:]),axis=0)
		_valid_Y = train_Y[i*block_len:(i+1)*block_len]
		_train_Y = np.concatenate((train_Y[0:i*block_len],train_Y[(i+1)*block_len:]),axis=0)
		mean_losses[i],params[i] = methods[m](_train_X,_train_Y,_valid_X,_valid_Y)
	print("Validation mean loss score: %f"%np.mean(mean_losses))
	#chosen_params = params[np.argmax(mean_losses)]
	#print("Test mean loss score: %f"%r2_score(test_Y,np.dot(test_X,chosen_params)))

