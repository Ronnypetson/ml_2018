import numpy as np
from sklearn.metrics import confusion_matrix
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
	models = []
	Y_ = []
	for i in range (10):
		lr_model = log_reg(learning_rate=learning_rate,train_iter=max_iter,class_=i)
		cost = lr_model.fit(train_X,train_Y)
		#plt.xlabel('iterations')
		#plt.ylabel('Cost')
		#plt.plot(cost)
		#plt.show()
		Y_.append(lr_model.predict(valid_X))
		models.append(lr_model)
	Y_ = np.transpose(Y_)
	Y_ = [np.argmax(np.array(y)) for y in Y_]
	conf = confusion_matrix(valid_Y,Y_)
	norm_acc = np.mean([conf[i][i]/np.sum(conf[i]) for i in range(conf.shape[0])])
	return norm_acc, models, conf #mse(valid_Y,Y_), np.transpose(thetas)

def multi_log_reg_(train_X,train_Y,valid_X,valid_Y,num_classes=10,learning_rate=0.1,max_iter=500):
	## Descida de gradiente
	lr_model = multi_log_reg(learning_rate=learning_rate,train_iter=max_iter,num_classes=num_classes)
	cost = lr_model.fit(train_X,train_Y)
	plt.xlabel('iterations')
	plt.ylabel('Cost')
	plt.plot(cost)
	plt.show()
	Y_ = lr_model.predict(valid_X)
	Y_ = [ np.argmax(v) for v in Y_ ]
	conf = confusion_matrix(valid_Y,Y_)
	norm_acc = np.mean([conf[i][i]/np.sum(conf[i]) for i in range(conf.shape[0])])
	return norm_acc, lr_model, conf #mse_(np.array(one_hot_Y),Y_), lr_model.theta

def neural_net_(train_X,train_Y,valid_X,valid_Y,learning_rate=0.0001,max_iter=2000,activation="relu",midl=100,fold=0):
	## Descida de gradiente
	lr_model = neural_net(learning_rate=learning_rate,train_iter=max_iter,layers_dims=[(785,midl),(midl,10)],activation=activation)
	cost = lr_model.fit(train_X,train_Y)
	fig = plt.figure()
	plt.xlabel('iterations')
	plt.ylabel('Cost')
	plt.plot(cost)
	plt.savefig('./plots/'+str(max_iter)+'_'+str(lr)+'_'+str(midl)+'_'+activation+'_'+str(fold)+'.png') #plt.show()
	Y_ = lr_model.predict(valid_X)
	Y_ = [ np.argmax(v) for v in Y_ ]
	conf = confusion_matrix(valid_Y,Y_)
	norm_acc = np.mean([conf[i][i]/np.sum(conf[i]) for i in range(conf.shape[0])])
	return norm_acc, lr_model, conf #norm_accuracy/lr_model.num_classes

def testar(test_X,test_Y,model):
	Y_ = model.predict(test_X)
	Y_ = [ np.argmax(v) for v in Y_ ]
	conf = confusion_matrix(test_Y,Y_)
	norm_acc = np.mean([conf[i][i]/np.sum(conf[i]) for i in range(conf.shape[0])])
	return norm_acc, conf

def testar_one_vs_all(test_X,test_Y,model):
	Y_ = []
	for m in model: Y_.append(m.predict(test_X))
	Y_ = np.transpose(Y_)
	Y_ = [np.argmax(np.array(y)) for y in Y_]
	conf = confusion_matrix(test_Y,Y_)
	norm_acc = np.mean([conf[i][i]/np.sum(conf[i]) for i in range(conf.shape[0])])
	return norm_acc, conf

## Read data from csv files
# Normalization of all data is division of all features by 255.0
# Load train-validation
train_X,train_Y = getXY('fashion-mnist-dataset/fashion-mnist_train.csv')
# Load test
test_X,test_Y = getXY('fashion-mnist-dataset/fashion-mnist_test.csv')

# "RegressaoLogistica_OneVsAll":log_reg_, "RegressaoLogistica_Multiclasse":multi_log_reg_, "RedeNeural":neural_net_

# num_iter = 2000
for lr in [0.0001,0.001,0.01,0.1]:
	for midl in [100,392,900]:
		for act in ["sigmoid","relu","leaky_relu","tanh"]:
			print("LR: %f, midl: %d, act: %s"%(lr,midl,act))
			# Aplicar validacao cruzada com k=5
			num_samples = train_X.shape[0]
			num_features = train_X.shape[1]
			k = 2
			chosen_models = {}
			chosen_confs = {}
			block_len = int(num_samples/k)
			methods = {"RedeNeural":neural_net_}
			for m in methods:
				print("Evaluating method "+m)
				norm_accuracy = np.zeros(k)
				models = []
				confs = []
				# Para cada fold
				for i in range(k):
					_valid_X = train_X[i*block_len:(i+1)*block_len]
					_train_X = np.concatenate((train_X[0:i*block_len],train_X[(i+1)*block_len:]),axis=0)
					_valid_Y = train_Y[i*block_len:(i+1)*block_len]
					_train_Y = np.concatenate((train_Y[0:i*block_len],train_Y[(i+1)*block_len:]),axis=0)
					norm_accuracy[i], model_, conf = methods[m](_train_X,_train_Y,_valid_X,_valid_Y,\
																						learning_rate=lr,max_iter=2000,activation=act,midl=midl,fold=i+1)
					models.append(model_)
					confs.append(conf)
				print("Validation mean accuracy for %s: %f"%(m,np.mean(norm_accuracy)))
				chosen_models[m] = models[np.argmax(norm_accuracy)]
				chosen_confs[m] = confs[np.argmax(norm_accuracy)]

			for m in methods:
				if m == "RegressaoLogistica_OneVsAll":
					print("Test mean accuracy for method %s: %f"%(m,testar_one_vs_all(test_X,test_Y,chosen_models[m])[0]))
				else:
					print("Test mean accuracy for method %s: %f"%(m,testar(test_X,test_Y,chosen_models[m])[0]))

