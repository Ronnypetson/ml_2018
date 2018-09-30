import numpy as np

# Le o arquivo e retorna os dados divididos em X e Y
def getXY(fl_path='fashion-mnist-dataset/fashion-mnist_train.csv'):
	# Loading the table
	fmnist_table = np.genfromtxt(fl_path,dtype=np.float32,delimiter=',',skip_header=1,encoding='ascii')
	np.random.shuffle(fmnist_table)
	Y = np.array([t[0] for t in fmnist_table])
	X = [t/255.0 for t in fmnist_table]
	for x in X: x[0] = 1.0
	return np.array(X), Y

