import numpy as np
import matplotlib.pyplot as plt

max_train_M = 943 # need to save these for check loading
max_train_N = 1678

def load_data(filename):
	dat = np.genfromtxt(filename)
	# i's, j's, y_{ij}
	i_vals = dat[:,0]
	j_vals = dat[:,1]
	# need to compute M, need to compute N
	M = int(max(np.unique(i_vals)))
	N = int(max(np.unique(j_vals)))
	Y = np.array([[0.0 for i in range(max_train_N)] for j in range(max_train_M)])
	max_i = 0
	for i,j,k in dat:
		if i <= max_train_M  and j <= max_train_N:
			Y[int(i)-1][int(j)-1] = k

	return Y

def error(Y,U,V):
    W = Y.copy() # only compute error over observed values
    W[W > 0] = 1
    W[W == 0] = 0
    error = np.sum((W * (Y - np.dot(U, V.T)))**2)
    return error/len(Y[Y>0])

def factorize(Y,k,eta = .01,reg_val = 0,epsilon = .001):
	M,N = Y.shape # initializes the shapes

	U = .5 * np.random.randn(M,k) # sets up U and V
	V = .5 * np.random.randn(N,k)

	users,movies = Y.nonzero() # removes missing values

	prev_loss = 0
	loss = error(Y,U,V)
	iteration = 0

	while True: # does SGD
		for i,j in np.random.permutation(zip(users,movies)): # only trains over observed values
		    val = Y[i,j] - np.dot(U[i,:],V[j,:])
		    
		    U[i,:] += eta* (-reg_val * U[i,:] + V[j,:] * val)
		    V[j,:] += eta * (-reg_val * V[j,:] + U[i,:] * val)

		if iteration > 20: # another stopping condition
			break
		iteration +=1

	return U,V

if __name__ == '__main__':
	ks = [10,20,30,50,100]
	e_ins = []
	e_outs = []
	Y_train = load_data("set6data/train.txt")
	Y_test = load_data("set6data/test.txt")

	# Plot without regularization
	for k in ks:
		U,V = factorize(Y_train,k)
		e_ins.append(error(Y_train,U,V))
		e_outs.append(error(Y_test,U,V))

	plt.plot(ks,e_ins,label = "Training Error")
	plt.plot(ks,e_outs,label = "Test Error")
	plt.xlabel("Number of Latent Factors")
	plt.ylabel("Error")
	plt.legend()
	plt.title("Training and Test Error vs Latent Factors")
	plt.savefig("no_reg_error_vs_k.png")
	plt.show()

	# Plot with Regularization (includes lambda values for each k)
	e_ins = []
	e_outs = []
	regs = [-4,-3,-2,-1,0]
	for reg in regs:
		print reg
		e_in_temp = []
		e_out_temp = []
		for k in ks:
			U,V = factorize(Y_train,k,reg_val = 10**reg)
			e_in_temp.append(error(Y_train,U,V))
			e_out_temp.append(error(Y_test,U,V))
		e_ins.append(e_in_temp)
		e_outs.append(e_out_temp)

	for i in range(len(e_ins)):
		plt.plot(ks,e_ins[i],label = str(regs[i]))
	plt.xlabel("Latent Factors")
	plt.ylabel("Mean Training Error")
	plt.title("Training Error vs Latent Factors")
	plt.legend()
	plt.savefig("reg_error_vs_k_train.png")
	plt.show()

	for i in range(len(e_outs)):
		plt.plot(ks,e_outs[i],label = str(regs[i]))
	plt.xlabel("Latent Factors")
	plt.ylabel("Mean Test Error")
	plt.title("Test Error vs Latent Factors")
	plt.legend()
	plt.savefig("reg_error_vs_k_test.png")
	plt.show()

