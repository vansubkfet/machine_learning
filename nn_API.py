import numpy as np
import pdb

def breakpoint(X):
	print('\n------------------------trace---------------------------\n')
	print(X)
	print('\n------------------------trace---------------------------\n')
	pdb.set_trace()

def randInitializeWeights(L_in,L_out):
	# add bias weight
	epsilon = 0.12
	W = np.zeros((L_out,L_in+1)) 
	W = np.random.rand(L_out,L_in+1) * (2 * epsilon) - epsilon
	return (W)

def sigmoid(z):
	g = np.zeros((z.shape))
	g = 1.0 / (1.0 + np.exp(-z))
	return (g)
	
def sigmoidGradient(z):
	g = np.zeros((z.shape))
	sig = sigmoid(z)
	ONE = np.ones((z.shape))
	g = sig * (ONE - sig)
	return (g)

def Forwardpropagation_layer_i(theta,X,number_layer_in,number_layer_out):
	num_examples = X.shape[0]
	# calculate parameter for hidden layer
	Z = np.zeros((number_layer_out,num_examples))
	A = np.zeros((Z.shape)) 
	Z = theta.dot(X.T)    
	A = sigmoid(Z)
	return (A,Z)
	
def Backpropagation_i(theta1,theta2,input_layer_size,hidden_layer_size, num_labels,x,label):
	A2,Z2 = Forwardpropagation_layer_i(theta1,x,input_layer_size,hidden_layer_size)
	A2 = A2.reshape(hidden_layer_size,1)   #25x1
	Z2 = Z2.reshape(hidden_layer_size,1)
	# add bias parameter
	Z2 = np.vstack((np.ones((1,1)),Z2))  # Z2 = 26x1
	A2 = np.vstack((np.ones((1,1)),A2))  # A2 = 26x1
	
	A3,Z3 = Forwardpropagation_layer_i(theta2,A2.T,hidden_layer_size,num_labels)  # A3= 10x1
	
	A3 = A3.reshape(num_labels,1)
	Z3 = Z3.reshape(num_labels,1)
	y_k = np.zeros((num_labels,1))
	sigma3 = np.zeros((y_k.shape))
	y_k[label-1] = 1
	
	sigma3 = A3 - y_k    #sigma3 = 10x1
	sigma2 = ((theta2.T).dot(sigma3)) * (sigmoidGradient(Z2))   #sigma2 = 26x1
	theta2_grad_i = sigma3.dot(A2.T) # [10x1].[1x26] = 10x26
	sigma2 = sigma2[1:]
	theta1_grad_i = sigma2.dot(x)
	return (theta1_grad_i,theta2_grad_i)
	
def nnCostFunction(theta,input_layer_size,hidden_layer_size, num_labels, X, y, lamda):
	J = 0.0
	theta1 = np.zeros((hidden_layer_size,input_layer_size+1))
	theta2 = np.zeros((num_labels,hidden_layer_size+1))
	
	theta1 = np.reshape(theta[:(hidden_layer_size *(input_layer_size+1))],(hidden_layer_size,(input_layer_size+1)))
	theta2 = np.reshape(theta[(hidden_layer_size *(input_layer_size+1)):],(num_labels,(hidden_layer_size+1)))
	
	num_examples = X.shape[0]
	# add bias parameter for input layer
	X = np.hstack((np.ones((num_examples,1)),X))  # 5000x401
	# calculate forwardpropagation
	A2,Z2 = Forwardpropagation_layer_i(theta1,X,input_layer_size,hidden_layer_size) # 26x5000
	# add bias parameter
	
	A2 = np.hstack((np.ones((num_examples,1)),A2.T))  # 5000x26
	A3,Z3 = Forwardpropagation_layer_i(theta2,A2,hidden_layer_size,num_labels) #10x5000
	ONE = np.ones((num_examples,1))
	for k in range(1,num_labels+1):
		h_theta_k = np.zeros((num_examples,1))
		y_k_1 = np.zeros((num_examples,1))
		
		h_theta_k = A3[k-1,:].reshape(num_examples,1)
		
		k_idx = np.where(y==k)
		y_k_1[k_idx] = 1
		y_k_0 = ONE - y_k_1

		log_h_theta_k_1 = np.log(h_theta_k)
		log_h_theta_k_0 = np.log(ONE - h_theta_k)
		
		sum_1 = (y_k_1.T).dot(log_h_theta_k_1)
		sum_0 = (y_k_0.T).dot(log_h_theta_k_0)
		J += - (sum_1 + sum_0) 
	J = J/num_examples
	theta1_without_bias = theta1[:,1:]
	theta2_without_bias = theta2[:,1:]
	J += (lamda/(2*num_examples)) * ((np.sum(theta1_without_bias ** 2)) + (np.sum(theta2_without_bias ** 2)));
	return (J)

def nnThetaGradient(theta,input_layer_size,hidden_layer_size, num_labels, X, y, lamda):
	print('.')
	theta1 = np.zeros((hidden_layer_size,input_layer_size+1))
	theta2 = np.zeros((num_labels,hidden_layer_size+1))
	
	theta1 = np.reshape(theta[:(hidden_layer_size *(input_layer_size+1))],(hidden_layer_size,(input_layer_size+1)))
	theta2 = np.reshape(theta[(hidden_layer_size *(input_layer_size+1)):],(num_labels,(hidden_layer_size+1)))
	
	theta1_grad = np.zeros((theta1.shape))
	theta2_grad = np.zeros((theta2.shape))
	
	
	num_examples = X.shape[0]
	# add bias parameter for input layer
	X = np.hstack((np.ones((num_examples,1)),X))  # 5000x401
	
	# calculate theta gradient 
	for i in xrange(num_examples):
		x = np.zeros((1,X.shape[1]))
		x = X[i,:].reshape(X.shape[1],1)
		x = x.T
		theta_grad_1,theta_grad_2 = Backpropagation_i(theta1,theta2,input_layer_size,hidden_layer_size, num_labels,x,y[i])
		theta1_grad = theta1_grad + theta_grad_1
		theta2_grad	= theta2_grad + theta_grad_2
			
	theta1_grad = theta1_grad/num_examples
	theta2_grad = theta2_grad/num_examples
	# add regularized 
	theta1_grad[:,1:] += theta1[:,1:] * (lamda/num_examples)
	theta2_grad[:,1:] += theta2[:,1:] * (lamda/num_examples)
	
	flat_theta1 = theta1_grad.flatten(0).reshape((hidden_layer_size * (input_layer_size+1)),1)
	flat_theta2 = theta2_grad.flatten(0).reshape((num_labels *(hidden_layer_size+1)),1)
	flat_theta = np.vstack((flat_theta1,flat_theta2))
	theta = flat_theta[:,0]
	return (theta)
	
	
	
	
	
	
	
	
	
	
	
	