import numpy as np
import os
import sys
import scipy.io as sio
import scipy.optimize as opt
import nn_API as API
import matplotlib.pyplot as plt

def load_data(file):
    data = sio.loadmat(file)
    return data

def predict(theta1,theta2,X):
	# Useful values
	num_examples = X.shape[0];
	num_labels = theta2.shape[0];
	
	prediction = np.zeros((num_examples,1))
	X = np.hstack((np.ones((num_examples,1)),X))  # 5000x401
	A2,Z2 = API.Forwardpropagation_layer_i(theta1,X,X.shape[1],theta1.shape[0])
	A2 = np.hstack((np.ones((num_examples,1)),A2.T))  # 5000x26
	A3,Z3 = API.Forwardpropagation_layer_i(theta2,A2,theta1.shape[0],num_labels)
	A3 = A3.T
	prediction = np.argmax(A3,axis=1) + 1
	return (prediction)
	


input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25  # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10  

input = load_data('ex4data1.mat')
#get X and y
X = input['X']    	# 5000x400
y = input['y']		# 5000x1

theta = load_data('ex4weights.mat')
# get weight matrix 
theta2 = theta['Theta2']	#  10x26 
theta1 = theta['Theta1']	#  25x401
print('Size of X= ' + str(X.shape))
print('Size of y= ' + str(y.shape))
print('Size of theta1= ' + str(theta1.shape))
print('Size of theta2= ' + str(theta2.shape))
print('---------------------------------------------')
print('display an example')
x = X[2000,:]
x = x.reshape(20,20)
x = np.abs(x)
max_value = np.max(x) / 10
x = x / max_value
plt.imshow(np.uint8(x))
plt.show()

lamda = 0
#Unroll parameters
flat_theta1 = theta1.flatten(0).reshape((hidden_layer_size * (input_layer_size+1)),1)
flat_theta2 = theta2.flatten(0).reshape((num_labels *(hidden_layer_size+1)),1)
flat_theta = np.vstack((flat_theta1,flat_theta2))
J = API.nnCostFunction(flat_theta,input_layer_size,hidden_layer_size,num_labels,X,y,lamda)
print('Cost function J = %f \n' % (J))
theta = API.nnThetaGradient(flat_theta,input_layer_size,hidden_layer_size,num_labels,X,y,lamda)
theta1_grad = np.reshape(theta[:(hidden_layer_size *(input_layer_size+1))],(hidden_layer_size,(input_layer_size+1)))
theta2_grad = np.reshape(theta[(hidden_layer_size *(input_layer_size+1)):],(num_labels,(hidden_layer_size+1)))

print('\nEvaluating sigmoid gradient...\n')
z = np.array([-1,-0.5,0,0.5,1])
g = API.sigmoidGradient(z);
print (g)

print('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = API.randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = API.randInitializeWeights(hidden_layer_size, num_labels);
#print(initial_Theta2)
print('\nTraining Neural Network... \n')
lamda = 1;
flat_theta1 = initial_Theta1.flatten(0).reshape((hidden_layer_size * (input_layer_size+1)),1)
flat_theta2 = initial_Theta2.flatten(0).reshape((num_labels *(hidden_layer_size+1)),1)
flat_theta = np.vstack((flat_theta1,flat_theta2))

theta = opt.fmin_cg(API.nnCostFunction, fprime=API.nnThetaGradient,x0=flat_theta, args=(input_layer_size,hidden_layer_size,num_labels,X, y, lamda), maxiter=50)

theta1_grad = np.reshape(theta[:(hidden_layer_size *(input_layer_size+1))],(hidden_layer_size,(input_layer_size+1)))
theta2_grad = np.reshape(theta[(hidden_layer_size *(input_layer_size+1)):],(num_labels,(hidden_layer_size+1)))


pred= predict(theta1_grad,theta2_grad,X)
num_examples = X.shape[0]
count = 0.0
for i in xrange(num_examples):
	if pred[i]==y[i]:
		count += 1

print('\nTraining Set Accuracy: %f\n' % (count/num_examples * 100))

'''









				