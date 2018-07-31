#!/usr/bin/env python

import sys
import numpy as np
import pandas as pd
import math as math

from math import sin
from decimal import Decimal
from random import shuffle


# Define Input Variables
numberofgrains=int(sys.argv[3])
numberoftimesteps=int(sys.argv[4])
learning_rate=float(sys.argv[5])
nodenumber=int(sys.argv[6])

# Open Strain Data

strain_data = np.loadtxt(sys.argv[1], delimiter=' ')
strain_data=strain_data[:,6:10]
strain_data=(strain_data - strain_data.min())/(strain_data.max() - strain_data.min())
# Open Fabric Data

fabric_data = np.genfromtxt(sys.argv[2], delimiter=(8, 8,8,6,6,6,6,6,3,5,3, 3), skip_header=(7))

# Training set, test set and verification data set formulation
print(fabric_data.shape)
print(strain_data.shape)
print(" stop ")

# Sort fabric data
#fabric_data[fabric_data[:,9].argsort()]

# Main Rearrang Loop

# Create Empty Input Array
All_Input = np.zeros(shape=(numberofgrains*3 + 4,numberoftimesteps*numberofgrains))
All_Output = np.zeros(shape= (3,numberoftimesteps*numberofgrains))
All_Input_cond=np.zeros(shape=(7,numberoftimesteps*numberofgrains))

print(All_Input.shape)
print(All_Output.shape)
print(All_Input_cond.shape)

print(fabric_data[0,0])

k=0
for j in range(numberoftimesteps-1):


	for i in range(numberofgrains-1):

		All_Output[0,k]=(fabric_data[i+((j+1)*numberofgrains),0]+180)/360
		All_Output[1,k]=(fabric_data[i+((j+1)*numberofgrains),1])/180		
		All_Output[2,k]=(fabric_data[i+((j+1)*numberofgrains),2]+180)/360
	
		All_Input_cond[0,k]=(fabric_data[i+(j*numberofgrains),0]+180)/360
		All_Input_cond[1,k]=(fabric_data[i+(j*numberofgrains),1]+180)/360
		All_Input_cond[2,k]=(fabric_data[i+(j*numberofgrains),2]+180)/360
		All_Input_cond[3,k]=strain_data[j,0]
		All_Input_cond[4,k]=strain_data[j,1]
		All_Input_cond[5,k]=strain_data[j,2]
		All_Input_cond[6,k]=strain_data[j,3]

		k+=1


np.savetxt('All_Output.txt', All_Output, delimiter=' ')
np.savetxt('Strain_out.txt', strain_data,delimiter=' ')

np.savetxt('All_Input_cond.txt', All_Input_cond, delimiter=' ')
# Randomly shuffle data
#np.random.shuffle(np.transpose(r))

# Seperate data formates


# Construction of Nueral Network

X=All_Input_cond.transpose()
y=All_Output.transpose()
X1=All_Input_cond.transpose()
y1=All_Output.transpose()

X=X[200000:1300000,:]
y=y[200000:1300000,:]
np.savetxt('T1.txt', X, delimiter=' ')
np.savetxt('T2.txt', y, delimiter=' ')
xPredicted=X1[190,:]
yPredicted=y1[190,:]
#X=np.array(([2,9],[1,5],[3,6]),dtype=float)
#y=np.array(([92],[86],[89]),dtype=float)
#xPredicted = np.array(([4,8]), dtype=float)

#X = X/np.amax(X, axis=0)
#xPredicted = xPredicted/np.amax(xPredicted, axis=0)
#y = y/100


class Neural_Network(object):

	def __init__(self,rw,inputSize,outputSize,hiddenSize):

		if rw==0:

			self.W1=np.loadtxt('w1_best.txt', delimiter=' ')
			self.W2=np.loadtxt('w2_best.txt', delimiter=' ')
	
		if rw==1:

			self.W1=np.loadtxt('w1_b.txt', delimiter=' ')
			self.W2=np.loadtxt('w2_b.txt', delimiter=' ')	

		else:	

			self.inputSize = inputSize
			self.outputSize = outputSize
			self.hiddenSize = hiddenSize

			self.W1 = np.random.randn(self.inputSize, self.hiddenSize)
			self.W2 = np.random.randn(self.hiddenSize, self.outputSize)


		


	def forward(self, X):
		self.z = np.dot(X, self.W1)
		self.z2 = self.sigmoid(self.z)
		self.z3 = np.dot(self.z2, self.W2)
		o = self.sigmoid(self.z3)
		return o

	def sigmoid(self,s):
		return 1/(1+np.exp(-s))

	def sigmoidPrime(self,s):
		return s*(1-s)

	def backward(self,X,y,o):

		self.o_error = y-o
		self.o_delta = self.o_error*self.sigmoidPrime(o)

		self.z2_error = self.o_delta.dot(self.W2.T)
		self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2)

		self.W1 += X.T.dot(self.z2_delta)
		self.W2 += self.z2.T.dot(self.o_delta)

	def train(self, X, y):
		o = self.forward(X)
		self.backward(X, y, o)

	def saveWeights(self):
		np.savetxt("w1_b.txt", self.W1, fmt="%s")
		np.savetxt("w2_b.txt", self.W2, fmt="%s")


	def predict(self):
		print("Predicted data based on trained weights: ")
		print("Input (scaled): \n" + str(xPredicted))
		print("Output: \n" + str(self.forward(xPredicted)))

	def predict1(self):
		return self.forward(xPredicted)

# Train Data


best_val=1
NN = Neural_Network(0,7,3,21)
for i in range(10):


	for j in range(30):
		#print("# " + str(i) + "\n")
		#print("Input (scaled): \n" + str(X))
		#print("Actual Output: \n" + str(y))
		#print("Predicted Output: \n" + str(NN.forward(X)))
		print(str(np.mean(np.square(y - NN.forward(X)))))
		NN.train(X, y)
		best_val=(np.mean(np.square(y - NN.forward(X))))
		NN.saveWeights

	
	if best_val < 0.01:

		NN=None
		NN = Neural_Network(1,7,3,21)
	else:
		NN=None
		NN = Neural_Network(2,7,3,21)

		
	
NN = Neural_Network(0,7,3,21)
NN.saveWeights()
NN.predict()
print(yPredicted)

NN = Neural_Network(0,7,3,21)

xp=X1[180,:]
for k in range(170000,170300):

	xPredicted=xp
	xp_n=NN.predict1()

#	print("stop")
#	print(xp[0:3])
	print(xp_n)
	xp=X1[k,:]
	print(xp[0:3])
#	xp[0:3]=xp_n	
#	print(xp[0:3])	

	




