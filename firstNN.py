import numpy as np
import sys
import random
from tkinter.filedialog import askopenfilename

#sigmoid function
def mysigmoid(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

#getting input from txt file
filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
f = open(filename)
data = f.read()
lines = data.splitlines()
input_arr = np.empty((0,2))
output_arr = np.array([])

for line in lines:
  input_temp = np.array([])
  input_temp = np.append(input_temp, np.float32(line.split()[0]))
  input_temp = np.append(input_temp, np.float32(line.split()[1]))
  input_arr = np.append(input_arr, [input_temp], axis=0)
  output_arr = np.append(output_arr, np.float32(line.split()[2]))

random.shuffle(input_temp)
partition = len(input_arr)*.9

#splitting data for populating training and testing input and output
input_train_data = input_arr[:int(partition)]
input_test_data = input_arr[int(partition):]
output_train_data = output_arr[:int(partition)]
output_test_data = output_arr[int(partition):]
#Transposing output array
output_train_data = output_train_data.reshape(-1, 1)

#seed random numbers for weight calculation
np.random.seed(1)

#Training neural network
#initialize random weights
weights1 = 2*np.random.random((2,len(input_train_data))) - 1
weights2 = 2*np.random.random((len(input_train_data),1)) - 1

for i in range(100000):
  #feed forward 
  layer1 = mysigmoid(np.dot(input_train_data, weights1))
  layer2 = mysigmoid(np.dot(layer1, weights2))
  #Back propagation
  #error calculation
  error = output_train_data - layer2
  if(i % 10000) == 0:
    print("Error:", np.mean(np.abs(error))) 
  #multiple error with gradient 
  layer2_delta = error * mysigmoid(layer2,True)
  layer1_delta = np.dot(layer2_delta, weights2.T) * mysigmoid(layer2,True)
  #update weights
  weights2 += np.dot(layer1.T, layer2_delta)
  weights1 += np.dot(input_train_data.T, layer1_delta)
  
#Testing neural network
#feed forward 
layer1 = mysigmoid(np.dot(input_test_data, weights1))
layer2 = mysigmoid(np.dot(layer1, weights2))
  
print("Output After Training:")
layer2 = layer2.reshape(-1, 1)
for i in range(len(layer2)):
  print("Actual output:", layer2[i], "Ideal output:", output_test_data[i])
