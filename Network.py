import numpy as np
import math
import random

class Network:

	'''
	Function : Init
	Arguments: input_len, hidden_len, output_len
	Returns  : None
	Details  : Initializes nodes, weights, lengths
	'''
	def __init__(self, input_len, hidden_len, output_len, step_size=1):

		# Layer lengths
		self.input_len = input_len
		self.hidden_len = hidden_len
		self.output_len = output_len
		self.step_size = step_size

		# Default target value
		self.target = -1

		# Weight matricies
		self.weights_zero = np.random.randn(self.input_len, self.hidden_len + 1)
		self.weights_one = np.random.randn(self.hidden_len + 1, self.output_len)

	'''
	Function : Print State (Short)
	Arguments: Self
	Returns  : None
	Details  : Used to test things
	'''
	def printStateShort(self):
		print("case: ", end="")
		print(self.input_nodes, end="")
		print(" -- > ", end="")
		print(self.output_nodes)
		
	'''
	Function : Print State (Long)
	Arguments: Self
	Returns  : None
	Details  : Returns the state of the matricies for testing purposes
	'''
	def printState(self):
		print("self.input_nodes: ", end="")
		print(self.input_nodes)
		print("self.hidden_nodes: ", end="")
		print(self.hidden_nodes)
		print("self.output_nodes: ", end="")
		print(self.output_nodes)
		print("self.weights_zero: ", end="")
		print(self.weights_zero)
		print("self.weights_one: ", end="")
		print(self.weights_one)

	'''
	Function : Cycle
	Arguments: Input matrix
	Returns  : Error
	Details  : Calcultes new values from new inputs
	'''
	def cycle(self, input):
		# Substitute in input
		self.input_nodes = input
		np.append(self.input_nodes,1)

		# Forward propagation
		self.hidden_zs = np.dot(self.input_nodes, self.weights_zero)
		self.hidden_nodes = sigmoid(self.hidden_zs)
		np.append(self.hidden_nodes,1)

		self.output_zs = np.dot(self.hidden_nodes, self.weights_one)
		self.output_nodes = sigmoid(self.output_zs)

		return self.output_nodes

	'''
	Function : Calculate Cost
	Arguments: Target output
	Returns  : Gradient vector
	Details  : 
	'''
	def calcCost(self, target):
		return np.sum(target - self.output_nodes)**2

	'''
	Function : Calculate Cost Prime
	Arguments: Target output
	Returns  : Gradient vector
	Details  : 
	'''
	def calcCostPrime(self):
		return (self.target - self.output_nodes) * 2

	'''
	Function : Calculate Weights Gradients
	Arguments: Self
	Returns  : New weights
	Details  : Uses partial derivatives
	'''
	def calculateWeightsGV(self):

		dcda = self.calcCostPrime()
		dadz = sigmoidprime(self.output_zs)
		dzdw = self.hidden_nodes

		delta = dcda * dadz

		w_gv_one = np.dot(dzdw.transpose(), delta)


		dcda = np.dot(delta, self.weights_one.T)
		dadz = sigmoidprime(self.hidden_zs)
		dzdw = self.input_nodes

		delta = dcda * dadz

		w_gv_zero = np.dot(dzdw.transpose(), delta)


		return w_gv_one, w_gv_zero

	'''
	Function : Back Propogate
	Arguments: Target and arg stepsize
	Returns  : None
	Details  : Calculates Weights GVs and applies to weights
	'''
	def backProp(self, target):

		self.target = target

		# Calculate gradient vectors
		w_gv_one, w_gv_zero = self.calculateWeightsGV()

		# Apply gradients to weight matricies using step_size
		self.weights_zero += w_gv_zero * self.step_size
		self.weights_one += w_gv_one * self.step_size

	def getOutput(self):
		return self.output_nodes


'''
Function : Sigmoid
Arguments: x
Returns  : sigmoid of x
Details  : Used in calculating node values
'''
def sigmoid(x):

	return 1/(1+np.exp(-x))

'''
Function : Sigmoid Prime
Arguments: x
Returns  : Sigmoid prime of x
Details  : Used for calculating partial derivatives
'''
def sigmoidprime(x):
	
	return sigmoid(x) * (1 - sigmoid(x))

'''
# XOR
cases = []
cases.append({"input":np.array([[0, 0]]), "target":np.array([[0, 0]])})
cases.append({"input":np.array([[1, 0]]), "target":np.array([[1, 1]])})
cases.append({"input":np.array([[0, 1]]), "target":np.array([[1, 1]])})
cases.append({"input":np.array([[1, 1]]), "target":np.array([[0, 0]])})

X = np.array(([[0, 0]]), dtype=float)
y = np.array(([[0]]), dtype=float)

# Train
n = Network(2,3,2)

for i in range(1000):
	print("[-------]")
	rand = random.randint(0,len(cases)-1)
	n.cycle(cases[rand]["input"])
	n.backProp(cases[rand]["target"])
	n.printStateShort()

n.printState()

for case in cases:
	n.cycle(case["input"])
	n.printStateShort()
'''