from Network import Network

import random
import numpy as np

from termcolor import colored

class Trainer:

	def __init__(self, cases, network_shape, num_trials=1, step_size=1):
		self.cases = cases
		self.num_trials = num_trials
		self.network_shape = network_shape
		self.success_history = [1] * 100

		self.NN = Network(*network_shape, step_size=step_size)

	def train(self):

		for i in range(self.num_trials):
			print(str(i) + ":", end="")
			rand = random.randint(0,len(self.cases)-1)
			self.NN.cycle(self.cases[rand]["input"])
			self.doPrinting(self.cases[rand])
			self.NN.backProp(self.cases[rand]["target"])

		
		_input = input("Show results? (y/n) ")
		if _input == "yes" or _input == "y":
			_input = input("Full results? (y/n) ")
			if _input == "yes" or _input == "y":
				self.showResults(full=True)
			else:
				self.showResults()

		_input = input("test: ")
		while _input != "exit":
			inputs = [0] * 15
			for i in range(len(_input)):
				inputs[i] = (ord(_input[i])-65)/25
			self.NN.cycle(np.array([inputs]))
			print(self.NN.getOutput())
			_input = input("test: ")

		

	def showResults(self, full=False):

		print("Successes: " + str(self.success_history.count(1)))
		print("Failures : " + str(self.success_history.count(0)))

		if full:
			self.NN.printState()

			for case in self.cases:
				self.NN.cycle(case["input"])
				self.doPrinting(case)

	def doPrinting(self, case):

		# Prints case
		if case["target"][0][0] == 1:
			print(colored(str(case["name"]), "cyan"), end=" ")
		elif case["target"][0][1] == 1:
			print(colored(str(case["name"]), "yellow"), end=" ")
		else:
			print(str(case["name"])+" ", end="")

		# Arrow
		print("-->", end=" ")

		# Prints output
		print((self.NN.getOutput()[0]*100).round()/100, end=" ")

		# Prints Correct/Incorrect
		max_choice = np.where(self.NN.getOutput()[0] == np.amax(self.NN.getOutput()[0]))[0]
		correct_choice = np.where(case["target"][0] == np.amax(case["target"][0]))[0][0]

		if max_choice == correct_choice:
			print(colored(" CORRECT! ","green"), end=" ")
			self.success_history.append(1)
		else:
			print(colored(" WRONG! ","red"), end=" ")
			self.success_history.append(0)

		# Prints cost
		print("Cost: " + colored((self.NN.calcCost(case["target"])*100).round()/100,'blue'), end=" ")

		# Prints success %
		print(self.success_history[len(self.success_history)-100 : len(self.success_history)].count(1), end="%")

		# End trial
		print("")

'''
# Load data
cases = []
cases.append({"input":np.array([[0, 0]]), "target":np.array([[0, 0.5]])})
cases.append({"input":np.array([[1, 0]]), "target":np.array([[1, 0.5]])})
cases.append({"input":np.array([[0, 1]]), "target":np.array([[1, 0.5]])})
cases.append({"input":np.array([[1, 1]]), "target":np.array([[0, 0.5]])})

t = Trainer(cases, (2,3,2))
t.train()
'''