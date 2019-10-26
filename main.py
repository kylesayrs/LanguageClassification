from Trainer import Trainer

import numpy as np
import random

def main():
	cases = loadData()

	t = Trainer(cases, (15,8,5), num_trials=50000, step_size=0.008)
	t.train()

def loadData():
	cases = []

	with open("data/English.txt") as file:
		for line in file:

			# ACCOUNTS FOR MORE ENGLISH WORDS THAN CHINESE WORDS
			if random.randint(0, 5) != 0:
				continue

			name, misc = line.split(",")
			case  = {}

			case["name"]  = name.lower()

			case["input"] = []
			for i in range(15):
				if i < len(case["name"]) : case["input"].append((ord(case["name"][i])-97)/25)
				else				  : case["input"].append(0)
			case["input"] = np.array([case["input"]])

			case["target"] = np.array([[1, 0, 0, 0, 0]])

			cases.append(case)

	with open("data/Chinese.txt") as file:
		for line in file:
			name, misc = line.split(",")
			case  = {}

			case["name"]  = name.lower()

			case["input"] = []
			for i in range(15):
				if i < len(case["name"]) : case["input"].append((ord(case["name"][i])-97)/25)
				else				  : case["input"].append(0)
			case["input"] = np.array([case["input"]])

			case["target"] = np.array([[0, 1, 0, 0, 0]])

			cases.append(case)
	
	return cases

if __name__ == "__main__":
	main()