import pandas as pd
import numpy as np

def read_data(path):
	with open(path) as f:
		lines = f.readlines()
	lines = [eval(line.strip()) for line in lines]
	X, y = zip(*lines)
	X = np.array(X)
	y = np.array(y)
	return X, y