#https://www.deep-ml.com/problems/83?from=Linear%20Algebra

import numpy as np

def calculate_dot_product(vec1, vec2) -> float:
	"""
	Calculate the dot product of two vectors.
	Args:
		vec1 (numpy.ndarray): 1D array representing the first vector.
		vec2 (numpy.ndarray): 1D array representing the second vector.
	"""
	# Your code here
	result = 0
	for v1, v2 in zip(vec1,vec2):
		result+=v1*v2
	return result
