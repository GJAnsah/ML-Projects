import numpy as np

def gradient_direction_magnitude(gradient: list) -> dict:
	"""
	Calculate the magnitude and direction of a gradient vector.
	
	Args:
		gradient: A list representing the gradient vector
	
	Returns:
		Dictionary containing:
		- magnitude: The L2 norm of the gradient
		- direction: Unit vector in direction of steepest ascent
		- descent_direction: Unit vector in direction of steepest descent
	"""
	# Your code here
	gradient = np.array(gradient)
	result = dict()
	result['magnitude'] = np.sqrt(np.dot(gradient,gradient.T))
	result['direction']= (gradient/result['magnitude'] if result['magnitude']  else [0,0])
	result['descent_direction']= -1*result['direction']
	return result
