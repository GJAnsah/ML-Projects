import numpy as np
def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
    # Return a list where each element is the dot product of a row of 'a' with 'b'.
    # If the number of columns in 'a' does not match the length of 'b', return -1.
    
    a=np.array(a)
    b=np.array(b)
    N = len(b)
    if a.shape[1]!=N:
        return -1
    result = np.zeros(a.shape[0])
    for i in range(N):
        result += b[i]*a[:,i]
    return result
