import numpy as np
import sys
from utils import *

if __name__ == "__main__":
	debug = True
	if len(sys.argv) > 1 and sys.argv[1] == "nodebug":
		debug = False		
		
"""P 1.1 Implement the Algorithm 5.2.1 (Householder QR), and calculate the QR decomposition for
the matrix"""
def house(x):
    m = len(x)
    sigma = np.dot(x[1:], x[1:])  # σ = x(2:m)^T * x(2:m)
    
    # Initialize v with 1 in the first position and x(2:m) in the rest
    v = np.array(x, dtype=float)
    v[1:] = x[1:]
    v[0] = 1
    if sigma == 0 and x[0] >= 0:
        beta = 0
    elif sigma == 0 and x[0] < 0:
        beta = -2
    else:
        mu = np.sqrt(x[0]**2 + sigma)
        if x[0] <= 0:
            v[0] = x[0] - mu
        else:
            v[0] = -sigma / (x[0] + mu)
        
        beta = 2 * v[0]**2 / (sigma + v[0]**2)
        v = v / v[0]  # Normalize v by v[0]
    
    return v, beta


def HouseHolderQR(A: np.array):
    if A.shape[0] < A.shape[1]: 
        return "Matrix has more columns than rows"
    
    m = A.shape[0]
    n = A.shape[1]
    Q = np.identity(m, dtype=float)
    tempQ = np.identity(m, dtype=float)
    for j in range(n):
        v, beta = house(A[j:, j])
        A[j:, j:] = (np.identity(A.shape[0] - j) - beta * np.outer(v, v)) @ A[j:, j:]
        if j < m - 1:
            A[j+1:, j] = v[1:m - j + 1]
            # Q = Q @ tempQ

    return A
# Example usage
A = np.array([
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1]
], dtype=float)
if debug or "q1" in sys.argv:
    print("Question 1")
    A = HouseHolderQR(A)
    print(A)