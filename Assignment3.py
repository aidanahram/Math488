import numpy as np
import sys
from utils import *
from Assignment1 import * 
from Assignment2 import *

debug = False
if __name__ == "__main__":
	debug = True
	if len(sys.argv) > 1 and sys.argv[1] == "nodebug":
		debug = False	

"""P 1.1 Implement the Power Method in 7.3.1. See 8.2.1 as well. And use an example to verify your
code can find the largest eigenvalue and corresponding eigenvector."""
def powerMethod(A: np.array, q: np.array, maxIter: int):
	for i in range(maxIter):
		z = matrixMultiplication(A, q)
		q = z / np.linalg.norm(z)
		l = q.T @ A @ q

	return q
A = np.array([
	[3, 0],
	[0, 2]
])
q = np.array([
	[1],
	[1]
])
A2 = np.array([
	[2,0,0,0],
	[1,4,0,1],
	[3,1,5,2],
	[0,0,0,1]
])
q2 = np.array([
	[1],
	[0],
	[1],
	[0]
])
# Eigenvalues
# λ_1 = 5
# λ_2 = 4
# λ_3 = 2
# λ_4 = 1
# Eigenvectors
# v_1 = (0, 0, 1, 0)
# v_2 = (0, -1, 1, 0)
# v_3 = (-6, 3, 5, 0)
# v_4 = (0, -4, -5, 12)
if __name__ == "__main__" and (debug or "q1" in sys.argv):
	print("Question 1")
	iterations = 100
	q = powerMethod(A.copy(), q, iterations)
	print(f"After {iterations} q is:\n {q}")
	q = powerMethod(A2.copy(), q2, iterations)
	print(f"After {iterations} q is:\n {q}")

"""P 1.2 Implement the Orthogonal Iteration in 7.3.2. See 8.2.4 as well. And use an example to verify
your code can find the first few largest eigenvalues and their corresponding eigenvector subspace."""
def orthogonalIteration(A: np.array, maxIter: int):
	n = A.shape[0]
	Q = np.random.rand(n, n)
	for i in range(maxIter):
		Q, R = np.linalg.qr(Q)
		Q = matrixMultiplication(A, Q)
	return Q