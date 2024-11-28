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
if __name__ == "__main__" and (debug or "q1" in sys.argv):
	print("Question 1")
	q = powerMethod(A.copy(), q, 100)
	print("q is:\n", q)

# Suppose A ∈ Cn×n and X−1AX = diag(λ1,...,λn) with X = [ x1 |···| xn ] . Assume
# that
# |λ1| > |λ2| ≥···≥ |λn|.
# Given a unit 2-norm q(0) ∈ Cn, the power method produces a sequence of vectors q(k)
# as follows:
# for k = 1, 2,...
# z(k) = Aq(k−1)
# q(k) = z(k)
# / z(k) 2 (7.3.3)
# λ(k) = [q(k)
# ]
# HAq(k)