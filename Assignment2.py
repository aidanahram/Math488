import numpy as np
import sys
from utils import *

if __name__ == "__main__":
	debug = True
	if len(sys.argv) > 1 and sys.argv[1] == "nodebug":
		debug = False		
		
"""P 1.1 Implement the Algorithm 5.2.1 (Householder QR), and calculate the QR decomposition for
the matrix"""
def house(x: np.array):
	normsq = 0
	for i in range(1, x.shape[0]):
		normsq += x[i] ** 2
	#norm = np.sqrt(norm)  # Sum of squares from x[1] to x[m-1]


	v = np.array(x, dtype=float)
	v[0] = 1
	if normsq == 0 and x[0] >= 0:
		beta = 0
	elif normsq == 0 and x[0] < 0:
		beta = -2
	else:
		u = np.sqrt(x[0] ** 2 + normsq)
		if x[0] <= 0:
			v[0] = x[0] - u
		else:
			v[0] = -normsq / (x[0] + u)
		beta = 2 * (v[0] ** 2) / (normsq + (v[0] ** 2))
		v = v / v[0]

	return v, beta


def HouseHolderQR(A: np.array):
    if A.shape[0] < A.shape[1]: 
        return "Rows are less than columns"
    return

A = np.array([
	[1, 1, 0],
	[1, 0, 1],
	[0, 1, 1]
])
if debug or "q1" in sys.argv:
	print("Question 1")
	res = house(A[:, 0])
	print(res)
	m = multiply_vector_transpose(res[0])
	print("M is ", res[1] * m)
	# print(res[0] * m)
	P = np.identity(A.shape[0]) - m
	print("P is \n", P)
	print(P @ A[:,0])
	# print(matrixMultiplication(P, A))
