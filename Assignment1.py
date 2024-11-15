import numpy as np
import sys
from utils import *

debug = False
if __name__ == "__main__":
	debug = True
	if len(sys.argv) > 1 and sys.argv[1] == "nodebug":
		debug = False		

"""P 1.1 Implement the Algorithm 3.1.3 column-oriented forward substitution. Find an example to
verify your code."""
def columnOrientedForwardSub(L: np.array, b: np.array):
	"""
	Function takes a lower triangular matrix L and a vector matrix b and solves the equation Lx = b.

	It returns the vector x that satisfies the equation.
	"""
	if L.shape[0] != L.shape[1]:
		return "Matrix not square"
	if L.shape[0] != b.shape[0]:
		return "Ax = b is not solveable"
	
	for j in range(L.shape[0] - 1):
		b[j] = b[j] / L[j][j]
		for i in range(j+1, L.shape[0]):
			b[i] = b[i] - b[j]*L[i][j]

	b[b.shape[0]-1] = b[b.shape[0]-1]/L[b.shape[0]-1][b.shape[0]-1]
	return b

# Question 1 Test
matrix = np.array([[2, 0, 0], 
				   [1, 5, 0],
				   [7, 9, 8]])
b = np.array([6.0, 2.0, 5.0])
if debug or "q1" in sys.argv:
	print("Question 1")
	print(columnOrientedForwardSub(matrix, b))
	print()


"""P 1.2 Implement the Algorithm 3.1.4 column-oriented back substitution. Find an example to verify
your code."""
def columnOrienteBackwardSub(U: np.array, b: np.array):
	"""
	Function takes a non-singular upper triangular matrix U and a vector matrix b and solves the equation Ux = b.

	It returns the vector x that satisfies the equation.
	"""
	if U.shape[0] != U.shape[1]:
		return "Matrix not square"
	if U.shape[0] != b.shape[0]:
		return "Ax = b is not solveable"
	
	for j in range(U.shape[0] - 1, 0, -1):
		b[j] = b[j] / U[j][j]
		for i in range(j):
			b[i] = b[i] - b[j] * U[i][j]
	b[0] = b[0] / U[0][0]
	return b

matrix = np.array([
	[2.0, 1.0, -1.0], 
	[0.0, 3.0, 2.0],
	[0.0, 0.0, 4.0]
])
b = np.array([5.0, 11.0, 12.0])
if debug or "q2" in sys.argv:
	print("Question 2")
	print(columnOrienteBackwardSub(matrix, b)) 
	print("Expected Result is [19/6, 5/3, 3.0]\n")
	matrix = np.array([
		[3, -2, 4],
		[0, 1, 5],
		[0, 0, 6]
	])
	b = np.array([10.0, 7.0, 12.0])
	print(columnOrienteBackwardSub(matrix, b)) 
	print("Expected Result is [-4/3, -3.0, 2.0]\n")


"""P 1.3 Implement the algorithm 3.2.1 LU decomposition without pivoting. and calculate A = LU
where"""
A = np.array([
	[2, 1, 1, 0],
	[4, 3, 3, 1],
	[8, 7, 9, 5],
	[6, 7, 9, 8]
])

def outerProductLU(A: np.array):
	"""
	Function takes a non-singular matrix A and computes A = LU. Where L is lower triangular and U is upper triangular

	Returns matrix L and matrix U
	"""
	if A.shape[0] != A.shape[1]:
		return "Matrix not square"

	for k in range(A.shape[0] - 1):
		for rho in range(k+1, A.shape[0]):
			A[rho][k] = A[rho][k] / A[k][k]
			for r in range(k+1, A.shape[0]):
				A[rho][r] = A[rho][r] - A[rho][k]*A[k][r]

	# Extract L and U from A
	L = np.tril(A, -1) + np.eye(A.shape[0])  # Lower triangular matrix with ones on the diagonal
	U = np.triu(A)  # Upper triangular matrix

	return L, U

if debug or "q3" in sys.argv:
	print("Question #3")
	L, U = outerProductLU(A)
	print("L = ", L)
	print("U = ", U)
	print()

"""P 1.4 Implement the algorithm 3.4.1 LU decomposition with partial pivoting, and calculate P A =
LU where"""
def maxPivotIndex(A: np.array, column: int):
	row, m = column, 0
	for i in range(column, A.shape[0]):
		if A[i][column] > m:
			m = A[i][column]
			row = i
	return row

def LUWithPartialPivoting(A: np.array):
	"""
	Function takes a non-singular matrix A and computes PA = LU. Where P is a permutation matrix, L is lower triangular and U is upper triangular

	Returns matrix P, L, and U
	"""
	if A.shape[0] != A.shape[1]:
		return "Matrix not square"
	
	P = np.identity(A.shape[0])

	for i in range(A.shape[0]):
		u = maxPivotIndex(A, i)
		I = np.identity(A.shape[0])
		I = swapRows(I, i, u)
		A = swapRows(A, i, u)
		P = matrixMultiplication(I, P)
		if A[i][i] != 0:
			for rho in range(i+1, A.shape[0]):
				A[rho][i] = A[rho][i] / A[i][i]
				for r in range(i+1, A.shape[0]):
					A[rho][r] = A[rho][r] - A[rho][i] * A[i][r]
	# Extract L and U from A
	L = np.tril(A, -1) + np.eye(A.shape[0])  # Lower triangular matrix with ones on the diagonal
	U = np.triu(A)  # Upper triangular matrix
	return P, L, U

A1 = np.array([
	[2, 1, 1, 0],
	[4, 3, 3, 1],
	[8, 7, 9, 5],
	[6, 7, 9, 8]
])

A2 = np.array([
	[3, 17, 10],
	[2, 4, -2],
	[6, 18, -12]
], dtype=float)

if debug or "q4" in sys.argv:
	print("Question 4")
	P, L, U = LUWithPartialPivoting(A2.copy())
	print("P = ", P)
	print("L = ", L)
	print("U = ", U)
	print(A2)
	print("PA = ", matrixMultiplication(P, A2), "\n = \n", matrixMultiplication(L, U), " = LU")
	P, L, U = LUWithPartialPivoting(A1.copy())
	print("P = ", P)
	print("L = ", L)
	print("U = ", U)
	print(A1)
	print("PA = ", matrixMultiplication(P, A1), "\n = \n", matrixMultiplication(L, U), " = LU")
	print()

"""P 1.5 Implement the algorithm 4.1.1 LDLT decomposition. Find an example to verify your code"""
def LDLT(A: np.array):
	"""
	Function takes a symettric non-singular matrix A and computes A = LDL^T. Where L is lower triangular and U is upper triangular

	Returns matrix L and D
	"""
	if A.shape[0] != A.shape[1]:
		return "Matrix not square"
	
	if not isSymettric(A):
		return "Matrix is not symetteric"
	v = np.zeros(A.shape[0])
	for j in range(A.shape[0]):
		for i in range(j):
			v[i] = A[j][i] * A[i][i]
		
		for i in range(j):
			A[j][j] = A[j][j] - (A[j][i] * v[i])
		
		# Update the elements below the diagonal for the lower triangular matrix L
		A[j+1:, j] = (A[j+1:, j] - np.dot(A[j+1:, :j], v[:j])) / A[j, j]
	
	L = np.tril(A, -1) + np.eye(A.shape[0])  # Lower triangular matrix with ones on the diagonal
	D = np.diag(A)
	return L, D

A = np.array([
	[4, 12, -16],
	[12, 37, -43],
	[-16, -43, 98]
], dtype=float)

if debug or "q5" in sys.argv:
	print("Question 5")
	L, D = LDLT(A)
	print("L = ", L)
	print("D = ", D)

"""P 1.6 Implement the algorithm 4.2.1 Cholesky Decomposition. And use it to solve the linear systems
6x + 15y + 55z = 76,
15x + 55y + 225z = 295,
55x + 225y + 979z = 1259."""
def CholeskyDecomp(A: np.array):
	"""
	Function takes a symettric positive definite matrix A and computes Cholesky Decompostion of A = GG^T. 

	Returns matrix G
	"""
	if A.shape[0] != A.shape[1]:
		return "Matrix not square"
	
	if not isSymettric(A):
		return "Matrix is not symetteric"
	
	v = np.zeros(A.shape[0])
	G = np.zeros(A.shape)
	for j in range(A.shape[0]):
		for i in range(j, A.shape[0]):
			v[i] = A[i][j]
		for k in range(j):
			for i in range(j, A.shape[0]):
				v[i] = v[i] - (G[j][k] * G[i][k])
		for i in range(j, A.shape[0]):
			G[i][j] = v[i] / np.sqrt(v[j])
	
	return G

A = np.array([
	[4, 1, 2],
	[1, 3, 0],
	[2, 0, 5]
], dtype=float)

if debug or "q6" in sys.argv:
	print("Question 6")
	G = CholeskyDecomp(A.copy())
	
	print("G = ", G)
	print("G^t = ", G.T)
	print("A = ", A, "\n = \n", matrixMultiplication(G, G.T))
	L = np.linalg.cholesky(A.copy())
	print(L)

