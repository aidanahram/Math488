import numpy as np
import sys


if __name__ == "__main__":
	debug = True
	if len(sys.argv) > 1 and sys.argv[1] == "nodebug":
		debug = False

### Helpful Functions
def matrixMultiplication(A, B):
    size_A = A.shape  # [2, 2]  size_A[0] , size_A[1]
    size_B = B.shape
    if size_A[1] != size_B[0]:  #column of A  == Row of B
        return "Matrix multiplication is not possible"

    C = np.zeros((size_A[0], size_B[1]))
    for i in range(size_A[0]):
        for j in range(size_B[1]):
            for k in range(size_A[1]):
                C[i][j] += A[i][k] * B[k][j]
    return C

def swapRows(A: np.array, index1: int , index2: int):
	temp = A[index2].copy()
	A[index2] = A[index1]
	A[index1] = temp
	return A

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
if debug:
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
if debug:
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

if debug:
	L, U = outerProductLU(A)
	print("Question #3")
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

if debug:
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
