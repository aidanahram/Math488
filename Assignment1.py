import numpy as np

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
# matrix = np.array([[2, 0, 0], 
# 				   [1, 5, 0],
# 				   [7, 9, 8]])
# b = np.array([6.0, 2.0, 5.0])
# print("Question 1")
# print(columnOrientedForwardSub(matrix, b))


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

# matrix = np.array([
# 	[2.0, 1.0, -1.0], 
# 	[0.0, 3.0, 2.0],
# 	[0.0, 0.0, 4.0]
# ])
# b = np.array([5.0, 11.0, 12.0])
# print("Question 2")
# print(columnOrienteBackwardSub(matrix, b)) 
# print("Expected Result is [19/6, 5/3, 3.0]")
# matrix = np.array([
# 	[3, -2, 4],
# 	[0, 1, 5],
# 	[0, 0, 6]
# ])
# b = np.array([10.0, 7.0, 12.0])
# print(columnOrienteBackwardSub(matrix, b)) 
# print("Expected Result is [-4/3, -3.0, 2.0]")


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

L, U = outerProductLU(A)
print("Question #3")
print("L = ", L)
print("U = ", U)



