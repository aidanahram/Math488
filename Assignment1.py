import numpy as np

"""P 1.1 Implement the Algorithm 3.1.3 column-oriented forward substitution. Find an example to
verify your code."""
def columnOrientedForwardSub(L: np.array, b: np.array):
	"""
	This function takes a lower triangular matrix L and a vector matrix b and solves the equation Lx = b.

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
# res = columnOrientedForwardSub(matrix, b)
# print(res)


"""P 1.2 Implement the Algorithm 3.1.4 column-oriented back substitution. Find an example to verify
your code."""
def columnOrienteBackwardSub(U: np.array, b: np.array):
	"""
	This function takes a non-singular upper triangular matrix U and a vector matrix b and solves the equation Ux = b.

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