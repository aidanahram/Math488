import numpy as np

"""P 1.1 Implement the Algorithm 3.1.3 column-oriented forward substitution. Find an example to
verify your code."""
def columnOrientedForwardSub(L: np.array, b: np.array):
	"""
	This function takes a lower triangular matrix L and a vector matrix b and solves the equation Ax = b.

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

matrix = np.array([[2, 0, 0], 
				   [1, 5, 0],
				   [7, 9, 8]])
b = np.array([6.0, 2.0, 5.0])
res = columnOrientedForwardSub(matrix, b)
print(res)
