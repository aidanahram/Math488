import numpy as np

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

def isSymettric(A: np.array):
	if A.shape[0] != A.shape[1]: 
		return False
	for i in range(A.shape[0]):
		for j in range(A.shape[0]):
			if A[i][j] != A[j][i]:
				return False
	return True

def multiply_vector_transpose(v1):
    return np.outer(v1, v1)