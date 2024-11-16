import numpy as np
import sys
import time
from utils import *
from Assignment1 import *

debug = False
if __name__ == "__main__":
	debug = True
	if len(sys.argv) > 1 and sys.argv[1] == "nodebug":
		debug = False		
		
"""P 1.1 Implement the Algorithm 5.2.1 (Householder QR), and calculate the QR decomposition for
the matrix"""
def house2(x: np.array):
	if(x.shape[0] == 1): return np.zeros(1), 0
	lengthsq = 0
	for i in range(x.shape[0]):
		lengthsq += x[i] ** 2
	beta = np.sqrt(lengthsq)
	e = np.zeros(x.shape[0])
	e[0] = 1
	v = x - beta * e
	lengthv = 0
	# print("beta is: ", beta)
	# print("v = x - beta * e = ", v)
	for i in range(v.shape[0]):
		lengthv += v[i] ** 2
	lengthv = np.sqrt(lengthv)
	u = v / lengthv
	return u, beta

def HouseHolderQR2(A: np.array):
	if A.shape[0] < A.shape[1]: 
		return "Matrix has more columns than rows"
	QT = np.identity(A.shape[0], dtype=float)
	for i in range(A.shape[1]): # For each column we need the householder matrix
		x = A[i:, i]
		# print("x is: ", x, "\ncalling house on x")
		v, beta = house2(x)
		P = np.identity(x.shape[0]) - 2 * multiply_vector_transpose(v)
		H = np.identity(A.shape[0], dtype=float)
		H[i:, i:] = P
		QT = H @ QT
		A = H @ A # A gets over written as R at the end

	return QT.T, A


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

A = np.array([
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1]
], dtype=float)
if __name__ == "__main__" and (debug or "q1" in sys.argv):
    print("Question 1")
    Q, R = HouseHolderQR2(A.copy())
    print("Q is:\n", Q)
    print("R is:\n", R)

"""P 1.2 Implement the Algorithm 5.2.4 (Given QR), and calculate the QR decomposition for the
same matrix"""
def givens_rotation(a, b):
    """Compute the Givens rotation matrix for a and b."""
    r = np.sqrt(a**2 + b**2)
    c = a / r
    s = -b / r
    return c, s

def qr_givens(A):
    """Perform QR factorization using Givens rotations."""
    m, n = A.shape
    R = A.copy()
    Q = np.identity(m)

    for i in range(0, n - 1):
        for j in range(i + 1, m):
            if R[i, j] != 0:
                cos, sin = givens_rotation(R[i, i], R[j, i])
                R[i], R[j] = (R[i] * cos - R[j] * sin), (R[i] * sin + R[j] * cos)
                Q[:, i], Q[:, j] = (Q[:, i] * cos - Q[:, j] * sin), (Q[:, i] * sin + Q[:, j] * cos)

    return Q, R

A = np.array([
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1]
], dtype=float)
if __name__ == "__main__" and (debug or "q2" in sys.argv):
    print("Question 2")
    Q, R = qr_givens(A)
    print("Q is:\n", Q)
    print("R is:\n", R)

"""P 1.3 Implement the Algorithm 5.3.1 (Normal Equations) to solve the least square problem. And
find an example to verify your code."""
def normal_equations(A: np.array, b: np.array):
    """Solve the least squares problem using normal equations."""
    if A.shape[1] != np.linalg.matrix_rank(A):
        return "Matrix is not full rank"
    
    C = A.T @ A
    d = A.T @ b

    G = CholeskyDecomp(C)
    y = columnOrientedForwardSub(G, d)
    x = columnOrienteBackwardSub(G.T, y)
    return x

A1 = np.array([
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1]
], dtype=float)
A2 = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
], dtype=float)
b = np.array([1, 2, 3], dtype=float)

if __name__ == "__main__" and (debug or "q3" in sys.argv):
    print("Question 3")
    x = normal_equations(A1.copy(), b) 
    print("My x_ls is:\n", x)
    y = np.linalg.lstsq(A1.copy(), b, rcond=None)[0]
    print("np.lingal.lstsq() is:\n", y)
    x = normal_equations(A2.copy(), b) 
    print("My x_ls is:\n", x)
    y = np.linalg.lstsq(A2.copy(), b, rcond=None)[0]
    print("np.lingal.lstsq() is:\n", y)

"""P 1.4 Implement the Algorithm 5.3.2 (Householder LS Solution). And find an example to verify
your code."""
def HouseHolderLS(A: np.array, b: np.array):
    """Solve the least squares problem using Householder QR.
	A = QR, Ax = QRx = b, Rx = Q^Tb
    """
    if A.shape[1] != np.linalg.matrix_rank(A):
        return "Matrix is not full rank"
    Q, R = HouseHolderQR2(A)
    
    y = Q.T @ b
    x = np.linalg.inv(R) @ y
    
    return x

if __name__ == "__main__" and (debug or "q4" in sys.argv):
	print("Question 4")
	x = HouseHolderLS(A1.copy(), b)
	print("My x_ls is:\n", x)
	y = np.linalg.lstsq(A1.copy(), b, rcond=None)[0]
	print("np.lingal.lstsq() is:\n", y)
	print("\n\n\nComparing HouseHolderLS with normal_equations")
	start = time.time()
	x = normal_equations(A1.copy(), b)
	print("It took: ", time.time() - start, "For normal_equations")
	start = time.time()
	x = HouseHolderLS(A1.copy(), b)
	print("It took: ", time.time() - start, "For HouseHolderLS")
	




