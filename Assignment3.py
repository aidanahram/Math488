import numpy as np
import sys
from utils import *
from Assignment1 import * 
from Assignment2 import *

debug = False
if __name__ == "__main__":
	iterations = 100
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
	q = powerMethod(A.copy(), q, iterations)
	print(f"After {iterations} q is:\n {q}")
	q = powerMethod(A2.copy(), q2, iterations)
	print(f"After {iterations} q is:\n {q}")

"""P 1.2 Implement the Orthogonal Iteration in 7.3.2. See 8.2.4 as well. And use an example to verify
your code can find the first few largest eigenvalues and their corresponding eigenvector subspace."""
def orthogonalIteration(A: np.array, maxIter: int):
	n = A.shape[0]
	Q = np.identity(n)
	for i in range(maxIter):
		Z = matrixMultiplication(A, Q)
		Q, R = np.linalg.qr(Z)
	return Q

if __name__ == "__main__" and (debug or "q2" in sys.argv):
	print("Question 2")
	q = orthogonalIteration(A.copy(), iterations)
	print(f"After {iterations} q is:\n {q}")
	q = orthogonalIteration(A2.copy(), iterations)
	print(f"After {iterations} q is:\n {q}")

"""P 1.3 Implement the QR iteration method in equation (7.3.1). And show an example to verify
your code."""
def QRIteration(A: np.array, maxIter: int):
	for i in range(maxIter):
		Q, R = np.linalg.qr(A)
		A = R @ Q
	return A

if __name__ == "__main__" and (debug or "q3" in sys.argv):
	print("Question 3")
	q = QRIteration(A.copy(), iterations)
	print(f"After {iterations} q is:\n {q}")
	q = QRIteration(A2.copy(), iterations)
	print(f"After {iterations} q is:\n {q}")

"""P 1.4 Implement the Algorithm 7.4.2 (Householder Reduction to Hesenberg Form). And find an
example to verify your code"""
def householderReduction(A: np.array):
	m, n = A.shape
	for k in range(n - 2):
		v, beta = house(A[k + 1:, k])
		A[k + 1:, k:] = A[k + 1:, k:] - beta * np.outer(v, v) @ A[k + 1:, k:]
		A[:, k + 1:] = A[:, k + 1:] - beta * A[:, k + 1:] @ np.outer(v, v)
	return A

if __name__ == "__main__" and (debug or "q4" in sys.argv):
	print("Question 4")
	print("Householder Reduction to Hessenberg Form") 
	new = householderReduction(A.copy())
	print(new)
	new = householderReduction(A2.copy())
	print(new)

"""P 1.5 Implement the Algorithm 8.3.1 (Houserholder Tridiagonalization). And find an example to
verify your code."""
def householderTridiagonalization(A: np.array):
	n = A.shape[0]
	for k in range(n - 2):
		v, beta = house(A[k + 1:, k])
		p = beta * A[k + 1:, k + 1:] @ v
		w = p - (beta * np.dot(p, v) / 2) * v
		A[k + 1, k] = np.linalg.norm(A[k + 1:, k])
		A[k, k + 1] = A[k + 1, k]
		A[k + 1:, k + 1:] = A[k + 1:, k + 1:] - np.outer(v, w) - np.outer(w, v)
	return A

if __name__ == "__main__" and (debug or "q5" in sys.argv):
	print("Question 5")
	print("Houserholder Tridiagonalization") 
	new = householderTridiagonalization(A.copy())
	print(new)
	new = householderTridiagonalization(A2.copy())
	print(new)

"""P 1.6 Implement the classical 3-step SVD algorithm on the bottom of Page 488 (Section 8.6.3).
And find an example to verify your code. A better SVD algorithm is shown in later of Section 8.6.3,
but it is not required."""
def classicalSVD(A: np.array):
    m, n = A.shape
    assert m >= n, "m must be greater than or equal to n"

    # Step 1: Form C = A^T A
    C = A.T @ A

    # Step 2: Use the symmetric QR algorithm to compute V1
    # For simplicity, we use numpy's eigh function to compute the eigenvalues and eigenvectors
    sigma_squared, V1 = np.linalg.eigh(C)

    # Step 3: Apply QR with column pivoting to AV1
    AV1 = A @ V1
    Q, R = np.linalg.qr(AV1, mode='reduced')

    # Compute U and V
    U = Q
    V = V1

    # Singular values are the square roots of the eigenvalues
    singular_values = np.sqrt(sigma_squared)

    return U, singular_values, V

if __name__ == "__main__" and (debug or "q6" in sys.argv):
    print("Question 6")
    print("Classical SVD Algorithm")
    A = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=float)
    U, singular_values, V = classicalSVD(A)
    print("U:\n", U)
    print("Singular values:\n", singular_values)
    print("V:\n", V)