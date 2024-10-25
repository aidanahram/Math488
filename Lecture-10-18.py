import numpy as np

# Forward Substitution
def forward_substitution(L, b):
	m, n = L.shape
	#x = np.zeros_like(b)
	b[0] = b[0]/L[0,0]

	for i in range(1,m):
		b[i] = (b[i] - np.dot(L[i,0:i],b[0:i]))/L[i,i]
	return b

A = np.array([[1,0,0],[2,3,0],[4,5,6]])
b = np.array([1, 5, 15])
x = forward_substitution(A,b)
print('Forward Substitution, solution is:', x)


# Backward Substitution
def backward_substitution(U, b):
	m, n = U.shape
	#b[0],b[1],b[2], ...., b[m-1]
	b[m-1] = b[m-1]/U[m-1,m-1]
	for i in range(m-2,-1,-1):
		b[i] = (b[i] - np.dot(U[i,i+1:m], b[i+1:m]))/U[i,i]
	return b

A = np.array([[1,2,3],[0,4,5],[0,0,6]])
b = np.array([6,9,6])
x = backward_substitution(A,b)
print('Backward Substitution solution is:', x)


# LU Decomposition
def Gaussian_elimination_without_pivoting(A):
	# return L, U
	m, n = A.shape
	U = np.array(A)
	L = np.eye(m)

	for k in range(0,m-1):
		for j in range(k+1,m):
			L[j,k] = U[j,k]/U[k,k]
			U[j,k:m] = U[j,k:m] - L[j,k] * U[k,k:m]
	return L, U

A = np.array([[1,4,7],[2,5,8],[3,6,10]])
L, U = Gaussian_elimination_without_pivoting(A)
print('LU Decomposition')
print('L',L)
print('U',U)



# Use LU Decomposition to solve Ax = b

A = np.array([[1,4,7],[2,5,8],[3,6,10]])
b = np.array([12, 15, 19])

L, U = Gaussian_elimination_without_pivoting(A)
y = forward_substitution(L,b)
x = backward_substitution(U,y)
print('The solution is', x)


