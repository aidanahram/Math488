import numpy as np

# Matrix addition
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

A + B
A - B
A * B
A / B
A@B # Matrix multiplication

def matrix_addition(A, B):
    size = A.shape
    C = np.zeros(size)
    for i in range(size[0]):
        for j in range(size[1]):
            C = A[i][j] + B[i][j]
    return C

def matrix_multiplication(A, B):
    size_A = A.shape  # [2, 2]  size_A[0] , size_A[1]
    size_B = B.shape
    if size_A[1] != size_B[0]:
        return "Matrix multiplication is not possible"
    C = np.zeros((size_A[0], size_B[1]))
    for i in range(size_A[0]):
        for j in range(size_B[1]):
            for k in range(size_A[1]):
                C[i][j] += A[i][k] * B[k][j]
    return C


# Algorithm 1.1: dot product of two vectors
def dot_product(x, y):
    length_x = len(x)
    length_y = len(y)
    if length_x != length_y:
        return "Dot product is not possible"
    #assert length_x == length_y, "Dot product is not possible"
    c = 0
    for i in range(length_x):  # for i=1:length(x)
        c += x[i] * y[i]
    return c

x = np.array([1, 2, 3])
y = np.array([1, 1, 1])
x_dot_y = dot_product(x, y)
print('x_dot_y:', x_dot_y)

# Algorithm 1.2
def saxpy(a, x, y):
    """
    Perform the operation y = y + a * x

    Args:
    a (float): Scalar value.
    x (np.ndarray): Vector of length n.
    y (np.ndarray): Vector of length n, to be updated.

    Returns:
    np.ndarray: Updated vector y.
    """
    n = len(x)
    for i in range(n):
        y[i] = y[i] + a * x[i]
    return y


# Example usage
a = 1.0  # Scalar value
x = np.array([1.0, 2.0, 3.0])  # Example vector x
y = np.array([4.0, 5.0, 6.0])  # Example vector y

# Perform SAXPY
result = saxpy(a, x, y)
print("Updated y:", result)
# Algorithm 1.1.3 (Row-Oriented Gaxpy)

import numpy as np

A = np.array([[1,2],[3,4], [5,6]])
x = np.array([7,8])
y = np.array([0,0])


def columnn_oriented_gaxpy(A, x, y):
    # Ax + y
    m, n = A.shape
    for i in range(m): #i=0,1,2,..., m-1
        for j in range(n): #j=0,1,2, ..., n-1
            y[i] = y[i] + A[i,j] * x[j]
    return y



x = np.array([1,-2,-3])

def norm_1(x):
    len_x = len(x)
    result = 0
    for i in range(len_x):
        result += abs(x[i])
    return result

print('1-norm of x:', norm_1(x))

def norm_2(x):
    len_x = len(x)
    result = 0
    for i in range(len_x):
        result += x[i]**2
    return result**(1/2)


# calculate the 1-norm of a matrix
def norm_1(A):
    size = A.shape
    result = 0
    for i in range(size[0]):
        for j in range(size[1]):
            result += abs(A[i][j])
    return result

# calculate the 1-norm of a matrix
def norm_1_v2(A):
    return np.max(np.sum(np.abs(A), axis=0))

# calculate the 1-norm of a matrix
def norm_1_v3(A):
    size = A.shape
    column_sums = [0] * size[1]
    for j in range(size[1]):
        for i in range(size[0]):
            column_sums[j] += abs(A[i][j])
    return max(column_sums)

# Example usage
A = np.array([[1, -2, 3], [-4, 5, -6]])
print('1-norm of A:', norm_1_v3(A))