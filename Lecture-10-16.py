import numpy as np

# Reivew of python

# Matrix addition
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

A + B
A - B
A * B  # A.*B   A%*%B
A / B
A@B # Matrix multiplication

#A, B,  A+B
def matrix_addition(A, B):
    size_A = A.shape
    size_B = B.shape

    if size_A != size_B:
        return "Matrix addition is not possible"
    size = size_A
    C = np.zeros(size)
    for i in range(size[0]):
        for j in range(size[1]):
            C = A[i][j] + B[i][j]
    return C

def matrix_multiplication(A, B):
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


# Chapter 1: Basic
# Algorithm 1.1: dot product of two vectors
def dot_product(x, y):
    length_x = len(x)
    length_y = len(y)
    if length_x != length_y:
        return "Dot product is not possible"
    #assert length_x == length_y, "Dot product is not possible"
    
    c = 0 # place holder
    for i in range(length_x):  # for i=1:length(x)
        c += x[i] * y[i]
    return c

x = np.array([1, 2, 3])
y = np.array([1, 1, 1])
x_dot_y = dot_product(x, y)
x_dot_y_v2 = np.dot(x, y)
print('x_dot_y:', x_dot_y)



# Algorithm 1.2
def saxpy(a, x, y):
    # y = y + a*x
    len_x = x.size #length(x)
    len_y = y.size #length(y)
    if len_x != len_y:
        return "Saxpy is not possible"

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


def row_oriented_gaxpy(A, x, y):
    # y = y + Ax
    m, n = A.shape
    for i in range(m): #i=0,1,2,..., m-1
        for j in range(n): #j=0,1,2, ..., n-1
            y[i] = y[i] + A[i,j] * x[j]
    return y


def column_oriented_gaxpy(A, x, y):
    # y = y + Ax
    for j in range(n):
        for i in range(m):
            y[i] = y[i] + A[i,j] * x[j]
    return y


# Chapter 2: Norm

x = np.array([1,-2,-3])

def norm_1(x):
    len_x = len(x)
    result = 0
    for i in range(len_x):
        result += abs(x[i])
    return result

def norm_1_v2(x):
    return sum(abs(x))


print('1-norm of x:', norm_1(x))
print('1-norm of x:', norm_1_v2(x))
print('1-norm of x:', np.linalg.norm(x, 1))



def norm_2(x):
    len_x = len(x)
    result = 0
    for i in range(len_x):
        result += x[i]**2
    return result**(1/2)

def norm_2_v2(x):
    return np.sqrt(sum(x**2))
    #return sum(x**2)**(1/2)

x = np.array([1,2,3])
print('2-norm of x:', norm_2(x))
print('2-norm of x:', norm_2_v2(x))
print('2-norm of x:', np.linalg.norm(x, 2))


# Matrix Norm
# calculate the 1-norm of a matrix

def norm_1(A):
    m, n = A.shape
    result = np.zeros(n)
    for j in range(n):  # column
        for i in range(m):  # row
            result[j] += abs(A[i,j])
    return max(result)

def nrom_1_v2(A):
    return max(np.sum(abs(A), axis=0))


A = np.array([[1, -2, 3], [-4, 5, -6]])
print('The 1 norm of A is', norm_1(A))
print('The 1 norm of A is', nrom_1_v2(A))
print('The 1 norm of A is', np.linalg.norm(A, 2))

norm_A_F = np.linalg.norm(A, 'fro')


# calculate the 1-norm of a matrix
# def norm_1_v2(A):
#     return np.max(np.sum(np.abs(A), axis=0))

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




