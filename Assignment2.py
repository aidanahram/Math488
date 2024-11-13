import numpy as np
import sys

if __name__ == "__main__":
	debug = True
	if len(sys.argv) > 1 and sys.argv[1] == "nodebug":
		debug = False		
		
"""P 1.1 Implement the Algorithm 5.2.1 (Householder QR), and calculate the QR decomposition for
the matrix"""
def HouseHolderQR(A: np.array):
	if A.shape[0] < A.shape[1]: 
		return "Rows are less than columns"
	return

if debug or "q1" in sys.argv:
	print("Question 1")