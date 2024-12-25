import numpy as np
A = np.array([[1,2], [3,-1]])
B = np.array([[-2,-2], [4,-3]])

print("a. A+2B =  \n", A + 2*B)
print("b. AB = \n", np.linalg.matmul(A,B))
print("b. BA = \n", np.linalg.matmul(B,A))
print("c. A transponse = \n", A.transpose())
print("d. B^2 = \n", np.linalg.matmul(B,B))
print("e. A transponse x B transponse = \n", np.linalg.matmul(A.transpose(), B.transpose()))
print("e. (A x B) transponse = \n", np.linalg.matmul(A,B).transpose())
print("f. det(A) = \n", np.linalg.det(A))
print("g. inverse of B = \n", np.linalg.inv(B))