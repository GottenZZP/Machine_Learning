import numpy as np

a = np.array([[3, 4], [2, 16]])
# print(np.linalg.inv(a))

A = np.matrix(a)
print(A)
print(A.I)
# print(A * A.I)
print(A.T)
