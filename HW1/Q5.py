import numpy as np

A = np.array([[0, 0, -1], [4, 1, 1], [-2, 2, 1]])
B = np.array([3, 1, 1])

C = np.array([[0, -2, 6], [-4, -2, -2], [2, 1, 1]])
D = np.array([1, -2, 0])

E = np.array([[2, -2], [-4, 3]])
F = np.array([3, -2])

try:
    x = np.linalg.solve(A, B)
    print(f"Solution a): {x}")
except np.linalg.LinAlgError as e:
    print(f"No unique solution for a)")

try:
    y = np.linalg.solve(C, D)
    print(f"Solution b): {y}")
except np.linalg.LinAlgError as e:
    print(f"No unique solution: Singular Matrix")

try:
    z = np.linalg.solve(E, F)
    print(f"Solution c): {z}")
except np.linalg.LinAlgError as e:
    print(f"No unique solution for c)")