import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('calibration.txt')
commanded_positions = data[:, 0]
measured_positions = data[:, 1]

A = np.vstack([commanded_positions, np.ones(len(commanded_positions))]).T
b = measured_positions

theta = np.linalg.pinv(A) @ b
m, c = theta

predicted_positions = A @ theta

plt.scatter(commanded_positions, measured_positions, color='blue', label='Data points')
plt.plot(commanded_positions, predicted_positions, color='red', label=f'Fitted line: y = {m:.2f}x + {c:.2f}')
plt.xlabel('Commanded Position')
plt.ylabel('Measured Position')
plt.title('Least-Squares Fit')
plt.legend()
plt.show()

print(f"Slope (m): {m}")
print(f"Intercept (c): {c}")