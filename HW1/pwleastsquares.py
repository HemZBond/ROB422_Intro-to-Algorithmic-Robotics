import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('calibration.txt')
commanded_positions = data[:, 0]
measured_positions = data[:, 1]

inside_indices = np.abs(commanded_positions) <= 0.5
outside_indices = ~inside_indices

A1 = np.vstack([commanded_positions[inside_indices], np.ones(np.sum(inside_indices))]).T
A2 = np.vstack([commanded_positions[outside_indices], np.ones(np.sum(outside_indices))]).T

b1 = measured_positions[inside_indices]
b2 = measured_positions[outside_indices]

theta1 = np.linalg.pinv(A1) @ b1 
theta2 = np.linalg.pinv(A2) @ b2 

m1, c1 = theta1
m2, c2 = theta2

predicted_positions_inside = A1 @ theta1
predicted_positions_outside = A2 @ theta2

predicted_positions = np.zeros_like(commanded_positions)
predicted_positions[inside_indices] = predicted_positions_inside
predicted_positions[outside_indices] = predicted_positions_outside

plt.scatter(commanded_positions, measured_positions, color='blue', label='Data points')
plt.plot(commanded_positions[inside_indices], predicted_positions_inside, color='red', label='Fitted line (inside)')
plt.plot(commanded_positions[outside_indices], predicted_positions_outside, color='green', label='Fitted line (outside)')
plt.xlabel('Commanded Position')
plt.ylabel('Measured Position')
plt.title('Piece-wise Least-Squares Fit')
plt.legend()
plt.show()

predicted_068 = np.array([0.68, 1]) @ theta2 
print(f"Predicted value for commanded position 0.68: {predicted_068}")
