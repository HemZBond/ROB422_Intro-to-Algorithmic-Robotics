import numpy as np

def rotation_z(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

def rotation_y(theta):
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

t1 = np.pi / 2    
t2 = -np.pi / 5    
t3 = np.pi       
R_zaxis1 = rotation_z(t1)  
R_yaxis2 = rotation_y(t2)  
R_zaxis3 = rotation_z(t3) 

R = np.dot(R_zaxis3, np.dot(R_yaxis2, R_zaxis1))

print("Final Rotation Matrix R:")
print(R)
