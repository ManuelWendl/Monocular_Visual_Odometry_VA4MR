import cv2
import numpy as np

# Define camera intrinsic matrix (simplified for demo)
K = np.array([
    [800, 0, 320],  # fx, 0, cx
    [0, 800, 240],  # 0, fy, cy
    [0,   0,   1]   # 0,  0,  1
])

# Simulate 3D points in the world frame
points_3D = np.array([
    [1, 1, 5],
    [2, -1, 6],
    [-1, -2, 7],
    [3, 0, 8],
    [0, 1, 9],
    [1, 0, 10]
])

# Define two camera poses (world to camera transformations)
R1 = np.eye(3)  # First camera at the origin
t1 = np.zeros((3, 1))

R2 = cv2.Rodrigues(np.array([0.1, 0.2, 0.3]))[0]  # Rotation for second camera
t2 = np.array([[0], [0], [1]])  # Translation for second camera

# Project points into the first and second camera views
P1 = K @ np.hstack((R1, t1))
P2 = K @ np.hstack((R2, t2))

points_2D_1 = P1 @ np.vstack((points_3D.T, np.ones(points_3D.shape[0])))
points_2D_2 = P2 @ np.vstack((points_3D.T, np.ones(points_3D.shape[0])))

# Normalize homogeneous coordinates
points_2D_1 /= points_2D_1[2, :]
points_2D_2 /= points_2D_2[2, :]

# Compute the Essential matrix
E, _ = cv2.findEssentialMat(points_2D_1[:2].T, points_2D_2[:2].T, K)

# Recover relative pose from the Essential matrix
points_2D_1 = points_2D_1[:2].T
points_2D_2 = points_2D_2[:2].T
_, R, t, mask = cv2.recoverPose(E, points_2D_1, points_2D_2, K)

# Print the results
print("Ground Truth R:\n", R2)
print("\nRecovered R:\n", R)

print("\nGround Truth t (unit vector):\n", t2.T / np.linalg.norm(t2))
print("\nRecovered t:\n", t.T)

# Verify that the recovered pose aligns well with the ground truth
