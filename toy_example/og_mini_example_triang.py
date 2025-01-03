import numpy as np
import cv2



# Define two 3D points in the world frame
points_3D_world = np.array([
    [1.0, 2.0, 10.0],
    [-1.0, -1.5, 8.0],
    [2.0, 1.5, 12.0]
]).T  # Shape (3, N)

# Projection matrices for two cameras
# First camera is at the origin
P1 = np.hstack((np.eye(3), np.zeros((3, 1))))  # [I | 0]

# Second camera has a slight translation and rotation
R = cv2.Rodrigues(np.array([0.0, -0.2, 0.1]))[0]  # Small rotation
t = np.array([[0.5], [0.0], [1.0]])  # Translation along X-axis
P2 = np.hstack((R, t))  # [R | t]

# Intrinsic matrix (identity for simplicity)
K = np.array([[7.1885e+02, 0, 6.07192e+02],
            [0, 7.18856e+02, 1.85215e+02],
            [0, 0, 1]])

# Compute projection matrices
P1_full = K @ P1
P2_full = K @ P2

# Project the 3D points into the two image planes
points_2D_1 = P1_full @ np.vstack((points_3D_world, np.ones((1, points_3D_world.shape[1]))))
points_2D_2 = P2_full @ np.vstack((points_3D_world, np.ones((1, points_3D_world.shape[1]))))

# Normalize homogeneous coordinates
points_2D_1 /= points_2D_1[2, :]
points_2D_2 /= points_2D_2[2, :]

# Extract first two rows
points_2D_1 = points_2D_1[:2]
points_2D_2 = points_2D_2[:2]

# Triangulate points from the two views
triangulated_points_homogeneous = cv2.triangulatePoints(P1_full, P2_full, points_2D_1, points_2D_2)

# Convert to inhomogeneous coordinates
triangulated_points = triangulated_points_homogeneous[:3, :] / triangulated_points_homogeneous[3, :]

# Print results
print("Original 3D points (world frame):")
print(points_3D_world.T)
print("\nTriangulated 3D points:")
print(triangulated_points.T)
