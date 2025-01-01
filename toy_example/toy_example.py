import cv2
import numpy as np
import matplotlib.pyplot as plt
from toy_utils import generate_3d_points, intrinsic_matrix, project_points, plot_trajectory, visualize_image,plot_trajectory_and_image

# Step 1: Define 3D points in the initial camera (world) frame
num_3d_pts = 16
xy_var = 0.7
z_base = 5
z_range = (0.8, 1.2)
points_3d = generate_3d_points(num_3d_pts, xy_var, z_base, z_range)

# Step 2: Define the intrinsic camera matri x (realistic values for a camera)
focal_length = 800
cx, cy = 320, 240
K = intrinsic_matrix(focal_length, cx, cy)

# Step 3: Define initial extrinsics (identity for initial pose)
R1 = np.eye(3)
t1 = np.zeros((3, 1))
extrinsics1 = np.hstack((R1, t1))
P1 = K @ extrinsics1

# Step 4: Project points to the first image
points_2d_last = project_points(points_3d, P1)

# Create first image
image1 = np.zeros((480, 640), dtype=np.uint8)
prev_image = image1.copy()
for pt in points_2d_last.astype(int):
    cv2.circle(image1, tuple(pt), radius=3, color=255, thickness=-1)
#visualize_image(image1, title="Initial Image")

# Iterative camera movement and rotation
num_iterations = 4
t_gt_WC = np.zeros((3, 1))
R_gt_CW = np.eye(3)

camera_positions_gt = [t_gt_WC.flatten()]
camera_orientations_gt = [R_gt_CW]
camera_positions_est = [t_gt_WC.flatten()]
camera_orientation_est = [R_gt_CW]
R_est_CW_accum = None
t_est_WC_accum = None

for i in range(num_iterations):
    print(f"Iteration {i + 1}/{num_iterations}")
    print("========================================")
    # Move the camera forward with random x, y, z translation
    xy_var = 0.05
    t_gt_Clast_C = np.array([
        [np.random.uniform(-xy_var, xy_var)],  # x translation
        [0],                                     # fixed y translation
        [0.1]                            # fixed z translation
    ])
    print("t_gt_Clast_C",t_gt_Clast_C)
    t_gt_WC += t_gt_Clast_C
    camera_positions_gt.append(t_gt_WC.flatten())
    camera_orientations_gt.append(R_gt_CW)

    # Rotate the camera around the y-axis
    angle = np.radians(np.random.uniform(0, 5))
    R_y = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])
    R_gt_CW = R_y @ R_gt_CW

    # Compute the new projection matrix
    extrinsics = np.hstack((R_gt_CW, t_gt_WC))
    P = K @ extrinsics

    # Project points to the image
    points_2d_current = project_points(points_3d, P)

    # Check if all points are within the image bounds
    within_bounds = np.all((points_2d_current[:, 0] >= 0) & (points_2d_current[:, 0] < 640) &
                           (points_2d_current[:, 1] >= 0) & (points_2d_current[:, 1] < 480))
    if not within_bounds:
        print(f"Warning: Some points are out of bounds in iteration {i + 1}.")

    # Find essential matrix
    #  Filter with RANSAC
    F, mask_RANSAC = cv2.findFundamentalMat(points_2d_last, points_2d_current, cv2.FM_RANSAC, ransacReprojThreshold=1.0, confidence=0.99)

    # Use the mask to filter inliers
    pts0_inliers = points_2d_last[mask_RANSAC.ravel() == 1]
    pts1_inliers = points_2d_current[mask_RANSAC.ravel() == 1]

    E, mask = cv2.findEssentialMat(pts0_inliers, pts1_inliers, K, method=cv2.FM_8POINT)

    # Recover pose
    _, R_est_C_Clast, t_est_Clast_C, mask_pose = cv2.recoverPose(E, pts0_inliers, pts1_inliers, K)

    # T_C_Clast = np.zeros((4, 4))
    # T_C_Clast[:3, :3] = R_est_Clast_C
    # T_C_Clast[:3, 3] = t_est_C_Clast.flatten()
    # T_C_Clast[3, 3] = 1

    # T_Clast_C = np.linalg.inv(T_C_Clast)
    # R_est_C_Clast = T_Clast_C[:3, :3]
    # t_est_Clast_C = T_Clast_C[:3, 3].reshape(-1, 1)


    print("t_est_Clast_C",t_est_Clast_C)
    
    # Estimate trajectory (relative to the initial pose)
    if i == 0:
        R_est_CW_accum = R_est_C_Clast
        t_est_WC_accum = t_est_Clast_C
    else:
        R_est_CW_accum = R_est_C_Clast @ R_est_CW_accum
        t_est_WC_accum = R_est_C_Clast @ t_est_WC_accum + t_est_Clast_C

    camera_positions_est.append(t_est_WC_accum.flatten())
    camera_orientation_est.append(R_est_CW_accum)

    # Create the image for this iteration
    image = np.zeros((480, 640), dtype=np.uint8)
    for pt in points_2d_current.astype(int):
        cv2.circle(image, tuple(pt), radius=3, color=255, thickness=-1)

    #plot_trajectory_and_image(camera_positions_gt, camera_positions_est, points_3d,prev_image, image, i + 1)
    #plot_trajectory_and_image(camera_positions_gt, camera_positions_est,camera_orientations_gt, camera_orientation_est, points_3d,image)
    #visualize_image(image, title=f"Iteration {i + 1} Image")

    # Update the last image points
    points_2d_last = points_2d_current
    prev_image = image.copy()   
    print()

# Plot ground truth and estimated camera positions in the world frame (x-z plane)

plot_trajectory_and_image(camera_positions_gt, camera_positions_est,camera_orientations_gt, camera_orientation_est, points_3d,image)
