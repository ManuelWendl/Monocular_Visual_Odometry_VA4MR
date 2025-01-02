import cv2
import numpy as np
import matplotlib.pyplot as plt
from toy_utils import generate_3d_points, intrinsic_matrix, project_points, \
    plot_trajectory, get_viz_img,plot_trajectory_and_image,move_camera,get_trans_img

# Step 1: Define 3D points in the initial camera (world) frame
num_3d_pts = 25
xy_var = 0.7
z_base = 4
z_range = (-0.25, 0.25)
points_3D_W = generate_3d_points(num_3d_pts, xy_var, z_base, z_range) # 3xN

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
points_2d_last = project_points(points_3D_W, P1)

# Create first image
image1 = np.zeros((480, 640), dtype=np.uint8)
prev_image = get_viz_img(points_2d_last, title="Initial Image")

# Iterative camera movement and rotation
num_iterations = 10
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
    R_gt_CW, t_gt_WC = move_camera(t_gt_WC,R_gt_CW)
    extrinsics = np.hstack((R_gt_CW, t_gt_WC))
    P = K @ extrinsics

    # Project points to the image
    points_2d_current = project_points(points_3D_W, P)
    curr_image = get_viz_img(points_2d_current, title=f"Iteration {i + 1} Image")
    
    # Use Pnp to estimate the camera pose --> Conclusion: solvePnPRansac returns "3dp to current camera frame"
    sucess, R_est_CW, t_est_W_C, inliers = cv2.solvePnPRansac(points_3D_W.T, points_2d_current.T, K, np.zeros(4), flags=cv2.SOLVEPNP_ITERATIVE, confidence=0.999 ,reprojectionError=2)
    R_est_CW, _ = cv2.Rodrigues(R_est_CW)


    # Triangulate points
    if i > 0:
        R_est_W_Clast = np.array(R_est_CW_accum).T
        R_est_C_Clast = R_est_CW @ R_est_W_Clast
        t_est_W_Clast = t_est_WC_accum
        t_est_Clast_C = t_est_W_C - np.array(t_est_WC_accum)
        P0 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P1 = K @ np.hstack((R_est_C_Clast, t_est_Clast_C))
        points_3D_triangulated_Clast = cv2.triangulatePoints(P0, P1, points_2d_last, points_2d_current)
        points_3D_triangulated_Clast /= points_3D_triangulated_Clast[3, :]
        points_3D_triangulated_Clast = points_3D_triangulated_Clast[:3, :].reshape(3, -1)

        # Filter out points behind the camera
        max_depth = points_3D_triangulated_Clast[2, :].mean()
        z_mask = (0 < points_3D_triangulated_Clast[2, :]) & (points_3D_triangulated_Clast[2, :] < max_depth)
        points_3D_triangulated_Clast_invalid = points_3D_triangulated_Clast[:, ~z_mask]
        points_3D_triangulated_Clast = points_3D_triangulated_Clast[:, z_mask]

        # Transform the triangulated points to the world frame
        points_3D_triangulated_W = R_est_W_Clast @ points_3D_triangulated_Clast + t_est_W_Clast
        points_3D_triangulated_W_invalid = R_est_W_Clast @ points_3D_triangulated_Clast_invalid + t_est_W_Clast
        
        #distances = np.linalg.norm(points_3D_triangulated_W.T - points_3d, axis=0)
        #print("Mean triangulation error:", np.mean(distances))

        trans_error = np.linalg.norm(t_gt_WC - t_est_W_C)
        print("Translation error:", trans_error)
        rot_error = cv2.Rodrigues(R_gt_CW @ R_est_CW.T)[0].sum()
        print("Rotation error:", np.degrees(rot_error))

            
    # Estimate trajectory (relative to the initial pose)
    if i == 0:
        R_est_CW_accum = R_est_CW
        t_est_WC_accum = t_est_W_C
    else:
        R_est_CW_accum = R_est_CW
        t_est_WC_accum = t_est_W_C

    # Create the image for this iteration
    trans_img = get_trans_img(points_2d_last, points_2d_current)

    #plot_trajectory_and_image(camera_positions_gt, camera_positions_est, points_3d,prev_image, image, i + 1)
    if i > 0:
        plot_trajectory_and_image(camera_positions_gt, camera_positions_est,camera_orientations_gt,\
                                   camera_orientation_est, points_3D_W,points_3D_triangulated_W, \
                                    points_3D_triangulated_W_invalid, trans_img)
        
    #visualize_image(image, title=f"Iteration {i + 1} Image")

    # Update vars
    points_2d_last = points_2d_current
    prev_image = curr_image 

    camera_positions_gt.append(t_gt_WC.flatten())
    camera_orientations_gt.append(R_gt_CW)
    camera_positions_est.append(t_est_WC_accum.flatten())
    camera_orientation_est.append(R_est_CW_accum)
    
    print()

# Plot ground truth and estimated camera positions in the world frame (x-z plane)

plot_trajectory_and_image(camera_positions_gt, camera_positions_est,camera_orientations_gt,\
                                   camera_orientation_est, points_3D_W,points_3D_triangulated_W, \
                                    points_3D_triangulated_W_invalid, trans_img)
        