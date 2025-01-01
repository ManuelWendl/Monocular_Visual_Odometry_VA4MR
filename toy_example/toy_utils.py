import numpy as np
import matplotlib.pyplot as plt
import cv2

def generate_3d_points(num_3d_pts, xy_var, z_base, z_range):
    z_values = z_base + np.random.uniform(*z_range, num_3d_pts)
    grid_size = int(np.ceil(np.sqrt(num_3d_pts)))
    x_coords = np.linspace(-xy_var, xy_var, grid_size)
    y_coords = np.linspace(-xy_var, xy_var, grid_size)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)

    # Flatten and limit to the required number of points
    x_flat = x_grid.flatten()[:num_3d_pts]
    y_flat = y_grid.flatten()[:num_3d_pts]

    # Add noise to x and y coordinates
    x_flat += np.random.uniform(-0.05, 0.05, num_3d_pts)  # Perturb x
    y_flat += np.random.uniform(-0.05, 0.05, num_3d_pts)  # Perturb y

    points_3d = np.column_stack((x_flat, y_flat, z_values))
    return points_3d

def intrinsic_matrix(focal_length, cx, cy):
    return np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ],np.float32)

def project_points(points_3d, P):
    points_2d = (P @ np.hstack((points_3d, np.ones((points_3d.shape[0], 1)))).T).T
    points_2d = points_2d / points_2d[:, 2][:, np.newaxis]
    return points_2d[:, :2]

def plot_trajectory_and_image(camera_positions_gt, camera_positions_est, camera_orientations_gt, camera_orientation_est, points_3d, image):
    camera_positions_gt = np.array(camera_positions_gt)
    camera_positions_est = np.array(camera_positions_est)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Trajectory plot
    ax = axes[0]
    ax.plot(camera_positions_gt[:, 0], camera_positions_gt[:, 2], marker='o', label="GT Trajectory")
    ax.plot(camera_positions_est[:, 0], camera_positions_est[:, 2], marker='x', label="Estimated Trajectory")

    # # Plot orientations for Estimated
    # arrow_length_scalar = 0.3
    # for pos, orient in zip(camera_positions_est[1:], camera_orientation_est[1:]):
    #     ax.quiver(pos[0], pos[2], arrow_length_scalar*orient[0, 0], arrow_length_scalar*orient[2, 0], color='magenta', scale=5)
    #     ax.quiver(pos[0], pos[2], arrow_length_scalar*orient[0, 2], arrow_length_scalar*orient[2, 2], color='cyan', scale=5)

    # # Plot orientations for GT
    # for pos, orient in zip(camera_positions_gt[1:], camera_orientations_gt[1:]):
    #     ax.quiver(pos[0], pos[2], arrow_length_scalar*orient[0, 0], arrow_length_scalar*orient[2, 0], color='red', scale=5)
    #     ax.quiver(pos[0], pos[2], arrow_length_scalar*orient[0, 2], arrow_length_scalar*orient[2, 2], color='blue', scale=5)

    # Formatting
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Z Position (m)")
    ax.set_title("Camera Trajectories and Orientations (Top View, X-Z Plane)")
    ax.legend()
    ax.grid()

    # Image plot
    axes[1].imshow(image, cmap='gray')
    axes[1].axis('off')
    axes[1].set_title("Current Image")


    fig.set_size_inches(32, 16)
    plt.tight_layout()
    plt.show()


def plot_trajectory(camera_positions_gt, camera_positions_est,camera_orientations_gt, camera_orientation_est, points_3d):
    camera_positions_gt = np.array(camera_positions_gt)
    camera_positions_est = np.array(camera_positions_est)

    fig = plt.figure()
    ax = plt.gca()

    # Plot trajectories
    ax.plot(camera_positions_gt[:, 0], camera_positions_gt[:, 2], marker='o', label="GT Trajectory")
    ax.plot(camera_positions_est[:, 0], camera_positions_est[:, 2], marker='x', label="Estimated Trajectory")
    #ax.scatter(points_3d[:, 0], points_3d[:, 2], color='green', label="3D Points")

    # Plot orientations for Estimated
    arrow_length_scalar = 0.3
    for pos, orient in zip(camera_positions_est, camera_orientation_est):
        ax.quiver(pos[0], pos[2], arrow_length_scalar*orient[0, 0], arrow_length_scalar*orient[2, 0], color='magenta', scale=5)
        ax.quiver(pos[0], pos[2], arrow_length_scalar*orient[0, 2], arrow_length_scalar*orient[2, 2], color='cyan', scale=5)

    # Plot orientations for GT
    for pos, orient in zip(camera_positions_gt, camera_orientations_gt):
        ax.quiver(pos[0], pos[2], arrow_length_scalar*orient[0, 0], arrow_length_scalar*orient[2, 0], color='red', scale=5)
        ax.quiver(pos[0], pos[2], arrow_length_scalar*orient[0, 2], arrow_length_scalar*orient[2, 2], color='blue', scale=5)

    # Formatting
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Z Position (m)")
    ax.set_title("Camera Trajectories, Orientations, and 3D Points (Top View, X-Z Plane)")
    ax.legend()
    ax.grid()

    fig.set_size_inches(16, 16)
    plt.show()

def visualize_image(image, title="Image"):
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title(title)
    plt.show()
