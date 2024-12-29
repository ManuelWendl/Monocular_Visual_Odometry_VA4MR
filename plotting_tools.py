
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb


def draw_matches(img0, img1, keypoints0, keypoints1, matches, mask):
    img_matches = cv2.drawMatches(img0, keypoints0, img1, keypoints1, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(20, 10))
    plt.imshow(img_matches)
    plt.title('Keypoint Matches')
    plt.savefig('out/keypoint_matches.png')
    plt.close()

def plot_3d_points(points3D):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points3D[0], points3D[1], points3D[2], c='r', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('3D Points')
    plt.savefig('out/3d_points.png')
    plt.close()

def plot_inlier_points(img0, img1, pts0_inliers, pts1_inliers):
    # Draw inlier points on the images
    # img0_inliers = img0.copy()
    # img1_inliers = img1.copy()

    # for i in range(pts0_inliers.shape[1]):
    #     cv2.circle(img0_inliers, (int(pts0_inliers[0][i]), int(pts0_inliers[1][i])), 5, (255, 255, 255), 50)
    #     print("pts0_inliers[0][i]:", pts0_inliers[0][i])
    #     print("pts0_inliers[1][i]:", pts0_inliers[1][i])

    # for i in range(pts1_inliers.shape[1]):
    #     cv2.circle(img1_inliers, (int(pts1_inliers[0][i]), int(pts1_inliers[1][i])), 1, (255, 255, 255), 50)

    # # Plot the images with inlier points
    # plt.figure(figsize=(20, 10))
    # plt.subplot(1, 2, 1)
    # plt.imshow(cv2.cvtColor(img0_inliers,cv2.COLOR_BGR2RGB) )#
    # plt.title('Inlier Points in Image 0')
    # plt.subplot(1, 2, 2)
    # plt.imshow(cv2.cvtColor(img1_inliers, cv2.COLOR_BGR2RGB)) #
    # plt.title('Inlier Points in Image 1')
    # plt.show()

    img0_rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    # Extract coordinates from pts0_inliers and pts1_inliers
    pts0_x = pts0_inliers[:, 0, 0]
    pts0_y = pts0_inliers[:, 0, 1]
    pts1_x = pts1_inliers[:, 0, 0]
    pts1_y = pts1_inliers[:, 0, 1]

    # Plot the images with inlier points
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(img0_rgb)
    plt.scatter(pts0_x, pts0_y, c='white', s=20, marker='o')
    plt.title('Inlier Points in Image 0')

    plt.subplot(1, 2, 2)
    plt.imshow(img1_rgb)
    plt.scatter(pts1_x, pts1_y, c='white', s=20, marker='o')
    plt.title('Inlier Points in Image 1')

    plt.savefig('out/inlier_points.png')
    plt.close()

def plot_camera_trajectory(translations, rotations, ground_truth, show_rot=True):
    """
    Plots the trajectory of a camera in 3D space given lists of translation vectors and rotation matrices.

    Parameters:
        translations (list of numpy arrays): A list of 3D translation vectors (shape (3,) or (3, 1)).
        rotations (list of numpy arrays): A list of 3x3 rotation matrices (shape (3, 3)).
        show_rot (bool): If True, plot rotations and translations. If False, plot only translations with spheres at endpoints.
    """
    if len(translations) != len(rotations):
        raise ValueError("The number of translations and rotations must be the same.")

    # Initialize lists for the trajectory points
    trajectory = []
    camera_axes = []  # For visualizing camera orientations

    for t, R in zip(translations, rotations):
        if t.shape == (3, 1):
            t = t.flatten()  # Convert (3, 1) to (3,)
        if t.shape != (3,) or R.shape != (3, 3):
            raise ValueError("Each translation must have shape (3,) or (3, 1) and each rotation must have shape (3, 3).")

        # Store trajectory points and orientation axes
        trajectory.append(t)
        camera_axes.append(R)

    # Convert trajectory to numpy array for easier plotting
    trajectory = np.array(trajectory)

    # Generate rainbow colors based on the number of trajectory points
    num_points = len(trajectory)
    hues = np.linspace(0, 1, num_points)  # Generate hues from 0 to 1
    colors = [hsv_to_rgb((hue, 1, 1)) for hue in hues]

    # Plot the trajectory in 3D
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot trajectory line with rainbow colors
    for i in range(num_points - 1):
        ax.plot(trajectory[i:i+2, 0], trajectory[i:i+2, 1], trajectory[i:i+2, 2], color=colors[i])

    if ground_truth != []:
        # Plot ground truth trajectory
        ground_truth = np.array(ground_truth)
        ax.plot(ground_truth[:, 0], ground_truth[:, 1], np.zeros_like(ground_truth[:, 1]), color='black', label='Ground Truth')

    if show_rot:
        # Plot orientation at each trajectory point
        for t, R in zip(trajectory, camera_axes):
            origin = t
            scale = 1.0
            x_axis = origin + R[:, 0] * scale  # Scale axis for visualization
            y_axis = origin + R[:, 1] * scale
            z_axis = origin + R[:, 2] * scale

            ax.quiver(origin[0], origin[1], origin[2],
                      x_axis[0] - origin[0], x_axis[1] - origin[1], x_axis[2] - origin[2],
                      color='red', label='X-axis' if t is trajectory[0] else "")
            ax.quiver(origin[0], origin[1], origin[2],
                      y_axis[0] - origin[0], y_axis[1] - origin[1], y_axis[2] - origin[2],
                      color='green', label='Y-axis' if t is trajectory[0] else "")
            ax.quiver(origin[0], origin[1], origin[2],
                      z_axis[0] - origin[0], z_axis[1] - origin[1], z_axis[2] - origin[2],
                      color='blue', label='Z-axis' if t is trajectory[0] else "")
    else:
        # Plot spheres at each translation endpoint
        for i, (t, color) in enumerate(zip(trajectory, colors)):
            size = 100 if i == 0 else 50  # First sphere is twice the size
            ax.scatter(t[0], t[1], t[2], color=color, s=size, label='Translation Point' if i == 0 else "")

    # Set labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Camera Trajectory ({'with rotations' if show_rot else 'only translations'})")
    #ax.legend()

    ax.axis('equal')

    plt.savefig('out/camera_trajectory.png')
    plt.close()


def plot_num_tracked_keypoints(num_tracked_keypoints_in_each_frame,stride):

    time_steps = np.arange(len(num_tracked_keypoints_in_each_frame))*stride

    # Plot the numbers over time steps
    plt.figure(figsize=(8, 5))
    plt.plot(time_steps, num_tracked_keypoints_in_each_frame, marker='o', linestyle='-', color='b', label='Data')

    # Add labels and title
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('Number of Tracked Keypoints Over all VO iterations')
    plt.grid(True)
    #plt.legend()

    # Show the plot
    plt.savefig('out/num_tracked_keypoints.png')
    plt.close()

