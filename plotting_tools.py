
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from scipy.interpolate import make_interp_spline


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

def plot_camera_trajectory(translations, rotations, ground_truth, landmarks, show_rot=True):
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
    plt.figure(figsize=(10, 7))
    # Plot trajectory line with rainbow colors
    for i in range(num_points - 1):
        plt.plot(trajectory[i:i+2, 0], trajectory[i:i+2, 2], color=colors[i])

    if len(ground_truth) > 0:
        # Plot ground truth trajectory
        ground_truth = np.array(ground_truth)
        plt.plot(ground_truth[:, 0], ground_truth[:, 1], color='black', label='Ground Truth')

    if len(landmarks) > 0:
        # Plot landmarks
        landmarks = np.array(landmarks)
        plt.scatter(landmarks[:, 0], landmarks[:, 2], color='orange', s=10, label='Landmarks')

    for i, (t, color) in enumerate(zip(trajectory, colors)):
        size = 100 if i == 0 else 50  # First sphere is twice the size
        plt.scatter(t[0], t[2], color=color, s=size, label='Translation Point' if i == 0 else "")

    # Set labels and title
    plt.title(f"Camera Trajectory ({'with rotations' if show_rot else 'only translations'})")
    #ax.legend()

    plt.axis('auto')

    plt.savefig('out/camera_trajectory.png')
    plt.pause(0.1)
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

### PLOTS FOR THE INTERFACE ###
def plot_interface(image, inliers, outliers, translations, rotations, ground_truth, num_tracked_landmarks_list, landmarks):
    fig = plt.figure(figsize=(10, 6))

    # Add subplots to the figure
    ax1 = fig.add_subplot(2, 2, 1)  # Top-left (inliers outliers)
    ax2 = fig.add_subplot(2, 2, 2)  # Top-right (trajectory of last 20 frames and landmarks)
    ax3 = fig.add_subplot(2, 2, 3)  # Bottom-left (num of tracked landmarks of 20 last frames)
    # ax4 = fig.add_subplot(2, 2, 4, projection='3d')  # Bottom-right (camera trajectory)
    ax4 = fig.add_subplot(2, 2, 4)

    inferface_plot_inliers_outliers(ax1, image, inliers, outliers)
    # interface_plot_camera_trajectory_3d(ax4, translations, rotations, ground_truth, show_rot=False)
    interface_plot_camera_trajectory_2d(ax2, translations, ground_truth)
    interface_plot_camera_trajectory_2d_20_prev(ax4, translations, ground_truth, landmarks)
    interface_plot_num_tracked_landmarks(ax3, num_tracked_landmarks_list)

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig('out/interface_plot.png')  # Save the plot to a file
    plt.close()  # Close the figure to avoid display in interactive environments



def interface_plot_num_tracked_landmarks(ax, num_tracked_landmarks_list):
    ax.set_title('# of tracked landmarks over the last 20 frames')

    # Calculate the maximum value in the list
    max_value = max(num_tracked_landmarks_list) if num_tracked_landmarks_list else 0

    # Set y-axis range dynamically
    if max_value > 200:
        ax.set_ylim([0, max_value + 10])  # Add a small margin
    else:
        ax.set_ylim([0, 200])

    # Generate dynamic x-values based on the length of the list
    num_points = len(num_tracked_landmarks_list)
    x_values = list(range(-num_points, 0))
    # Set x-ticks for every 5th point, if possible
    ticks = [x for x in x_values if x % 5 == 0]
    ax.set_xticks(ticks)

    # Plot the data
    ax.plot(x_values, num_tracked_landmarks_list, marker='o', linestyle='-', color='firebrick')

    # Optional: Add labels and grid
    ax.set_xlabel("Frames")
    ax.set_ylabel("# of Tracked Landmarks")
    ax.grid(True, linestyle='--', alpha=0.5)



def inferface_plot_inliers_outliers(ax, image, inliers, outliers):
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if inliers is None or len(inliers) == 0:
        ax.set_title('Current image with no RANSAC inliers and outliers')
        return
    if isinstance(inliers, np.ndarray):
        if inliers.ndim == 1:
            inliers = inliers.reshape(1, -1)
        else:
            inliers = inliers.squeeze().tolist()
    if isinstance(outliers, np.ndarray):
        if outliers.ndim == 1:
            outliers = outliers.reshape(1, -1)
        else:
            outliers = outliers.squeeze().tolist()

    for inlier in inliers:
        # Flatten to 1D list
        ax.scatter(inlier[0], inlier[1], c='green', s=40, marker='x') 

    if outliers == []:
        print("No outliers")
    else:
        if isinstance(outliers[0], float):
            ax.scatter(outliers[0], outliers[1], c='red', s=40, marker='x')
        else:
            for outlier in outliers:
                # Flatten to 1D list
                ax.scatter(outlier[0], outlier[1], c='red', s=40, marker='x')
    
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Current image with RANSAC inliers and outliers')



def interface_plot_camera_trajectory_2d(ax, translations, ground_truth):
    trajectory = []

    for t in translations:
        if t.shape == (3, 1):
            t = t.flatten()  # Convert (3, 1) to (3,)
        if t.shape != (3,):
            raise ValueError("Each translation must have shape (3,) or (3, 1).")

        # Append only x, y coordinates
        trajectory.append(t[:3])  # Take the first two elements (x, y)

    # Convert trajectory to numpy array for easier plotting
    trajectory = np.array(trajectory)

    # Plot the 2D trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 2], marker='o', linestyle='-', color='blue', label='Estimated Trajectory')

    if len(ground_truth) > 0:
        # Extract x, y ground truth values and plot them
        ground_truth = np.array(ground_truth)
        ax.plot(ground_truth[:, 0], ground_truth[:, 1], linestyle='--', color='black', label='Ground Truth')

    # Add labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Full Trajectory")
    ax.legend(loc='upper right', fontsize=7)

    # Optional: Set equal aspect ratio for a proper spatial view
    ax.axis('equal')
    ax.grid(True, linestyle='--', alpha=0.5)



def interface_plot_camera_trajectory_2d_20_prev(ax, translations, ground_truth, landmarks):
    translations = translations[-20:]  # Use only the last 20 translations
    trajectory = []

    for t in translations:
        if t.shape == (3, 1):
            t = t.flatten()  # Convert (3, 1) to (3,)
        if t.shape != (3,):
            raise ValueError("Each translation must have shape (3,) or (3, 1).")

        # Append only x, y coordinates
        trajectory.append(t[:3])  # Take the first three elements (x, y, z)

    # Convert trajectory to numpy array for easier plotting
    trajectory = np.array(trajectory)

    # Plot the 2D trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 2], marker='o', linestyle='-', color='blue', label='Estimated Trajectory')

    if len(ground_truth) > 0:
        # Extract x, y ground truth values and plot them
        ground_truth = np.array(ground_truth)
        ax.plot(ground_truth[:, 0], ground_truth[:, 1], linestyle='--', color='black', label='Ground Truth')

    if len(landmarks) > 0:
        # Plot landmarks
        landmarks = np.array(landmarks)
        ax.scatter(landmarks[:, 0], landmarks[:, 2], color='orange', s=10, label='Landmarks')

        # Calculate min and max values including landmarks
        x_min = min(trajectory[:, 0].min(), landmarks[:, 0].min())
        x_max = max(trajectory[:, 0].max(), landmarks[:, 0].max())
        y_min = min(trajectory[:, 2].min(), landmarks[:, 2].min())
        y_max = max(trajectory[:, 2].max(), landmarks[:, 2].max())
    else:
        # Calculate min and max values based only on the trajectory
        x_min, x_max = trajectory[:, 0].min(), trajectory[:, 0].max()
        y_min, y_max = trajectory[:, 2].min(), trajectory[:, 2].max()

    # Add a small margin to the limits
    x_margin = (x_max - x_min) * 0.1 if x_max != x_min else 1
    y_margin = (y_max - y_min) * 0.1 if y_max != y_min else 1

    ax.set_xlim([x_min - x_margin, x_max + x_margin])
    ax.set_ylim([y_min - y_margin, y_max + y_margin])

    # Add labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Trajectory with Landmarks (over last 20 frames)")
    ax.legend(loc='upper right', fontsize=7)

    # Optional: Set equal aspect ratio for a proper spatial view
    ax.set_aspect('auto')  # Allow dynamic scaling
    ax.grid(True, linestyle='--', alpha=0.5)






def interface_plot_camera_trajectory_3d(ax, translations, rotations, ground_truth, show_rot): # same as plot_camera_trajectory func but without saving stuff
    if len(translations) != len(rotations):
        raise ValueError("The number of translations and rotations must be the same.")

    trajectory = []
    camera_axes = []

    for t, R in zip(translations, rotations):
        if t.shape == (3, 1):
            t = t.flatten()
        if t.shape != (3,) or R.shape != (3, 3):
            raise ValueError("Each translation must have shape (3,) or (3, 1) and each rotation must have shape (3, 3).")

        trajectory.append(t)
        camera_axes.append(R)

    trajectory = np.array(trajectory)

    # Generate rainbow colors based on the number of trajectory points
    num_points = len(trajectory)
    hues = np.linspace(0, 1, num_points)  # Generate hues from 0 to 1
    colors = [hsv_to_rgb((hue, 1, 1)) for hue in hues]

    for i in range(num_points - 1):
        ax.plot(trajectory[i:i+2, 0], trajectory[i:i+2, 1], trajectory[i:i+2, 2], color=colors[i])

    if len(ground_truth) >0:
        ground_truth = np.array(ground_truth)
        ax.plot(ground_truth[:, 0], ground_truth[:, 1], np.zeros_like(ground_truth[:, 1]), color='black', label='Ground Truth')

    if show_rot:
        for t, R in zip(trajectory, camera_axes):
            origin = t
            scale = 1.0
            x_axis = origin + R[:, 0] * scale
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
        for i, (t, color) in enumerate(zip(trajectory, colors)):
            size = 100 if i == 0 else 50  # First sphere is twice the size
            ax.scatter(t[0], t[2], t[1], color=color, s=size, label='Translation Point' if i == 0 else "")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Camera Trajectory ({'with rotations' if show_rot else 'only translations'})")
    ax.legend(loc='upper right', fontsize=7)
    ax.view_init(elev=40, azim=-45)


def reprojection_error_plot(self, mean_error, std_error, max_error, below_threshold):
    plt.figure(figsize=(10, 6))
    plt.plot(mean_error, label="Mean Error", marker="o")
    plt.plot(std_error, label="Std Dev", marker="s")
    plt.plot(max_error, label="Max Error", marker="^")
    plt.title(f"Reprojection Errors")
    plt.xlabel("Frame")
    plt.ylabel("Reprojection Error (pixels)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"out/reprojection_error.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(below_threshold, label="% Below 1 Pixel", marker="o", color="green")

    # Fit a spline to the percentage data
    if len (below_threshold) > 5:
        x = np.arange(len(below_threshold))
        y = np.array(below_threshold)
        spline = make_interp_spline(x, y, k=3)  # cubic spline
        x_smooth = np.linspace(x.min(), x.max(), 15)
        y_smooth = spline(x_smooth)
        plt.plot(x_smooth, y_smooth, label="Trend", color="blue", linestyle="--")


    plt.title(f"Percentage of Points Below 1 Pixel Reprojection Error")
    plt.xlabel("Frame")
    plt.ylabel("Percentage (%)")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"out/percentage_below_1.png")
    plt.close()
