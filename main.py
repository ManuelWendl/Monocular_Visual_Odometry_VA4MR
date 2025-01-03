import os
import numpy as np
import cv2
from VisualOdometryPipeLine import VisualOdometryPipeLine
from plotting_tools import plot_camera_trajectory, plot_num_tracked_keypoints, inferface_plot_inliers_outliers, plot_interface
from utils import load_data_set,load_frame

# Setup
ds = 0  # 0: KITTI, 1: Malaga, 2: parking
debug = True
interface_plot = True
num_frames_to_process = 200 # 2761 (Kitti) 
stride = 2 if ds == 1 else 1  # Stride for frame processing
bootstrap_frames = [100,101+stride]

# Options
options = {
    'min_dist_landmarks': 2,
    'max_dist_landmarks': 100,
    'min_baseline_angle': 1,
    'feature_ratio': 0.5,
    'PnP_conf': 0.99999,
    'PnP_error': 2,
}


# Tracking data
num_tracked_keypoints = []
positions_list = []
rotations_list = []

# Load data set
K, img0, img1, malaga_left_images, ground_truth = load_data_set(ds, bootstrap_frames)

# INITIALIZATION 
print("Commencing initialisation")
print(f'\n\nProcessing frame {bootstrap_frames[1]}\n=====================')

VO = VisualOdometryPipeLine(K, options)
VO.initialization(img0, img1)

# CONTINUOUS OPERATION 
print("Commencing continuous operation")

#for i in range(bootstrap_frames[1] + 1, last_frame + 1):
for i in range(bootstrap_frames[1] + 1, num_frames_to_process): #first make it run for the first frames and extend later
    print(f'\n\nProcessing frame {i}\n=====================')
    image = load_frame(ds, i, malaga_left_images)

    VO.continuous_operation(image)

    R = VO.transforms[-1][0]
    t = VO.transforms[-1][1]

    positions_list.append(t)
    rotations_list.append(R)
    num_tracked_keypoints.append(VO.num_pts[-1])

    if debug: plot_camera_trajectory(positions_list, rotations_list,ground_truth, VO.matched_landmarks, show_rot=False)

    if interface_plot: 

        inlier_pts_current = VO.inlier_pts_current
        outlier_pts_current = VO.outlier_pts_current
        num_tracked_landmarks_list = VO.num_tracked_landmarks_list
        plot_interface(image, inlier_pts_current, outlier_pts_current, 
                       positions_list, rotations_list, ground_truth, num_tracked_landmarks_list)


print(f"VO pipeline executed over {num_frames_to_process} frames")

inlier_pts_current = VO.inlier_pts_current
outlier_pts_current = VO.outlier_pts_current
num_tracked_landmarks_list = VO.num_tracked_landmarks_list
plot_interface(image, inlier_pts_current, outlier_pts_current, 
               positions_list, rotations_list, ground_truth, num_tracked_landmarks_list)

# Plot camera trajectory
plot_camera_trajectory(positions_list, rotations_list,ground_truth, [], show_rot=False)

# Plot number of tracked keypoints
plot_num_tracked_keypoints(num_tracked_keypoints, stride)

