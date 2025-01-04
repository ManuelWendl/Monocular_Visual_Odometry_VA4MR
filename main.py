import os
import numpy as np
import cv2
from VisualOdometryPipeLine import VisualOdometryPipeLine
from plotting_tools import plot_camera_trajectory, plot_num_tracked_keypoints, inferface_plot_inliers_outliers, plot_interface
from utils import load_data_set,load_frame

# Setup
ds = 1  # 0: KITTI, 1: Malaga, 2: parking
debug = False
interface_plot = False
num_frames_to_process = 300 # 2761 (Kitti) , 2121 (Malaga) , 598 (Parking)
stride = 1  # Stride for frame processing
bootstrap_frames = [0,3+stride]

# Working bootstraps####
# KITTI: [0, 1+stride]
# Malaga: [0, ?] # Stride 3 works good
# Parking: [0, 3+stride]
#########################


# Options
options = {
    'min_dist_landmarks': 0,
    'max_dist_landmarks': 100,
    'min_baseline_angle': 2,
    'min_baseline_frames': 2,
    'feature_ratio': 0.5,
    'PnP_conf': 0.99,
    'PnP_error': 5,
    'PnP_iterations': 100,
    'Reproj_threshold': 100,
    'non_lin_refinement': False
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

R = VO.transforms[-1][0]
t = VO.transforms[-1][1]

positions_list.append(t)
rotations_list.append(R)
num_tracked_keypoints.append(VO.num_pts[-1])

if debug: plot_camera_trajectory(positions_list, rotations_list,ground_truth, VO.matched_landmarks, show_rot=False)

# CONTINUOUS OPERATION 
print("Commencing continuous operation")

#for i in range(bootstrap_frames[1] + 1, last_frame + 1):
for i in range(bootstrap_frames[1] + 1,bootstrap_frames[1] + 1+  num_frames_to_process,stride): #first make it run for the first frames and extend later
    print(f'\n\nProcessing frame {i}\n=====================')
    image = load_frame(ds, i, malaga_left_images)

    VO.continuous_operation(image)

    R = VO.transforms[-1][0]
    t = VO.transforms[-1][1]

    positions_list.append(t)
    rotations_list.append(R)
    num_tracked_keypoints.append(VO.num_pts[-1])

    if debug: plot_camera_trajectory(positions_list, rotations_list,ground_truth, VO.matched_landmarks, show_rot=False)

    if i % 50 == 0: 
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

