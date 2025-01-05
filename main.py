import os
import numpy as np
import cv2
from VisualOdometryPipeLine import VisualOdometryPipeLine
from plotting_tools import plot_camera_trajectory, plot_num_tracked_keypoints, plot_interface
from utils import load_data_set,load_frame

# Setup
ds = 2  # 0: KITTI, 1: Malaga, 2: parking
debug = False
interface_plot = False

if ds == 0:
    last_frame = 2761 # 2761
    bootstrap_frames = [0,2]

    options = {
    # Landmark options
    'min_dist_landmarks': 1,
    'max_dist_landmarks': 150,
    'min_baseline_angle': 2,
    'min_baseline_frames': 2,
    
    # Feature detection options
    'feature_ratio': 0.8,
    'feature_max_corners': 1400,
    'feature_quality_level': 0.1,
    'feature_min_dist': 10,
    'feature_block_size': 3,
    'feature_use_harris': False,

    # KLT options
    'winSize': (15, 15),
    'maxLevel': 5,
    'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.01),

    # PnP options
    'PnP_conf': 0.99,
    'PnP_error': 8,
    'PnP_iterations': 500,
    }

elif ds == 1:
    last_frame = 2120
    bootstrap_frames = [0,6]

    options = {
    # Landmark options
    'min_dist_landmarks': 0,
    'max_dist_landmarks': 100,
    'min_baseline_angle': 2,
    'min_baseline_frames': 2,
    
    # Feature detection options
    'feature_ratio': 0.8,
    'feature_max_corners': 1400,
    'feature_quality_level': 0.03,
    'feature_min_dist': 10,
    'feature_block_size': 3,
    'feature_use_harris': False,

    # KLT options
    'winSize': (15, 15),
    'maxLevel': 10,
    'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.01),

    # PnP options
    'PnP_conf': 0.99,
    'PnP_error': 5,
    'PnP_iterations': 500,
    }
    
elif ds == 2:
    last_frame = 598
    bootstrap_frames = [0,6]

    options = {
    # Landmark options
    'min_dist_landmarks': 1,
    'max_dist_landmarks': 50,
    'min_baseline_angle': 2,
    'min_baseline_frames': 2,
    
    # Feature detection options
    'feature_ratio': 0.8,
    'feature_max_corners': 1400,
    'feature_quality_level': 0.1,
    'feature_min_dist': 10,
    'feature_block_size': 3,
    'feature_use_harris': False,

    # KLT options
    'winSize': (15, 15),
    'maxLevel': 10,
    'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.02),

    # PnP options
    'PnP_conf': 0.99,
    'PnP_error': 5,
    'PnP_iterations': 500,
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

for i in range(bootstrap_frames[1] + 1, last_frame): 
    print(f'\n\nProcessing frame {i}\n=====================')
    image = load_frame(ds, i, malaga_left_images)

    VO.continuous_operation(image)

    R = VO.transforms[-1][0]
    t = VO.transforms[-1][1]

    positions_list.append(t)
    rotations_list.append(R)
    num_tracked_keypoints.append(VO.num_pts[-1])

    if debug: plot_camera_trajectory(positions_list, rotations_list,ground_truth, VO.matched_landmarks, show_rot=False)

    if interface_plot or i % 50 == 0: 
        inlier_pts_current = VO.inlier_pts_current
        outlier_pts_current = VO.outlier_pts_current
        num_tracked_landmarks_list = VO.num_tracked_landmarks_list
        plot_interface(image, inlier_pts_current, outlier_pts_current, 
                       positions_list, rotations_list, ground_truth, num_tracked_landmarks_list, VO.matched_landmarks)


print(f"VO pipeline executed over {last_frame} frames")

inlier_pts_current = VO.inlier_pts_current
outlier_pts_current = VO.outlier_pts_current
num_tracked_landmarks_list = VO.num_tracked_landmarks_list
plot_interface(image, inlier_pts_current, outlier_pts_current, 
               positions_list, rotations_list, ground_truth, num_tracked_landmarks_list, VO.matched_landmarks)

# Plot camera trajectory
plot_camera_trajectory(positions_list, rotations_list,ground_truth, [], show_rot=False)

# Plot number of tracked keypoints
plot_num_tracked_keypoints(num_tracked_keypoints)

