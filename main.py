import os
import numpy as np
import cv2
from VisualOdometryPipeLine import VisualOdometryPipeLine
from plotting_tools import plot_camera_trajectory, plot_num_tracked_keypoints, inferface_plot_inliers_outliers, plot_interface
from utils import load_data_set,load_frame

# Setup
ds = 2  # 0: KITTI, 1: Malaga, 2: parking
interface_plot = True
num_frames_to_process = 598 # 2761 (Kitti) 
stride = 2 if ds == 1 else 1  # Stride for frame processing
bootstrap_frames = [1,1+stride]

# Tracking data
num_tracked_keypoints = []
positions_list = []
rotations_list = []

# Load data set
K, img0, img1, malaga_left_images, ground_truth = load_data_set(ds, bootstrap_frames)

# INITIALIZATION 
print("Commencing initialisation")
print(f'\n\nProcessing frame {bootstrap_frames[1]}\n=====================')

VO = VisualOdometryPipeLine(K)
VO.initialization(img0, img1)

# CONTINUOUS OPERATION 
print("Commencing continuous operation")

#for i in range(bootstrap_frames[1] + 1, last_frame + 1):
for i in range(bootstrap_frames[1] + 1, num_frames_to_process): #first make it run for the first frames and extend later
    print(f'\n\nProcessing frame {i}\n=====================')
    image = load_frame(ds, i, malaga_left_images)

    VO.continuous_operation(image)

    positions_list.append(VO.t)
    rotations_list.append(VO.R)
    num_tracked_keypoints.append(VO.pts_last)

    
    if interface_plot: 

        inlier_pts_current = VO.inlier_pts_current
        outlier_pts_current = VO.outlier_pts_current
        num_tracked_landmarks_list = VO.num_tracked_landmarks_list
        plot_interface(image, inlier_pts_current, outlier_pts_current, 
                       positions_list, rotations_list, ground_truth, num_tracked_landmarks_list)


print(f"VO pipeline executed over {num_frames_to_process} frames")

# Plot camera trajectory
plot_camera_trajectory(positions_list, rotations_list, ground_truth, show_rot=False)

# Plot number of tracked keypoints
plot_num_tracked_keypoints(num_tracked_keypoints, stride)

