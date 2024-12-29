import os
import numpy as np
import cv2
from VisualOdometryPipeLine import VisualOdometryPipeLine
from plotting_tools import plot_camera_trajectory, plot_num_tracked_keypoints
from utils import load_data_set,load_frame

# Setup
ds = 0  # 0: KITTI, 1: Malaga, 2: parking
debug = True
num_frames_to_process = 160
stride = 2
bootstrap_frames = [1,1+stride]

# Tracking data
num_tracked_keypoints = []
positions_list = []
rotations_list = []

# Load data set
K, img0, img1, malaga_left_images = load_data_set(ds,bootstrap_frames)

# INITIALIZATION 
print("Commencing initialisation")
print(f'\n\nProcessing frame {bootstrap_frames[1]}\n=====================')

VO = VisualOdometryPipeLine(K)
R,t = VO.initialization(img0, img1)

#for i in range(bootstrap_frames[1] + 1, last_frame + 1):
for i in range(bootstrap_frames[1] + 1, num_frames_to_process): #first make it run for the first frames and extend later
    print(f'\n\nProcessing frame {i}\n=====================')
    image = load_frame(ds, i,malaga_left_images)

    R,t = VO.continuous_operation(image)

    positions_list.append(t)
    rotations_list.append(R)
    num_tracked_keypoints.append(VO.pts_last)

    
    # Plot current camera pose
    if debug: plot_camera_trajectory(positions_list, rotations_list,show_rot=False)

print(f"VO pipeline executed over {num_frames_to_process} frames")

# Plot camera trajectory
plot_camera_trajectory(positions_list, rotations_list,show_rot=True)

# Plot number of tracked keypoints
plot_num_tracked_keypoints(num_tracked_keypoints,stride)

