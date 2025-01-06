import os
import time
import numpy as np
import cv2
from VisualOdometryPipeLine import VisualOdometryPipeLine
from utils import load_data_set,load_frame
import matplotlib
import matplotlib.pyplot as plt

t_start = time.time()

# Setup
ds = 1  # 0: KITTI, 1: Malaga, 2: parking
interface_plot = True

if ds == 0:
    last_frame = 2761
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


if interface_plot:
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))


# Load data set
K, img0, img1, malaga_left_images, ground_truth = load_data_set(ds, bootstrap_frames)

# INITIALIZATION 
print("Commencing initialisation")
print(f'\n\nProcessing frame {bootstrap_frames[1]}\n=====================')

VO = VisualOdometryPipeLine(K, options)
VO.initialization(img0, img1)

t = VO.transforms[-1][1]

translations = np.array(t).reshape(-1,3)
num_tracked_keypoints = np.array([VO.num_pts[-1]]).reshape(-1,1)

if interface_plot:
    image_plot = axs[0,0].imshow(img1, cmap='gray')
    outlier_plot = axs[0,0].plot(VO.outlier_pts_current[:, 0], VO.outlier_pts_current[:, 1], 'rx', markersize=6, label='Outliers')
    inlier_plot = axs[0,0].plot(VO.inlier_pts_current[:, 0], VO.inlier_pts_current[:, 1], 'gx', markersize=6, label='Inliers')
    axs[0,0].set_title('Current image with RANSAC inliers and outliers')
    axs[0,0].legend(loc=4, borderaxespad=0.)

    trajectory_plot = axs[0,1].plot(translations[:, 0], translations[:, 2], 'bo-', linewidth=1, markersize=3, label='Trajectory')
    if len(ground_truth) >0:
        axs[0,1].plot(ground_truth[:, 0], ground_truth[:, 1], 'k--', label='Ground Truth')
    axs[0,1].set_title("Full Trajectory")
    axs[0,1].set_xlabel("X")
    axs[0,1].set_ylabel("Y")
    axs[0,1].legend()

    num_tracked_landmarks_plot = axs[1,0].plot([0],num_tracked_keypoints, '-', color='black', linewidth=1)
    axs[1,0].set_title('# of tracked landmarks over the last 20 frames')
    axs[1,0].set_xlabel("Frames")
    axs[1,0].set_ylabel("# of Tracked Landmarks")

    trajectory_plot1 = axs[1,1].plot(translations[:, 0], translations[:, 2], 'bo-', linewidth=1, markersize=3, label='Trajectory')
    if len(ground_truth) >0:
        axs[1,1].plot(ground_truth[:, 0], ground_truth[:, 1], 'k--', label='Ground Truth')
    landmaeks_plot = axs[1,1].plot(VO.matched_landmarks[:, 0], VO.matched_landmarks[:, 2], 'ro', markersize=6, label='Landmarks')
    axs[1,1].set_title('Landmarks over the last 20 frames')
    axs[1,1].set_xlabel("X")
    axs[1,1].set_ylabel("Y")
    axs[1,1].legend()

    if matplotlib.get_backend() != 'agg':
        plt.pause(0.001)
    else:
        plt.savefig('out/interface_plot.png')


# CONTINUOUS OPERATION 
print("Commencing continuous operation")

for i in range(bootstrap_frames[1] + 1, last_frame): 
    print(f'\n\nProcessing frame {i}\n=====================')
    image = load_frame(ds, i, malaga_left_images)

    VO.continuous_operation(image)

    t = VO.transforms[-1][1]

    translations = np.append(translations, t.T, axis=0)
    num_tracked_keypoints = np.append(num_tracked_keypoints, VO.num_pts[-1])

    if interface_plot: 
        image_plot.set_data(image)
        if VO.outlier_pts_current.shape[0] > 0:
            outlier_plot[0].set_data(VO.outlier_pts_current[:, 0], VO.outlier_pts_current[:, 1])
        inlier_plot[0].set_data(VO.inlier_pts_current[:, 0], VO.inlier_pts_current[:, 1])

        trajectory_plot[0].set_data(translations[:, 0], translations[:, 2])
        axs[0,1].set_xlim([min(translations[:,0])-100,max(translations[:,0])+100])
        axs[0,1].set_ylim([min(translations[:,2])-100,max(translations[:,2])+100])

        if i > bootstrap_frames[1] + 20:
            num_tracked_landmarks_plot[0].set_data(np.arange(i-19,i+1),num_tracked_keypoints[-20:])
        else:
            num_tracked_landmarks_plot[0].set_data(np.arange(bootstrap_frames[1],i+1),num_tracked_keypoints)
        axs[1,0].set_xlim([i-min(i,21),i-1])
        axs[1,0].set_ylim([min(num_tracked_keypoints[max(-i,-20):])-10,max(num_tracked_keypoints[max(-i,-20):])+10])

        trajectory_plot1[0].set_data(translations[max(-i,-20):, 0], translations[max(-i,-20):, 2])
        landmaeks_plot[0].set_data(VO.matched_landmarks[:, 0], VO.matched_landmarks[:, 2])
        axs[1,1].set_xlim([translations[-1,0] - 100, translations[-1,0] + 100])
        axs[1,1].set_ylim([translations[-1,2] - 100, translations[-1,2] + 100])
        
        if matplotlib.get_backend() != 'agg':
            plt.pause(0.001)
        else:
            plt.savefig('out/interface_plot.png')


plt.savefig('out/interface_plot.png')

print(f"VO pipeline executed over {last_frame} frames")

time_elapsed = time.time() - t_start

print(f"Time elapsed: {time_elapsed/60:.2f} min")
