import os
import numpy as np
import cv2
from Initilization import initialization, draw_matches, plot_3d_points, plot_inlier_points
# Setup
ds = 0  # 0: KITTI, 1: Malaga, 2: parking

# Parking paths
kitti_path = './data/kitti'
malaga_path = './data/malaga'
parking_path = './data/parking'

if ds == 0:
    # need to set kitti_path to folder containing "05" and "poses"
    assert 'kitti_path' in globals(), "kitti_path variable must be defined"
    ground_truth = np.loadtxt(os.path.join(kitti_path, 'poses/05.txt'))
    ground_truth = ground_truth[:, [-9, -1]]
    last_frame = 4540
    K = np.array([[7.188560000000e+02, 0, 6.071928000000e+02],
                  [0, 7.188560000000e+02, 1.852157000000e+02],
                  [0, 0, 1]])
elif ds == 1:
    # Path containing the many files of Malaga 7.
    assert 'malaga_path' in globals(), "malaga_path variable must be defined"
    images = sorted(os.listdir(os.path.join(malaga_path, 
        'malaga-urban-dataset-extract-07_rectified_800x600_Images')))
    left_images = images[2::2]  # Take every second file starting from the third
    last_frame = len(left_images)
    K = np.array([[621.18428, 0, 404.0076],
                  [0, 621.18428, 309.05989],
                  [0, 0, 1]])
elif ds == 2:
    # Path containing images, depths and all...
    assert 'parking_path' in globals(), "parking_path variable must be defined"
    last_frame = 598
    K = np.array([[331.37, 0, 320],
                   [0, 369.568, 240],
                   [0,0,1]])
    ground_truth = np.loadtxt(os.path.join(parking_path, 'poses.txt'))
    ground_truth = ground_truth[:, [-9, -1]]
else:
    raise ValueError("Invalid dataset selection")

# Bootstrap
bootstrap_frames = [0,5]
# need to set bootstrap_frames
if ds == 0:
    img0 = cv2.imread(os.path.join(kitti_path, '05/image_0', 
        f'{bootstrap_frames[0]:06d}.png'), cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(os.path.join(kitti_path, '05/image_0', 
        f'{bootstrap_frames[1]:06d}.png'), cv2.IMREAD_GRAYSCALE)
elif ds == 1:
    img0 = cv2.imread(os.path.join(malaga_path, 
        'malaga-urban-dataset-extract-07_rectified_800x600_Images', 
        left_images[bootstrap_frames[0]]), cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(os.path.join(malaga_path, 
        'malaga-urban-dataset-extract-07_rectified_800x600_Images', 
        left_images[bootstrap_frames[1]]), cv2.IMREAD_GRAYSCALE)
elif ds == 2:
    img0 = cv2.imread(os.path.join(parking_path, 
        f'images/img_{bootstrap_frames[0]:05d}.png'), cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(os.path.join(parking_path, 
        f'images/img_{bootstrap_frames[1]:05d}.png'), cv2.IMREAD_GRAYSCALE)
else:
    raise ValueError("Invalid dataset selection")

#Initialisation:
R, t, points3D, pts0, pts1, keypoints0, keypoints1, matches, mask = initialization(img0, img1, K)
draw_matches(img0, img1, keypoints0, keypoints1, matches, mask)
plot_3d_points(points3D)
plot_inlier_points(img0, img1, pts0, pts1)  
"""
# Continuous operation
for i in range(bootstrap_frames[1] + 1, last_frame + 1):
    print(f'\n\nProcessing frame {i}\n=====================')
    if ds == 0:
        image = cv2.imread(os.path.join(kitti_path, '05/image_0', f'{i:06d}.png'), cv2.IMREAD_GRAYSCALE)
    elif ds == 1:
        image = cv2.imread(os.path.join(malaga_path, 
            'malaga-urban-dataset-extract-07_rectified_800x600_Images', 
            left_images[i]), cv2.IMREAD_GRAYSCALE)
    elif ds == 2:
        image = cv2.imread(os.path.join(parking_path, f'images/img_{i:05d}.png'), cv2.IMREAD_GRAYSCALE)
    else:
        raise ValueError("Invalid dataset selection")
    
    # Makes sure that plots refresh.
    cv2.waitKey(1)
    
    prev_img = image
"""