import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Get first two frames from kitty set, K matrix  and ground truth poses #####
# Read the first two frames
img0 = cv2.imread('data/kitti/05/image_0/000000.png', cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread('data/kitti/05/image_0/000001.png', cv2.IMREAD_GRAYSCALE)
K = np.array([[7.188560000000e+02, 0, 6.071928000000e+02],
                    [0, 7.188560000000e+02, 1.852157000000e+02],
                    [0, 0, 1]])
ground_truth = np.loadtxt('data/kitti/poses/05.txt')
ground_truth = ground_truth[:, [-9, -5, -1]]
tgt_C1_C0 = ground_truth[1,:]
tgt_C2_C0 = ground_truth[2,:]

# Tools #####################################################################
def invert(R, t):
    R_inv = R.T
    t_inv = -R_inv @ t
    return R_inv, t_inv
def draw_matches(match_img, ptsA, ptsB):
    # Convert gray image to bgr image
    match_img = cv2.cvtColor(match_img, cv2.COLOR_GRAY2BGR)
    for i in range(len(ptsA)):
        ptA = ptsA[i,:].astype(int)
        ptB = ptsB[i,:].astype(int)
        #cv2.circle(img0, tuple(map(int, pt0)), 2, (0, 255, 0), -1)
        #cv2.circle(img_trans, tuple(map(int, pt1)), 2, (0, 255, 0), -1)
        cv2.line(match_img, ptA, ptB, (0, 255, 0), 2,0,0)
    return match_img

def draw_pts(pts,_img):
    # Convert gray image to bgr image
    _img = cv2.cvtColor(_img, cv2.COLOR_GRAY2BGR)
    for i in range(pts.shape[0]):
        ptA = pts[i,:].astype(int)
        cv2.circle(_img, ptA, 2, (0, 255, 0), -1)

    return _img
#############################################################################

# Visualize ground truth ####################################################
def plot_trajectory_with_spheres_2d(gt_trajectory, pts_3d, num_points=10):
    """ 
    Plots a 2D trajectory in the X-Z plane given an Nx3 array, with individual points as spheres.

    Parameters:
    - trajectory: Nx3 NumPy array representing the trajectory (columns: X, Y, Z).
    - num_points: Number of points to plot. If None, all points are plotted.
    """
    title = "2D Trajectory (X-Z Plane)"
    labels = ('X', 'Z')
    sphere_color = 'blue'
    line_color = 'blue'
    line_width = 1
    sphere_size = 50

    if gt_trajectory.shape[1] != 3:
        raise ValueError("Input trajectory must be an Nx3 array.")
    
    # Determine the number of points to plot
    if num_points is None or num_points > gt_trajectory.shape[0]:
        num_points = gt_trajectory.shape[0]
    
    # Select the first `num_points`
    trajectory_to_plot = gt_trajectory[:num_points]

    # Unpack coordinates (only X and Z are needed for the 2D plot)
    x, z = trajectory_to_plot[:, 0], trajectory_to_plot[:, 2]
    
    # Plot the trajectory
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    ax.plot(x, z, color=line_color, linewidth=line_width, label='Trajectory Path')  # Connecting line
    ax.scatter(x, z, color=sphere_color, s=sphere_size, label='Trajectory Points')  # Points as spheres

    # Plot 3d points
    ax.scatter(pts_3d[:, 0], pts_3d[:, 2], color='red',marker='x', s=sphere_size, label='3D Points')

    # Axis labels and title
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(title)
    ax.legend()
    ax.grid()
    plt.show()
    
    plt.show()
#plot_trajectory_with_spheres_2d(ground_truth,pts_3d_W)
#############################################################################


# Get SIFT features in both images and match with bruteforce matcher ########
sift = cv2.SIFT_create()
keypoints0, descriptors0 = sift.detectAndCompute(img0, None)
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
descriptors1_all = descriptors1
matcher = cv2.BFMatcher()
matches_01 = matcher.knnMatch(descriptors0, descriptors1, k=2)
# Get good matches using ratio test
good_matches_01 = []
bad_matches_01 = []
for m, n in matches_01:
    if m.distance < 0.25 * n.distance:
        good_matches_01.append(m)
    else:
        bad_matches_01.append(m)
#############################################################################


# Get matching 2d points in both images #####################################
kpts_0 = np.float32([keypoints0[m.queryIdx].pt for m in good_matches_01])
kpts_1 = np.float32([keypoints1[m.trainIdx].pt for m in good_matches_01])
#############################################################################


# Update descriptors with only good matches ##################################
descriptors0 = descriptors0[[m.queryIdx for m in good_matches_01], :]
descriptors1_cands = descriptors1[[m.trainIdx for m in bad_matches_01], :]
descriptors1 = descriptors1[[m.trainIdx for m in good_matches_01], :]
#############################################################################


# Estimate pose using Essential matrix ######################################
E, _ = cv2.findEssentialMat(kpts_0, kpts_1, K)
# pts0 = pts0[E_mask.ravel() == 1,:]
# pts1 = pts1[E_mask.ravel() == 1,:]
# descriptors0 = descriptors0[E_mask.ravel() == 1,:]
# descriptors1 = descriptors1[E_mask.ravel() == 1,:]
# Recover pose from the essential matrix
_, R_C1_C0, t_C1_C0, recover_pose_mask = cv2.recoverPose(E, kpts_0, kpts_1, K)
initial_scaling = tgt_C1_C0[2] / t_C1_C0[2]
t_C1_C0 *= initial_scaling

#print("t_C1_C0: ", t_C1_C0)
#R_C1_C0, t_C1_C0 = invert(R_C0_C1, t_C0_C1)
kpts_0 = kpts_0[recover_pose_mask.ravel() == 255,:]
kpts_1 = kpts_1[recover_pose_mask.ravel() == 255,:]
descriptors0 = descriptors0[recover_pose_mask.ravel() == 255,:]
descriptors1 = descriptors1[recover_pose_mask.ravel() == 255,:]
cam_trajectory = [np.zeros((3,1)),t_C1_C0]
#############################################################################


# Mark matches in images ####################################################
img_trans_01 = draw_matches(img1.copy(), kpts_0, kpts_1)
#############################################################################

# Triangulate points based on img0,img1 and estimated pose ##################
P0 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
P1 = K @ np.hstack((R_C1_C0, t_C1_C0))
pts_W_init = cv2.triangulatePoints(P0,P1, kpts_0.T, kpts_1.T)
pts_W_init /= pts_W_init[3, :]
pts_W_init = pts_W_init[:3, :].T
pts_W_init *= initial_scaling
print("Triangulated points shape: ", pts_W_init.shape)
#############################################################################


# Filter triangulated points ################################################
# Create mask for points in front of camera
recover_pose_mask = (pts_W_init[:, 2] > 0) & (pts_W_init[:, 2] < 100)
pts_W_init = pts_W_init[recover_pose_mask,:]
#############################################################################


# Get all matches between img1 and img2 #####################################
img2 = cv2.imread('data/kitti/05/image_0/000002.png', cv2.IMREAD_GRAYSCALE)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
descriptors2_all = descriptors2
matches_1all_2 = matcher.knnMatch(descriptors1_all, descriptors2, k=2)
good_matches_1all_2 = []
for m, n in matches_1all_2:
    if m.distance < 0.25 * n.distance:
        good_matches_1all_2.append(m)

kpts_1all_2 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches_1all_2])
kpts_2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches_1all_2])
#############################################################################


# Clean 3d points ###########################################################
tracking_matches_12 = matcher.knnMatch(descriptors1, descriptors2, k=2)
tracking_matches_12_mask = np.zeros((pts_W_init.shape[0],1), dtype=bool)
good_tracking_matches_12 = []
i = 0
for m, n in tracking_matches_12:
    if m.distance < 0.25 * n.distance:
        tracking_matches_12_mask[i] = True
        good_tracking_matches_12.append(m)
    i+=1
kpts_1_match2 = np.float32([keypoints1[m.queryIdx].pt for m in good_tracking_matches_12])
kpts_2_match1 = np.float32([keypoints2[m.trainIdx].pt for m in good_tracking_matches_12])
descriptors2 = descriptors2[[m.trainIdx for m in good_tracking_matches_12], :]

pts_W_init = pts_W_init[tracking_matches_12_mask.ravel() == 1,:]
#############################################################################


# Mark matches in images ####################################################
pts1_img = draw_pts(kpts_1all_2,img1)
pts2_img = draw_pts(kpts_2,img2)
# cv2.imshow("img1",pts1_img)
# cv2.imshow("img2",pts2_img)
# cv2.waitKey(0)


img_trans_12 = draw_matches(img2.copy(), kpts_1all_2, kpts_2)
#############################################################################

# Use PnP to estimate pose between img1 and img2 #############################
sucess, R_C1_C2, t_C1_C2, inliers = cv2.solvePnPRansac(pts_W_init, kpts_2_match1, K, np.zeros(4), flags=cv2.SOLVEPNP_ITERATIVE, confidence=0.999 ,reprojectionError=2)
R_C1_C2 = cv2.Rodrigues(R_C1_C2)[0]

R_C2_C1, t_C2_C1 = invert(R_C1_C2, t_C1_C2)

R_C2_W = R_C2_C1 @ R_C1_C0
t_C2_W = t_C2_C1 + t_C1_C0

cam_trajectory.append(t_C2_W)
print("t_C2_W: ", t_C2_W)
#############################################################################


# Triangulate points based on img1,img2 and estimated pose ##################
P2 = K @ np.hstack((R_C2_C1, t_C2_C1))
pts_C1 = cv2.triangulatePoints(P1,P2, kpts_1all_2.T, kpts_2.T)
pts_C1 /= pts_C1[3, :]
pts_C1 = pts_C1[:3, :].T


# Filter triangulated points ################################################
triang_mask = (pts_C1[:, 2] > 0) & (pts_C1[:, 2] < 100)
pts_C1 = pts_C1[triang_mask,:]

R_C0_C1 = R_C1_C0.T
t_C0_C1 = -R_C0_C1 @ t_C1_C0

pts_W_12 = R_C0_C1 @ pts_W_init.T + t_C0_C1
pts_W_12 = pts_W_12.T

diff = np.linalg.norm(pts_W_12 - pts_W_init, axis=1)
print("Mean diff: ", np.mean(diff))

print("Triangulated points shape: ", pts_W_12.shape)
#############################################################################


#############################################################################
## DO IT FOR ANOTHER FRAME ##################################################
#############################################################################

# Get all matches between img2 and img3 #####################################
img3 = cv2.imread('data/kitti/05/image_0/000003.png', cv2.IMREAD_GRAYSCALE)
keypoints3, descriptors3 = sift.detectAndCompute(img3, None)
matches_2all_3 = matcher.knnMatch(descriptors2_all, descriptors3, k=2)
good_matches_2all_3 = []
for m, n in matches_2all_3:
    if m.distance < 0.25 * n.distance:
        good_matches_2all_3.append(m)

kpts_2all_3 = np.float32([keypoints2[m.queryIdx].pt for m in good_matches_2all_3])
kpts_3 = np.float32([keypoints3[m.trainIdx].pt for m in good_matches_2all_3])
#############################################################################


# Clean 3d points ###########################################################
tracking_matches_23 = matcher.knnMatch(descriptors2, descriptors3, k=2)
tracking_matches_23_mask = np.zeros((pts_W_12.shape[0],1), dtype=bool)
good_tracking_matches_23 = []
i = 0
for m, n in tracking_matches_23:
    if m.distance < 0.25 * n.distance:
        tracking_matches_23_mask[i] = True
        good_tracking_matches_23.append(m)
    i+=1
kpts_2_match3 = np.float32([keypoints2[m.queryIdx].pt for m in good_tracking_matches_23])
kpts_3_match2 = np.float32([keypoints3[m.trainIdx].pt for m in good_tracking_matches_23])
pts_W_12 = pts_W_12[tracking_matches_23_mask.ravel() == 1,:]
#############################################################################


# Mark matches in images ####################################################
img_trans_23 = draw_matches(img2.copy(), kpts_2all_3, kpts_3)
#############################################################################

# Use PnP to estimate pose between img2 and img3 #############################
sucess, R_C2_C3, t_C2_C3, inliers = cv2.solvePnPRansac(pts_W_12, kpts_3_match2, K, np.zeros(4), flags=cv2.SOLVEPNP_ITERATIVE, confidence=0.999 ,reprojectionError=2)
R_C2_C3 = cv2.Rodrigues(R_C2_C3)[0]

R_C3_C2, t_C3_C2 = invert(R_C2_C3, t_C2_C3)

R_C3_W = R_C3_C2 @ R_C2_C1 @ R_C1_C0
t_C3_W = t_C3_C2 + t_C2_C1 + t_C1_C0

cam_trajectory.append(t_C3_W)
print("t_C3_W: ", t_C3_W)
#############################################################################


# Triangulate points based on img1,img2 and estimated pose ##################
P3 = K @ np.hstack((R_C3_C2, t_C3_C2))
pts_C2 = cv2.triangulatePoints(P2,P3, kpts_2all_3.T, kpts_3.T)
pts_C2 /= pts_C2[3, :]
pts_C2 = pts_C2[:3, :].T


# Filter triangulated points ################################################
triang_mask = (pts_C2[:, 2] > 0) & (pts_C2[:, 2] < 100)
pts_C2 = pts_C2[triang_mask,:]

R_C1_C2 = R_C2_C1.T
t_C1_C2 = -R_C1_C2 @ t_C2_C1

pts_W_23 = R_C0_C1 @ R_C1_C2 @ pts_W_12.T + t_C2_C1 + t_C0_C1
pts_W_23 = pts_W_23.T

diff = np.linalg.norm(pts_W_23 - pts_W_12, axis=1)
print("Mean diff: ", np.mean(diff))

print("Triangulated points shape: ", pts_W_23.shape)
#############################################################################

# Visualize trajectory, current image and 3D points ########################################
def plot_cam_trajectory_and_landmarks_and_img(translations, img, pts_3d_01,pts_3d_12,pts_3d_23):

    # Initialize lists for the trajectory points
    trajectory = []

    for t in translations:
        if t.shape == (3, 1):
            t = t.flatten()  # Convert (3, 1) to (3,)
        if t.shape != (3,):
            raise ValueError("Each translation must have shape (3,) or (3, 1) and each rotation must have shape (3, 3).")

        # Store trajectory points and orientation axes
        trajectory.append(t)

    # Convert trajectory to numpy array for easier plotting
    trajectory = np.array(trajectory)
    
    # Plot the trajectory in 2D (X-Z plane) and image
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))

    # Plot trajectory on the first Axes
    ax[0].plot(trajectory[:, 0], trajectory[:, 2], color='black', label='Camera Trajectory')
    ax[0].scatter(trajectory[:, 0], trajectory[:, 2], color='blue', s=20, label='Trajectory Points')
    ax[0].scatter(pts_3d_01[:, 0], pts_3d_01[:, 2], color='green', marker='x', s=20, label='3D Points init')
    ax[0].scatter(pts_3d_12[:, 0], pts_3d_12[:, 2], color='red', marker='x', s=20, label='3D Points 12')
    ax[0].scatter(pts_3d_23[:, 0], pts_3d_23[:, 2], color='orange', marker='x', s=20, label='3D Points 23')

    ax[0].set_xlabel('X')
    ax[0].set_ylabel('Z')
    ax[0].set_title("Camera Trajectory and 3D Landmarks")
    ax[0].grid()
    ax[0].legend()

    # Plot the image on the second Axes
    ax[1].imshow(img, cmap='gray')
    ax[1].set_title("Camera Image")
    ax[1].axis('off')

    plt.tight_layout()
    plt.show()

plot_cam_trajectory_and_landmarks_and_img(cam_trajectory, img_trans_12, pts_W_init,pts_W_12,pts_W_23)
############################################################################################