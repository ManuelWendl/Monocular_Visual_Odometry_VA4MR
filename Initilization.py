import cv2
import numpy as np
import matplotlib.pyplot as plt


def initialization(img0, img1, K):
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    keypoints0, descriptors0 = sift.detectAndCompute(img0, None)
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)

    # Match descriptors using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors0, descriptors1)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    """
    knn_matches = bf.knnMatch(descriptors0, descriptors1, k=3)

    # Apply ratio test
    ratio_thresh=0.8
    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    good_matches = sorted(good_matches, key=lambda x: x.distance)

    matches=good_matches #matches =matches for comparison
    """
    # Extract matched keypoints
    pts0 = np.float32([keypoints0[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts1 = np.float32([keypoints1[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    pts_0_ex=pts0
    pts_1_ex=pts1

    # Estimate Essential matrix:

    # Normalize points
    pts0_norm = cv2.undistortPoints(pts0, K, None)
    pts1_norm = cv2.undistortPoints(pts1, K, None)

    #filter with RANSAC
    F, mask_RANSAC = cv2.findFundamentalMat(pts0_norm, pts1_norm, cv2.FM_RANSAC, ransacReprojThreshold=1.0, confidence=0.99)

    # Use the mask to filter inliers
    pts0_inliers = pts0[mask_RANSAC.ravel() == 1]
    pts1_inliers = pts1[mask_RANSAC.ravel() == 1]

    pts0_inliers = cv2.undistortPoints(pts0_inliers, K, None) #recommeded by the CoPilot, probably not necessary
    pts1_inliers = cv2.undistortPoints(pts1_inliers, K, None)


    # Estimate Fundamental matrix using the 8-point algorithm
    E, mask_es = cv2.findEssentialMat(pts0_inliers, pts1_inliers, K, cv2.FM_8POINT)

    # Recover relative camera pose
    _, R, t, mask_pose = cv2.recoverPose(E, pts0_inliers, pts1_inliers, K)

    print("R:", R)
    print("t:", t)

    # Ensure pts0 and pts1 are in the correct shape for triangulation
    if pts0.shape[0] < 8 or pts1.shape[0] < 8:
        raise ValueError("Not enough inlier points for triangulation")

    pts0 = pts0_inliers.reshape(2, -1)
    pts1 = pts1_inliers.reshape(2, -1)

    points4D = cv2.triangulatePoints(np.hstack((np.eye(3), np.zeros((3, 1)))), np.hstack((R, t)), pts0, pts1)
    points3D = points4D[:3] / points4D[3]
    # Triangulate points to reconstruct 3D landmarks

    return R, t, points3D, pts_0_ex, pts_1_ex, keypoints0, keypoints1, matches, mask_RANSAC

def draw_matches(img0, img1, keypoints0, keypoints1, matches, mask):
    img_matches = cv2.drawMatches(img0, keypoints0, img1, keypoints1, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(20, 10))
    plt.imshow(img_matches)
    plt.title('Keypoint Matches')
    plt.show()

def plot_3d_points(points3D):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points3D[0], points3D[1], points3D[2], c='r', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('3D Points')
    plt.show()

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

    plt.show()