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

    # Extract matched keypoints
    pts0 = np.float32([keypoints0[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts1 = np.float32([keypoints1[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Estimate Essential matrix:

    # Normalize points
    pts0_norm = cv2.undistortPoints(pts0, K, None)
    pts1_norm = cv2.undistortPoints(pts1, K, None)

    # Estimate Fundamental matrix using the 8-point algorithm
    F, mask = cv2.findFundamentalMat(pts0_norm, pts1_norm, cv2.FM_8POINT)

    # Convert Fundamental matrix to Essential matrix
    E = K.T @ F @ K

    # Use the mask to filter inliers
    pts0 = pts0[mask.ravel() == 1]
    pts1 = pts1[mask.ravel() == 1]

    # Recover relative camera pose
    _, R, t, mask = cv2.recoverPose(E, pts0, pts1, K)

    print(f"pts0 shape: {pts0.shape}")
    print(f"pts1 shape: {pts1.shape}")
    print(f"R shape: {R.shape}")
    print(f"t shape: {t.shape}")

    # Ensure pts0 and pts1 are in the correct shape for triangulation
    if pts0.shape[0] < 8 or pts1.shape[0] < 8:
        raise ValueError("Not enough inlier points for triangulation")

    pts0 = pts0.reshape(2, -1)
    pts1 = pts1.reshape(2, -1)

    points4D = cv2.triangulatePoints(np.hstack((np.eye(3), np.zeros((3, 1)))), np.hstack((R, t)), pts0, pts1)
    points3D = points4D[:3] / points4D[3]

    # Triangulate points to reconstruct 3D landmarks

    return R, t, points3D, keypoints0, keypoints1, matches

def draw_matches(img0, img1, keypoints0, keypoints1, matches):
    img_matches = cv2.drawMatches(img0, keypoints0, img1, keypoints1, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
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