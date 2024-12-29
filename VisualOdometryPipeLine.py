import cv2
import numpy as np

class VisualOdometryPipeLine:
    def __init__(self, K):
        self.sift = cv2.SIFT_create()   # Simple SIFT detector
        self.K = K                      # Camera matrix
        self.R = np.eye(3)              # Rotation matrix
        self.t = np.zeros((3, 1))       # Translation vector
        self.pts_last = None            # Last frame keypoints TODO: Look if necessary
        self.desc_last = None           # Last frame descriptors
        self.keys_last = None           # Last frame keypoints
        self.feature_ratio = 0.25       # Ratio for feature matching
        self.inlier_pts_current = None  # Current frame inlier points (RANSAC)
        self.outlier_pts_current = None # Current frame outlier points (RANSAC)
        self.num_tracked_landmarks_list = [] # Number of tracked landmarks list (inliers of RANSAC) for the last 20 frames
        

    def initial_feature_matching(self, img0, img1):
        # Detect keypoints and compute descriptors
        keypoints0, descriptors0 = self.sift.detectAndCompute(img0, None)
        keypoints1, descriptors1 = self.sift.detectAndCompute(img1, None)

        # Match descriptors using BFMatcher
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors0, descriptors1,k=2)

        # Apply ratio test
        good_matches = []
        for m,n in matches:
            if len(good_matches) > 1000:  # Limit number of matches
                break
            if m.distance < self.feature_ratio*n.distance:
                good_matches.append(m)

        # Extract matched keypoints
        pts0 = np.float32([keypoints0[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts1 = np.float32([keypoints1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Store descriptors of second image 
        self.desc_last = descriptors1
        self.keys_last = keypoints1

        return pts0, pts1
    
    def feature_matching(self, img):
        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.sift.detectAndCompute(img, None)

        # Match descriptors using BFMatcher
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(self.desc_last, descriptors,k=2)

        # Apply ratio test
        good_matches = []
        for m,n in matches:
            if len(good_matches) > 1000:  # Limit number of matches
                break
            if m.distance < self.feature_ratio*n.distance:
                good_matches.append(m)

        # Extract matched keypoints
        pts_last = np.float32([self.keys_last[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts_current = np.float32([keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Store descriptors of new image
        self.desc_last = descriptors
        self.keys_last = keypoints

        return pts_last, pts_current

    def initialization(self, img0, img1):
        pts_last, pts_current = self.initial_feature_matching(img0, img1)

        # Estimate Essential matrix:
        E, ransac_mask = cv2.findEssentialMat(pts_last, pts_current, self.K, method=cv2.RANSAC, prob=0.999, threshold=1)

        # Filter inliers:
        inl_current = pts_current[ransac_mask.ravel() == 1]
        inl_last = pts_last[ransac_mask.ravel() == 1]

        # Estimate relative camera pose of new second frame
        _, R, t,_ = cv2.recoverPose(E, inl_last, inl_current, self.K)

        self.t = self.t + self.R.dot(t)
        self.R = R.dot(self.R)

        self.pts_last = pts_current
    
    def continuous_operation(self, img):
        pts_last, pts_current = self.feature_matching(img)

        # Estimate Essential matrix:
        E, ransac_mask = cv2.findEssentialMat(pts_last, pts_current, self.K, method=cv2.RANSAC, prob=0.999, threshold=1)

        # Filter inliers:
        inl_current = pts_current[ransac_mask.ravel() == 1]
        inl_last = pts_last[ransac_mask.ravel() == 1]

        # Filter outliers:
        outl_current = pts_current[ransac_mask.ravel() == 0]
        outl_last = pts_last[ransac_mask.ravel() == 0]

        # Update the ransac inliers and outliers from the current image for plotting purposes
        self.inlier_pts_current = inl_current
        self.outlier_pts_current = outl_current

        if len(self.num_tracked_landmarks_list) < 20:
            self.num_tracked_landmarks_list.append(len(self.inlier_pts_current))
        elif len(self.num_tracked_landmarks_list) == 20:
            self.num_tracked_landmarks_list.pop(0)
            self.num_tracked_landmarks_list.append(len(self.inlier_pts_current))

        # Estimate relative camera pose of new second frame
        _, R, t,_ = cv2.recoverPose(E, inl_last, inl_current, self.K)

        self.t = self.t + self.R.dot(t)
        self.R = R.dot(self.R)

        self.pts_last = len(inl_current)
    