import cv2
import numpy as np

class VisualOdometryPipeLine:
    def __init__(self,K):
        self.sift = cv2.SIFT_create()   # Simple SIFT detector
        self.feature_ratio = 0.2       # Ratio for feature matching
        self.K = K                      # Camera matrix
        self.matcher = cv2.BFMatcher()  # Matcher for feature matching

        self.R = np.eye(3)              # Rotation matrix
        self.t = np.zeros((3, 1))       # Translation vector
        self.num_pts = 0                # Number of tracked points

        self.matched_descriptor = []    # List to store matched descriptors
        self.matched_landmarks = []     # List to store matched landmarks

        self.potential_descriptor = []  # List to store potential descriptors
        self.potential_keys = []        # List to store potential keypoints

    
    def ratio_test(self, matches):
        good_matches = []
        bad_matches = []
        for m,n in matches:
            if m.distance < self.feature_ratio*n.distance:
                good_matches.append(m)
            else:
                bad_matches.append(m)
        return good_matches, bad_matches
        

    def initial_feature_matching(self, img0, img1):
        # Detect keypoints and compute descriptors
        keypoints0, descriptors0 = self.sift.detectAndCompute(img0, None)
        keypoints1, descriptors1 = self.sift.detectAndCompute(img1, None)

        # Match descriptors using BFMatcher
        matches = self.matcher.knnMatch(descriptors0, descriptors1,k=2)

        # Apply ratio test
        good_matches, bad_matches = self.ratio_test(matches)

        # Extract matched keypoints
        pts0 = np.float32([keypoints0[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts1 = np.float32([keypoints1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Store descriptors of second image 
        self.matched_descriptor = [descriptors1[m.trainIdx] for m in good_matches]

        # Store potential features for next image
        self.potential_descriptor = [descriptors1[m.trainIdx] for m in bad_matches]
        self.potential_keys = [keypoints1[m.trainIdx] for m in bad_matches]

        return pts0, pts1
    
    def feature_matching(self, img):
        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.sift.detectAndCompute(img, None)

        # Match already matched descriptors using BFMatcher
        old_matches = self.matcher.knnMatch(np.float32(self.matched_descriptor), descriptors,k=2)

        # Apply ratio test
        good_old_matches, bad_old_matches = self.ratio_test(old_matches)

        # Update matched landmarks
        self.matched_landmarks = self.matched_landmarks[[m.queryIdx for m in good_old_matches]]

        pts_last = []
        pts_current = []
        pts_current_landmarks = np.float32([keypoints[m.trainIdx].pt for m in good_old_matches])

        # Check if there are enough matches
        if len(good_old_matches) < 100:
            # Match potential features
            new_matches = self.matcher.knnMatch(np.array(self.potential_descriptor), descriptors,k=2)

            # Apply ratio test
            good_new_matches, bad_new_matches = self.ratio_test(new_matches)

            # Extract matched keypoints
            pts_last = np.float32([self.potential_keys[m.queryIdx].pt for m in good_new_matches]).reshape(-1, 1, 2)
            pts_current = np.float32([keypoints[m.trainIdx].pt for m in good_new_matches]).reshape(-1, 1, 2)

            # Store descriptors of new image
            matched_descriptors = [descriptors[m.trainIdx] for m in good_old_matches]
            matched_descriptors.extend([descriptors[m.trainIdx] for m in good_new_matches])

            self.matched_descriptor = matched_descriptors
        else:
            self.matched_descriptor = [descriptors[m.trainIdx] for m in good_old_matches]
        
        self.potential_descriptor = descriptors
        self.potential_keys = keypoints

        return pts_current_landmarks, pts_last, pts_current

    def initialization(self, img0, img1):
        pts_last, pts_current = self.initial_feature_matching(img0, img1)

        # Estimate Essential matrix:
        E, ransac_mask = cv2.findEssentialMat(pts_last, pts_current, self.K, method=cv2.RANSAC, prob=0.999, threshold=1)

        # Filter inliers:
        inl_current = pts_current[ransac_mask.ravel() == 1]
        inl_last = pts_last[ransac_mask.ravel() == 1]
        self.matched_descriptor = [self.matched_descriptor[i] for i in range(len(ransac_mask)) if ransac_mask[i] == 1]

        # Estimate relative camera pose of new second frame
        _, R, t,_ = cv2.recoverPose(E, inl_last, inl_current, self.K)

        new_t = self.t + self.R.dot(t)
        new_R = R.dot(self.R)

        matched_landmarks = cv2.triangulatePoints(self.K@np.hstack((self.R, self.t)),self.K@np.hstack((new_R, new_t)), inl_last.squeeze().T, inl_current.squeeze().T).T
        self.matched_landmarks = matched_landmarks[:, :3] / matched_landmarks[:, 3, None]

        # Disambiguate landmarks
        depth_last = (self.R @ self.matched_landmarks.T + self.t).T[:, 2]
        depth_current = (new_R @ self.matched_landmarks.T + new_t).T[:, 2]
        valid_indices = (depth_last > 0) & (depth_current > 0)

        self.matched_landmarks = self.matched_landmarks[valid_indices]
        self.matched_descriptor = [self.matched_descriptor[i] for i in range(len(valid_indices)) if valid_indices[i]]


        self.R = new_R
        self.t = new_t

        self.num_pts = len(inl_current)
    
    def continuous_operation(self, img):
        pts_current_landmarks, pts_last, pts_current = self.feature_matching(img)

        sucess = False

        if len(pts_current_landmarks) >= 50:
            sucess, R, t, inliers = cv2.solvePnPRansac(self.matched_landmarks, pts_current_landmarks, self.K, np.zeros(4), flags=cv2.SOLVEPNP_ITERATIVE, confidence=0.999 ,reprojectionError=2)
            R, _ = cv2.Rodrigues(R)

            if sucess and len(inliers) > 4:
                # Filter only inliers
                # self.matched_landmarks = self.matched_landmarks[np.int16(inliers.squeeze())]
                # self.matched_descriptor = np.float32(self.matched_descriptor)[np.int16(inliers.squeeze())]
                print("SolvePnP")

        if not sucess:
            E, ransac_mask = cv2.findEssentialMat(pts_last, pts_current, self.K, method=cv2.RANSAC, prob=0.999, threshold=1)
            # Filter inliers:
            inl_current = pts_current[ransac_mask.ravel() == 1]
            inl_last = pts_last[ransac_mask.ravel() == 1]
            # Estimate relative camera pose of new second frame
            _, R, t,_ = cv2.recoverPose(E, inl_last, inl_current, self.K)

            t = self.t + self.R.dot(t)
            R = R.dot(self.R)

            inliers = inl_current

        if pts_current != []:
            # Triangulate new points
            new_landmarks = cv2.triangulatePoints(self.K@np.hstack((self.R, self.t)), self.K@np.hstack((R, t)), pts_last.squeeze().T, pts_current.squeeze().T).T
            new_landmarks = new_landmarks[:, :3] / new_landmarks[:, 3, None]
            self.matched_landmarks = np.vstack((self.matched_landmarks, new_landmarks))

            # Disambiguate landmarks
            vec_last = (self.R @ self.matched_landmarks.T + self.t).T
            vec_current = (R @ self.matched_landmarks.T + t).T
            depth_last = vec_last[:, 2]
            depth_current = vec_current[:, 2]
            angle1_last = np.arctan2(vec_last[:, 1], vec_last[:, 0])
            angle1_current = np.arctan2(vec_current[:, 1], vec_current[:, 0])
            angle2_last = np.arctan2(vec_last[:, 2], vec_last[:, 0])
            angle2_current = np.arctan2(vec_current[:, 2], vec_current[:, 0])
            angle3_last = np.arctan2(vec_last[:, 1], vec_last[:, 2])
            angle3_current = np.arctan2(vec_current[:, 1], vec_current[:, 2])
            diff_angle1 = np.abs(angle1_last - angle1_current)
            diff_angle2 = np.abs(angle2_last - angle2_current)
            diff_angle3 = np.abs(angle3_last - angle3_current)
            dist = np.linalg.norm(vec_current - vec_last, axis=1)
            valid_indices = (depth_last > 0) & (depth_current > 0) & (diff_angle1 < 0.1) & (diff_angle2 < 0.1) & (diff_angle3 < 0.1) & (dist > 0.1) & (dist < 10)

            self.matched_landmarks = self.matched_landmarks[valid_indices]
            self.matched_descriptor = [self.matched_descriptor[i] for i in range(len(valid_indices)) if valid_indices[i]]

        self.R = R
        self.t = t
        # self.t = self.t + self.R.dot(t)
        # self.R = R.dot(self.R)
        if inliers is not None:
            self.num_pts = len(inliers)
        else:
            self.num_pts = 0
    