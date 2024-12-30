import cv2
import numpy as np

class VisualOdometryPipeLine:
    def __init__(self,K):
        self.sift = cv2.SIFT_create()   # Simple SIFT detector
        self.feature_ratio = 0.8        # Ratio for feature matching
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
    
    def get_World_Camera_Pose(self,R,t):
        Rnew = R.T
        tnew = -Rnew.dot(t)
        return Rnew, tnew
    
    def get_Camera_World_Pose(self,R,t):
        T = np.vstack((np.hstack((R,t)),np.array([0,0,0,1])))
        Tinverse = np.linalg.inv(T)
        Rnew = Tinverse[:3,:3]
        tnew = Tinverse[:3,3, None]
        return Rnew, tnew

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
        if len(good_old_matches) < 1000:
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

        # Get transformation matrix
        R_last, t_last = self.get_World_Camera_Pose(self.R,self.t)
        R_current, t_current = self.get_World_Camera_Pose(new_R,new_t)

        # Triangulate new points
        matched_landmarks = cv2.triangulatePoints(self.K@np.hstack((R_last,t_last)),self.K@np.hstack((R_current, t_current)), inl_last.squeeze().T, inl_current.squeeze().T).T
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

        if len(pts_current_landmarks) >= 10:
            sucess, R, t, inliers = cv2.solvePnPRansac(self.matched_landmarks, pts_current_landmarks, self.K, None, flags=cv2.SOLVEPNP_EPNP, confidence=0.9999 ,reprojectionError=1)
            R, _ = cv2.Rodrigues(R)

            R, t = self.get_Camera_World_Pose(R,t)

            if sucess and len(inliers) > 4:
                # Filter only inliers

                filter = True
                if filter:
                    squeezed_inliers = inliers.squeeze()
                    new_descriptors = np.arange(len(self.matched_landmarks),len(self.matched_descriptor))

                    self.matched_landmarks = self.matched_landmarks[np.int16(squeezed_inliers)]

                    all_inliers = np.hstack((squeezed_inliers,new_descriptors))
                    self.matched_descriptor = np.float32(self.matched_descriptor)[np.int16(all_inliers)]
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
            # Get transformation matrix
            R_current, t_current = self.get_World_Camera_Pose(R,t)
            R_last, t_last = self.get_World_Camera_Pose(self.R,self.t)

            # Triangulate new points
            new_landmarks = cv2.triangulatePoints(self.K@np.hstack((R_last, t_last)), self.K@np.hstack((R_current, t_current)), pts_last.squeeze().T, pts_current.squeeze().T).T
            new_landmarks = new_landmarks[:, :3] / new_landmarks[:, 3, None]

            # Disambiguate landmarks
            vec_last = (R_last @ new_landmarks.T + t_last).T
            vec_current = (R_current @ new_landmarks.T + t_current).T
            depth_last = vec_last[:, 2]
            depth_current = vec_current[:, 2]

            # Distance landmarks to camera
            dist = np.linalg.norm(new_landmarks - t_current.T, axis=1)

            # Filter with reprojection error
            proj_last = vec_last@self.K.T
            proj_last = proj_last[:,:2]/proj_last[:,2,None]
            error_last = np.linalg.norm(proj_last-pts_last.squeeze(),axis=1)
            proj_current = vec_current@self.K.T
            proj_current = proj_current[:,:2]/proj_current[:,2,None]
            error_current = np.linalg.norm(proj_current-pts_current.squeeze(),axis=1)

            valid_indices = (depth_last > 0) & (depth_current > 0) & (error_current < 1) & (error_last < 1) & (dist < 100)

            new_landmarks = new_landmarks[valid_indices]

            valid_indices_descriptor = np.hstack((np.ones((len(self.matched_landmarks),),dtype=bool),valid_indices))
            self.matched_descriptor = [self.matched_descriptor[i] for i in range(len(valid_indices_descriptor)) if valid_indices_descriptor[i]]


            self.matched_landmarks = np.vstack((self.matched_landmarks, new_landmarks))


        self.R = R
        self.t = t
        # self.t = self.t + self.R.dot(t)
        # self.R = R.dot(self.R)
        if inliers is not None:
            self.num_pts = len(inliers)
        else:
            self.num_pts = 0
    