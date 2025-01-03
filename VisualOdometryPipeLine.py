import cv2
import numpy as np
from copy import deepcopy
from plotting_tools import reprojection_error_plot


class VisualOdometryPipeLine:
    def __init__(self, K, options):
        self.options = options
        self.sift = cv2.SIFT_create(nfeatures=500)  # Simple SIFT detector
        self.K = K                                  # Camera matrix
        self.matcher = cv2.BFMatcher()              # Matcher for feature matching

        R_CW = np.eye(3)                            # Rotation matrix
        t_CW = np.zeros((3, 1))                     # Translation vector
        
        self.num_pts = []                           # List to store number of tracked landmarks
        self.transforms = []                        # List to store camera transformations
        self.transforms.append((R_CW, t_CW)) 

        self.matched_keypoints = []                 # List to store matched keypoints
        self.matched_landmarks = []                 # List to store matched landmarks
        self.matched_descriptors = []               # List to store matched descriptors

        self.potential_keys = []                    # List to store potential keypoints
        self.potential_first_keys = []              # List to store first oixel coords of potential keypoints
        self.potential_transforms = []              # List to store the camera transformation index of potential keypoints
        self.potential_descriptors = []             # List to store the descriptors of potential keypoints
        self.potentail_frame = None                 # Last frame with potential_keys

        self.inlier_pts_current = None              # Current frame inlier points (RANSAC)
        self.outlier_pts_current = None             # Current frame outlier points (RANSAC)
        self.num_tracked_landmarks_list = []        # Number of tracked landmarks list (inliers of RANSAC) for the last 20 frames

        # for plotting of the reprojection error
        self.repr_error_plot = True                # Set to True to plot the reprojection error
        self.repr_error_mean = []
        self.repr_error_std = []
        self.repr_error_max = []
        self.repr_error_below_threshold = []
    
    def get_World_Camera_Pose(self,R,t):
        Rnew = R.T
        tnew = -Rnew @ t
        return Rnew, tnew
    
    def filter_potential(self, mask):
        self.potential_keys = self.potential_keys[mask,:]
        self.potential_first_keys = self.potential_first_keys[mask,:]
        self.potential_transforms = self.potential_transforms[mask,:]
        self.potential_descriptors = self.potential_descriptors[mask,:]

    def filter_landmarks(self, mask):
        self.matched_landmarks = self.matched_landmarks[mask,:]
        self.matched_keypoints = self.matched_keypoints[mask,:]
        self.matched_descriptors = self.matched_descriptors[mask,:]

    def triangulate_landmarks(self, R_current_CW, t_current_CW):

        def check_baseline(u_current, v_current, u_last, v_last):

            def get_ray(u,v):
                w = np.ones_like(u)
                vec_current = np.vstack((u,v,w))
                return vec_current / np.linalg.norm(vec_current, axis=0)
        
            vec_current = get_ray(u_current, v_current)
            vec_last = get_ray(u_last, v_last)

            cos = np.sum(vec_current.T*vec_last.T, axis=1) / (np.linalg.norm(vec_current, axis=0) * np.linalg.norm(vec_last, axis=0))
            cos = np.clip(cos, -1.0, 1.0)
            alphas = np.degrees(np.arccos(cos))
            return alphas < self.options['min_baseline_angle']
        
        # Filter keypoints with small baseline
        too_short_baseline = check_baseline(self.potential_keys[:,0], self.potential_keys[:,1], self.potential_first_keys[:,0], self.potential_first_keys[:,1])

        def disambguate_landmark(R_current_CW, t_current_CW, R_last_CW, t_last_CW, landmark):
            dist = np.linalg.norm(R_current_CW @ landmark + t_current_CW)
            z_current_C = R_current_CW @ landmark + t_current_CW
            z_last_C = R_last_CW @ landmark + t_last_CW
            return z_current_C[2] > 0 and z_last_C[2] > 0 and dist > self.options['min_dist_landmarks'] and dist < self.options['max_dist_landmarks']

        for i in range(self.potential_keys.shape[0]):
            if too_short_baseline[i]:
                continue

            # Get transform of first keypoint from transform list
            R_past_CW, t_past_CW = self.transforms[int(self.potential_transforms[i])]

            # Triangulate points
            new_landmark = cv2.triangulatePoints(
                self.K@np.hstack((R_past_CW, t_past_CW)),
                self.K@np.hstack((R_current_CW, t_current_CW)),
                self.potential_first_keys[i].reshape(-1, 1),
                self.potential_keys[i].reshape(-1, 1)
            )
            new_landmark = new_landmark[:3] / new_landmark[3]
            
            if True: #disambguate_landmark(R_current_CW, t_current_CW, R_past_CW, t_past_CW, new_landmark):
                if len(self.matched_landmarks) == 0:
                    self.matched_landmarks = new_landmark.T
                    self.matched_keypoints = self.potential_keys[i].reshape(1,2)
                    self.matched_descriptors = self.potential_descriptors[i].reshape(1,-1)
                else:
                    self.matched_landmarks = np.append(self.matched_landmarks,new_landmark.T, axis=0)
                    self.matched_keypoints = np.append(self.matched_keypoints, self.potential_keys[i].reshape(1,2), axis=0)
                    self.matched_descriptors = np.append(self.matched_descriptors, self.potential_descriptors[i].reshape(1,-1), axis=0)

        self.filter_potential(too_short_baseline)
        
    def feature_matching(self, img0, img1):

        def ratio_test(matches):
            good_matches = []
            for m,n in matches:
                if m.distance < self.options['feature_ratio']*n.distance:
                    good_matches.append(m)
                
            return good_matches
    
        # Detect keypoints and compute descriptors
        keypoints0, descriptors0 = self.sift.detectAndCompute(img0, None)
        keypoints1, descriptors1 = self.sift.detectAndCompute(img1, None)

        # Match descriptors using BFMatcher
        matches = self.matcher.knnMatch(descriptors0, descriptors1,k=2)
            
        # Apply ratio test
        good_matches = ratio_test(matches)

        # Check with allready matched and potential descriptors
        if len(self.matched_descriptors) > 0:
            joint_descriptors = np.append(self.matched_descriptors,self.potential_descriptors,axis=0)   
            used_matches = self.matcher.knnMatch(descriptors1, joint_descriptors, k=2)

            good_used_matches = ratio_test(used_matches)
        
            # Filter matches with allready matched and potential descriptors
            mask_used = np.isin(np.array(good_matches), good_used_matches)
            good_matches = np.array(good_matches)[~mask_used]

        # Extract matched keypoints
        pts0 = np.float32([keypoints0[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        pts1 = np.float32([keypoints1[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

        # Store potential features for next image
        if self.potentail_frame is None:
            self.potential_keys = pts1
            self.potential_first_keys = pts0
            self.potential_descriptors = np.float32([descriptors1[m.trainIdx] for m in good_matches])
            self.potential_transforms = np.ones((len(pts1), 1)) * (len(self.transforms)-1)  
        else:
            self.potential_keys = np.append(self.potential_keys, pts1, axis=0)
            self.potential_first_keys = np.append(self.potential_first_keys, pts0, axis=0)
            self.potential_descriptors = np.append(self.potential_descriptors, np.float32([descriptors1[m.trainIdx] for m in good_matches]), axis=0)
            self.potential_transforms = np.append(self.potential_transforms,(np.ones((len(pts1), 1)) * (len(self.transforms)-1)), axis=0)
            
        self.potentail_frame = img1
    
    def feature_tracking(self, img):

        # Track landmarks with KLT:
        matched_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.potentail_frame, img, self.matched_keypoints, None)
        tracked = (status == 1).squeeze()
        self.matched_keypoints = matched_pts[tracked]
        self.matched_landmarks = self.matched_landmarks[tracked]
        self.matched_descriptors = self.matched_descriptors[tracked]

        # Track potential features with KLT:
        potential_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.potentail_frame, img, self.potential_keys, None)
        tracked = (status == 1).squeeze()
        self.potential_keys = potential_pts
        self.filter_potential(tracked)
        self.potentail_frame = img


    def initialization(self, img0, img1):

        self.feature_matching(img0, img1)

        # Estimate Essential matrix:
        E, ransac_mask = cv2.findEssentialMat(self.potential_first_keys, self.potential_keys, self.K, method=cv2.RANSAC, prob=0.99, threshold=1)

        # Filter inliers:
        inliers = ransac_mask.ravel() == 1
        self.filter_potential(inliers)

        # Estimate relative camera pose of new second frame
        _, R_current_CW, t_current_CW,_ = cv2.recoverPose(E, self.potential_first_keys, self.potential_keys, self.K)

        self.triangulate_landmarks(R_current_CW, t_current_CW)
       
        self.transforms.append((R_current_CW, t_current_CW))

        self.num_pts = [sum(inliers)]
    

    def continuous_operation(self, img):

        self.feature_tracking(img)

        success = False

        if len(self.matched_keypoints) >= 8:
            success, R_CW, t_CW, inliers = cv2.solvePnPRansac(self.matched_landmarks, self.matched_keypoints, self.K, np.zeros(0), flags=cv2.SOLVEPNP_ITERATIVE, confidence=self.options['PnP_conf'] ,reprojectionError=self.options['PnP_error'])

            R_CW, _ = cv2.Rodrigues(R_CW)
            
            if success:
                if self.repr_error_plot:
                    inlier_indices = inliers.squeeze()  # Indices of inliers from PnP
                    # Calculate reprojection error for inliers
                    reprojected_points, _ = cv2.projectPoints(self.matched_landmarks[inlier_indices], R_CW, t_CW, self.K, None)
                    reprojection_error = np.linalg.norm(self.matched_keypoints[inlier_indices] - reprojected_points.squeeze(axis=1), axis=1)

                    mean_error = np.mean(reprojection_error)
                    std_error = np.std(reprojection_error)
                    max_error = np.max(reprojection_error)
                    below_threshold = np.sum(reprojection_error < 1.0) / len(reprojection_error) * 100

                    self.repr_error_mean.append(mean_error)
                    self.repr_error_std.append(std_error)
                    self.repr_error_max.append(max_error)
                    self.repr_error_below_threshold.append(below_threshold)
                    print(f"Reprojection Error (Mean): {mean_error:.2f} pixels")
                    print(f"Reprojection Error (Std Dev): {std_error:.2f} pixels")
                    print(f"Reprojection Error (Max): {max_error:.2f} pixels")
                    print(f"Percentage Below 1 Pixel: {below_threshold:.2f}%")
                    reprojection_error_plot(self, self.repr_error_mean, self.repr_error_std, self.repr_error_max, self.repr_error_below_threshold)

                mask = np.isin(np.arange(len(self.matched_landmarks)), inliers.squeeze()).astype(np.bool)

                # Store outliers for plots
                self.outlier_pts_current = self.matched_keypoints[~mask]
                self.inlier_pts_current = self.matched_keypoints[mask]

                self.filter_landmarks(mask)

        if not success:
            raise Exception("PnP Failed")

        if len(self.num_tracked_landmarks_list) < 20:
            self.num_tracked_landmarks_list.append(len(self.inlier_pts_current))
        elif len(self.num_tracked_landmarks_list) == 20:
            self.num_tracked_landmarks_list.pop(0)
            self.num_tracked_landmarks_list.append(len(self.inlier_pts_current))

        # Triangulate new landmarks
        self.triangulate_landmarks(R_CW, t_CW)

        # Get new features
        if self.matched_landmarks.shape[0] < 500 or self.potential_keys.shape[0] < 500:
            self.feature_matching(self.potentail_frame, img)
        else:
            self.potentail_frame = img

        self.transforms.append((R_CW, t_CW))

        self.num_pts.append(len(inliers))
        
    