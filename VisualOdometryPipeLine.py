import cv2
import numpy as np
from copy import deepcopy
from plotting_tools import reprojection_error_plot


class VisualOdometryPipeLine:
    def __init__(self, K, options):
        self.options = options
        self.sift = cv2.SIFT_create()  # Simple SIFT detector
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
    
    def invert_transform(self,R,t):
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

        def check_baseline(first_key, current_key, R_past_CW, R_current_CW):

            def get_ray(pts): 
                pts.reshape(2,1)               
                hom = np.hstack((pts, [1]))
                return hom.reshape(3,1)
            
            K_inv = np.linalg.inv(self.K)
        
            vec_current = get_ray(current_key)
            vec_past = get_ray(first_key)

            vec_current = K_inv @ vec_current

            rel_rot = (R_current_CW.T @ R_past_CW).T
            vec_past = np.matmul(rel_rot,K_inv) @ vec_past

            cos = np.sum(vec_current.T*vec_past.T, axis=1) / (np.linalg.norm(vec_current, axis=0) * np.linalg.norm(vec_past, axis=0))
            cos = np.clip(cos, -1.0, 1.0)
            alphas = np.degrees(np.arccos(cos))
            return alphas < self.options['min_baseline_angle']
        
        def disambguate_landmark(R_current_CW, t_current_CW, R_last_CW, t_last_CW, landmark):
            ray_current = R_current_CW @ landmark + t_current_CW
            ray_past = R_last_CW @ landmark + t_last_CW
            z_current_C = ray_current[2]
            z_past_C = ray_past[2]
            return z_current_C > self.options['min_dist_landmarks'] and z_past_C > self.options['min_dist_landmarks'] and z_current_C < self.options['max_dist_landmarks'] and z_past_C < self.options['max_dist_landmarks']

        def check_reproj_error(R,t,landmark,keypoint,K):
            P = self.K@np.hstack((R,t))
            reproj = P@np.vstack((landmark,[1]))
            reproj = reproj[:2]/reproj[2]
            error = np.linalg.norm(keypoint - reproj)
            return error < self.options['Reproj_threshold']


        R_current_WC, t_current_WC = self.invert_transform(R_current_CW, t_current_CW)

        too_short_baseline = np.zeros((self.potential_keys.shape[0],), dtype=bool)

        for i in range(self.potential_keys.shape[0]):
            # Get transform of first keypoint from transform list
            if len(self.transforms) > 1:
                if len(self.transforms)-self.potential_transforms[i] <= self.options['min_baseline_frames']:
                    too_short_baseline[i] = True
                    continue

            R_past_CW, t_past_CW = self.transforms[int(self.potential_transforms[i])]

            if check_baseline(self.potential_first_keys[i,:], self.potential_keys[i,:], R_past_CW, R_current_CW):
                too_short_baseline[i] = True
                continue

            R_past_WC, t_past_WC = self.invert_transform(R_past_CW, t_past_CW)

            # Triangulate points
            new_landmark = cv2.triangulatePoints(
                self.K@np.hstack((R_past_WC, t_past_WC)),
                self.K@np.hstack((R_current_WC, t_current_WC)),
                self.potential_first_keys[i].reshape(-1, 1),
                self.potential_keys[i].reshape(-1, 1)
            )
            new_landmark = new_landmark[:3] / new_landmark[3]
            
            if disambguate_landmark(R_current_WC, t_current_WC, R_past_WC, t_past_WC, new_landmark):
                if len(self.matched_landmarks) == 0:
                    self.matched_landmarks = new_landmark.T
                    self.matched_keypoints = self.potential_keys[i].reshape(1,2)
                    self.matched_descriptors = self.potential_descriptors[i].reshape(1,-1)
                else:
                    self.matched_landmarks = np.append(self.matched_landmarks,new_landmark.T, axis=0)
                    self.matched_keypoints = np.append(self.matched_keypoints, self.potential_keys[i].reshape(1,2), axis=0)
                    self.matched_descriptors = np.append(self.matched_descriptors, self.potential_descriptors[i].reshape(1,-1), axis=0)
            else:
                too_short_baseline[i] = True
        

        self.filter_potential(too_short_baseline)

    def ratio_test(self,matches):
        good_matches = []
        for m,n in matches:
            if m.distance < self.options['feature_ratio']*n.distance:
                good_matches.append(m)
            
        return good_matches
        
    def initial_feature_matching(self, img0, img1):

        # Detect keypoints and compute descriptors
        keypoints0, descriptors0 = self.sift.detectAndCompute(img0, None)
        keypoints1, descriptors1 = self.sift.detectAndCompute(img1, None)

        # Match descriptors using BFMatcher
        matches = self.matcher.knnMatch(descriptors0, descriptors1,k=2)
            
        # Apply ratio test
        good_matches = self.ratio_test(matches)

        # Extract matched keypoints
        pts0 = np.float32([keypoints0[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        pts1 = np.float32([keypoints1[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

        # Store potential features for next image
        if len(good_matches) > 0:
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

    def feature_adding(self, img):

        pts = cv2.goodFeaturesToTrack(img, maxCorners=1400, qualityLevel=0.1, minDistance=10, mask=None).squeeze()
        keypts = cv2.KeyPoint.convert(pts)
        _, desc = self.sift.compute(img, keypts)
        
        valid_dist = np.array([np.all(np.linalg.norm(pts[i,:] - self.potential_keys, axis=1) > 10) for i in range(pts.shape[0])])
        pts = pts[valid_dist]
        desc = desc[valid_dist]

        if self.potential_keys.shape[0] == 0:
            self.potential_keys = pts
            self.potential_first_keys = pts
            self.potential_descriptors = desc
            self.potential_transforms = np.ones((len(pts), 1)) * len(self.transforms)
        else:
            self.potential_keys = np.append(self.potential_keys, pts, axis=0)
            self.potential_first_keys = np.append(self.potential_first_keys, pts, axis=0)
            self.potential_descriptors = np.append(self.potential_descriptors, desc, axis=0)
            self.potential_transforms = np.append(self.potential_transforms, np.ones((len(pts), 1)) * len(self.transforms), axis=0)

        self.potentail_frame = img

    
    def feature_tracking(self, img):

        klt_parameters = {
            'winSize': (15, 15),
            'maxLevel': 10,
            'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03)
        }

        # Track landmarks with KLT:
        matched_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.potentail_frame, img, self.matched_keypoints, None, **klt_parameters)
        tracked = (status == 1).squeeze()
        self.matched_keypoints = matched_pts[tracked]
        self.matched_landmarks = self.matched_landmarks[tracked]
        self.matched_descriptors = self.matched_descriptors[tracked]


        if self.potential_keys.shape[0] > 1:
            # Track potential features with KLT:
            potential_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.potentail_frame, img, self.potential_keys, None, **klt_parameters)
            tracked = (status == 1).squeeze()
            self.potential_keys = potential_pts
            self.filter_potential(tracked)
        self.potentail_frame = img

    def non_lin_refine_pose(self, R_vec, t, landmarks, keypoints, K):
        R_refined, t_refined = cv2.solvePnPRefineLM(landmarks, keypoints, K, np.zeros(4), R_vec, t, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-2))
        return R_refined, t_refined


    def initialization(self, img0, img1):

        self.initial_feature_matching(img0, img1)

        # Estimate Essential matrix:
        E, ransac_mask = cv2.findEssentialMat(self.potential_first_keys, self.potential_keys, self.K, method=cv2.RANSAC, prob=0.99, threshold=1)

        # Filter inliers:
        inliers = ransac_mask.ravel() == 1
        self.filter_potential(inliers)

        # Estimate relative camera pose of new second frame
        _, R_current_CW, t_current_CW,_ = cv2.recoverPose(E, self.potential_first_keys, self.potential_keys, self.K)

        t_current_CW *= np.sign(t_current_CW[2])*2

        self.triangulate_landmarks(R_current_CW, t_current_CW)
       
        self.transforms.append((R_current_CW, t_current_CW))

        self.num_pts = [sum(inliers)]
    

    def continuous_operation(self, img):

        self.feature_tracking(img)

        success = False

        if len(self.matched_keypoints) >= 8:
            success, R_WC, t_WC, inliers = cv2.solvePnPRansac(self.matched_landmarks, self.matched_keypoints, self.K, np.zeros(4), flags=cv2.SOLVEPNP_P3P, confidence=self.options['PnP_conf'], reprojectionError=self.options['PnP_error'],iterationsCount=self.options['PnP_iterations'])

            # Refine pose with non-linear optimization
            if self.options['non_lin_refinement']:
                R_WC, t_WC = self.non_lin_refine_pose(R_WC, t_WC, self.matched_landmarks, self.matched_keypoints, self.K)

            R_WC = cv2.Rodrigues(R_WC)[0]

            R_CW, t_CW = self.invert_transform(R_WC, t_WC)
            
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

                # self.filter_landmarks(mask)

        if not success:
            raise Exception("PnP Failed")

        if len(self.num_tracked_landmarks_list) < 20:
            self.num_tracked_landmarks_list.append(len(self.inlier_pts_current))
        elif len(self.num_tracked_landmarks_list) == 20:
            self.num_tracked_landmarks_list.pop(0)
            self.num_tracked_landmarks_list.append(len(self.inlier_pts_current))

        # Triangulate new landmarks
        if self.potential_keys.shape[0] > 1:
            self.triangulate_landmarks(R_CW, t_CW)

        # Get new features
        self.feature_adding(img)

        self.transforms.append((R_CW, t_CW))

        self.num_pts.append(len(inliers))
        
    