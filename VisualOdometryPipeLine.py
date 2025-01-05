import cv2
import numpy as np

class VisualOdometryPipeLine:
    """
    Visual odometry pipeline for monocular camera system. 
    Locally consistent pose estimation with feature tracking and matching.

    Args:
        K (np.array): Camera matrix
        options (dict): Dictionary with options for the pipeline

    Attributes:
        options (dict): Dictionary with options for the pipeline
        sift (cv2.SIFT): SIFT detector
        matcher (cv2.BFMatcher): Matcher for feature matching
        K (np.array): Camera matrix
        num_pts (list): List to store number of tracked landmarks
        transforms (list): List to store camera transformations
        matched_keypoints (list): List to store matched keypoints
        matched_landmarks (list): List to store matched landmarks
        matched_descriptors (list): List to store matched descriptors
        potential_keys (list): List to store potential keypoints
        potential_first_keys (list): List to store first pixel coords of potential keypoints
        potential_transforms (list): List to store the camera transformation index of potential keypoints
        potential_descriptors (list): List to store the descriptors of potential keypoints
        potential_frame (np.array): Last frame with potential_keys
        inlier_pts_current (np.array): Current frame inlier points (RANSAC)
        outlier_pts_current (np.array): Current frame outlier points (RANSAC)
        num_tracked_landmarks_list (list): Number of tracked landmarks list (inliers of RANSAC)
    """

    def __init__(self, K, options):
        self.options = options
        self.sift = cv2.SIFT_create()               # SIFT detector
        self.matcher = cv2.BFMatcher()              # Matcher for feature matching
        self.K = K                                  # Camera matrix
        
        self.num_pts = []                           # List to store number of tracked landmarks
        self.transforms = []                        # List to store camera transformations

        R_CW = np.eye(3)                            # Rotation matrix
        t_CW = np.zeros((3, 1))                     # Translation vector
        self.transforms.append((R_CW, t_CW))        # Append initial camera pose

        self.matched_keypoints = []                 # List to store matched keypoints
        self.matched_landmarks = []                 # List to store matched landmarks
        self.matched_descriptors = []               # List to store matched descriptors

        self.potential_keys = []                    # List to store potential keypoints
        self.potential_first_keys = []              # List to store first pixel coords of potential keypoints
        self.potential_transforms = []              # List to store the camera transformation index of potential keypoints
        self.potential_descriptors = []             # List to store the descriptors of potential keypoints
        self.potential_frame = None                 # Last frame with potential_keys

        self.inlier_pts_current = None              # Current frame inlier points (RANSAC)
        self.outlier_pts_current = None             # Current frame outlier points (RANSAC)
        self.num_tracked_landmarks_list = []        # Number of tracked landmarks list (inliers of RANSAC) for the last 20 frames
    

    def invert_transform(self,R,t):
        """
        Inversion of transformation matrix. Short form of the inverse of a rigid body transformation matrix.

        Args:
            R (np.array): Rotation matrix
            t (np.array): Translation vector

        Returns:
            Rnew (np.array): Inverted rotation matrix
            tnew (np.array): Inverted translation vector
        """

        Rnew = R.T
        tnew = -Rnew @ t
        return Rnew, tnew
    

    def filter_potential(self, mask):
        """
        Filter potential keypoints based on mask. 

        Args:
            mask (np.array): Mask to filter potential keypoints
        """

        self.potential_keys = self.potential_keys[mask,:]
        self.potential_first_keys = self.potential_first_keys[mask,:]
        self.potential_transforms = self.potential_transforms[mask,:]
        self.potential_descriptors = self.potential_descriptors[mask,:]


    def filter_landmarks(self, mask):
        """
        Filter landmarks based on mask.

        Args:
            mask (np.array): Mask to filter landmarks
        """

        self.matched_landmarks = self.matched_landmarks[mask,:]
        self.matched_keypoints = self.matched_keypoints[mask,:]
        self.matched_descriptors = self.matched_descriptors[mask,:]


    def triangulate_landmarks(self, R_current_CW, t_current_CW):
        """
        Triangulate landmarks based on current and current and past camera poses for 
        long enough tracked keypoints with a sufficient baseline.

        Args:
            R_current_CW (np.array): Current camera rotation matrix
            t_current_CW (np.array): Current camera translation vector
        """

        def check_baseline(first_key, current_key, R_past_CW, R_current_CW):
            """
            Check if baseline between two camera poses is too small for meaningful triangulation.

            Args:
                first_key (np.array): First tracked keypoint in pixel coordinates
                current_key (np.array): Current keypoint in pixel coordinates
                R_past_CW (np.array): Rotation matrix of first tracked camera pose
                R_current_CW (np.array): Rotation matrix of current camera pose

            Returns:
                bool: True if baseline is too small, False otherwise
            """

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
            """
            Disambiguate landmark based on depth consistency.

            Args:
                R_current_CW (np.array): Current camera rotation matrix
                t_current_CW (np.array): Current camera translation vector
                R_last_CW (np.array): Rotation matrix of last camera pose
                t_last_CW (np.array): Translation vector of last camera pose
                landmark (np.array): Landmark to disambiguate

            Returns:
                bool: True if landmark is disambiguated, False otherwise
            """

            ray_current = R_current_CW @ landmark + t_current_CW
            ray_past = R_last_CW @ landmark + t_last_CW
            z_current_C = ray_current[2]
            z_past_C = ray_past[2]
            return z_current_C > self.options['min_dist_landmarks'] and z_past_C > self.options['min_dist_landmarks'] and z_current_C < self.options['max_dist_landmarks'] and z_past_C < self.options['max_dist_landmarks']

        R_current_WC, t_current_WC = self.invert_transform(R_current_CW, t_current_CW)
        too_short_baseline = np.zeros((self.potential_keys.shape[0],), dtype=bool)

        for i in range(self.potential_keys.shape[0]):
            if len(self.transforms) > 1:
                if len(self.transforms)-self.potential_transforms[i] <= self.options['min_baseline_frames']:
                    too_short_baseline[i] = True
                    continue

            R_past_CW, t_past_CW = self.transforms[int(self.potential_transforms[i])]

            if check_baseline(self.potential_first_keys[i,:], self.potential_keys[i,:], R_past_CW, R_current_CW):
                too_short_baseline[i] = True
                continue

            R_past_WC, t_past_WC = self.invert_transform(R_past_CW, t_past_CW)

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

        
    def initial_feature_matching(self, img0, img1):
        """
        Initial feature matching between two bootstrap frames.

        Args:
            img0 (np.array): First frame
            img1 (np.array): Second bootstrap frame
        """

        def ratio_test(matches):
            good_matches = []
            for m,n in matches:
                if m.distance < self.options['feature_ratio']*n.distance:
                    good_matches.append(m)
            
            return good_matches

        keypoints0, descriptors0 = self.sift.detectAndCompute(img0, None)
        keypoints1, descriptors1 = self.sift.detectAndCompute(img1, None)

        matches = self.matcher.knnMatch(descriptors0, descriptors1,k=2)
        good_matches = ratio_test(matches)

        pts0 = np.float32([keypoints0[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        pts1 = np.float32([keypoints1[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

        if len(good_matches) > 0:
            if self.potential_frame is None:
                self.potential_keys = pts1
                self.potential_first_keys = pts0
                self.potential_descriptors = np.float32([descriptors1[m.trainIdx] for m in good_matches])
                self.potential_transforms = np.ones((len(pts1), 1)) * (len(self.transforms)-1)  
            else:
                self.potential_keys = np.append(self.potential_keys, pts1, axis=0)
                self.potential_first_keys = np.append(self.potential_first_keys, pts0, axis=0)
                self.potential_descriptors = np.append(self.potential_descriptors, np.float32([descriptors1[m.trainIdx] for m in good_matches]), axis=0)
                self.potential_transforms = np.append(self.potential_transforms,(np.ones((len(pts1), 1)) * (len(self.transforms)-1)), axis=0)
            
    def feature_adding(self, img):
        """
        Adding new features duing continuous operation. Use properly distanced features to avoid duplicates.

        Args:
            img (np.array): Current frame
        """

        pts = cv2.goodFeaturesToTrack(img, maxCorners=self.options['feature_max_corners'], qualityLevel=self.options['feature_quality_level'], minDistance=self.options['feature_min_dist'], blockSize=self.options['feature_block_size'],useHarrisDetector=self.options['feature_use_harris'], mask=None).squeeze()
        keypts = cv2.KeyPoint.convert(pts)
        _, desc = self.sift.compute(img, keypts)
        
        valid_dist = np.array([np.all(np.linalg.norm(pts[i,:] - self.potential_keys, axis=1) > self.options['feature_min_dist']) for i in range(pts.shape[0])])
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

    
    def feature_tracking(self, img):
        """
        Feature tracking with KLT. Functions tracks matched landmark keypoints and potential keypoints.

        Args:
            img (np.array): Current frame
        """

        klt_params = dict(winSize=self.options['winSize'],maxLevel=self.options['maxLevel'],criteria=self.options['criteria'])

        matched_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.potential_frame, img, self.matched_keypoints, None, **klt_params)
        tracked = (status == 1).squeeze()
        self.matched_keypoints = matched_pts[tracked]
        self.matched_landmarks = self.matched_landmarks[tracked]
        self.matched_descriptors = self.matched_descriptors[tracked]

        if self.potential_keys.shape[0] > 1:
            potential_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.potential_frame, img, self.potential_keys, None, **klt_params)
            tracked = (status == 1).squeeze()
            self.potential_keys = potential_pts
            self.filter_potential(tracked)
            

    def non_lin_refine_pose(self, R_WC, t_WC, landmarks, keypoints, K):
        """
        Non-linear refinement of camera pose using triangulated landmarks and keypoints.

        Args:
            R_WC (np.array): Rotation matrix of camera pose
            t_WC (np.array): Translation vector of camera pose
            landmarks (np.array): Landmarks 
            keypoints (np.array): Keypoints
            K (np.array): Camera matrix

        Returns:
            R_refined (np.array): Refined rotation matrix
            t_refined (np.array): Refined translation vector
        """
        R_WC_vec = cv2.Rodrigues(R_WC)[0]
        R_refined, t_refined = cv2.solvePnPRefineLM(landmarks, keypoints, K, np.zeros(4), R_WC_vec, t_WC, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, self.options['non_lin_refinement_max_iter'], self.options['non_lin_refinement_eps']))
        return R_refined, t_refined


    def initialization(self, img0, img1):
        """
        Initialization of the pipeline with two bootstrap frames.
        (1) Initial feature matching
        (2) essential matrix estimation
        (3) camera pose estimation 
        (4) triangulation of first landmarks

        Args:
            img0 (np.array): First bootstrap frame
            img1 (np.array): Second bootstrap frame
        """

        self.initial_feature_matching(img0, img1)

        E, ransac_mask = cv2.findEssentialMat(self.potential_first_keys, self.potential_keys, self.K, method=cv2.RANSAC, prob=0.99, threshold=1)

        inliers = ransac_mask.ravel() == 1
        self.filter_potential(inliers)

        _, R_current_CW, t_current_CW,_ = cv2.recoverPose(E, self.potential_first_keys, self.potential_keys, self.K)

        t_current_CW *= np.sign(t_current_CW[2]) # Make sure t is in front of the camera

        self.triangulate_landmarks(R_current_CW, t_current_CW)
       
        self.transforms.append((R_current_CW, t_current_CW))
        self.num_pts = [sum(inliers)]
        self.potential_frame = img1
    

    def continuous_operation(self, img):
        """
        Continuous operation of the pipeline. After initialization:
        (1) Feature tracking
        (2) Pose estimation using PnP RANSAC
        (3) Triangulation of new landmarks
        (4) Feature adding

        Args:
            img (np.array): Current frame
        """

        self.feature_tracking(img)

        success = False

        if len(self.matched_keypoints) >= 8:
            success, R_WC, t_WC, inliers = cv2.solvePnPRansac(self.matched_landmarks, self.matched_keypoints, self.K, np.zeros(4), flags=cv2.SOLVEPNP_P3P, confidence=self.options['PnP_conf'], reprojectionError=self.options['PnP_error'],iterationsCount=self.options['PnP_iterations'])

            R_WC = cv2.Rodrigues(R_WC)[0]

            R_CW, t_CW = self.invert_transform(R_WC, t_WC)

            if self.options['non_lin_refinement']:
                R_WC_vec, t_WC = self.non_lin_refine_pose(R_WC, t_WC, self.matched_landmarks, self.matched_keypoints, self.K)
                R_WC = cv2.Rodrigues(R_WC_vec)[0]
                R_CW, t_CW = self.invert_transform(R_WC, t_WC)
            
            if success:
                mask = np.isin(np.arange(len(self.matched_landmarks)), inliers.squeeze()).astype(np.bool)

                self.outlier_pts_current = self.matched_keypoints[~mask]
                self.inlier_pts_current = self.matched_keypoints[mask]

                if self.options['discard_outliers']:
                    self.filter_landmarks(mask)

        if not success:
            raise Exception("PnP Failed")

        if len(self.num_tracked_landmarks_list) < 20:
            self.num_tracked_landmarks_list.append(len(self.inlier_pts_current))
        elif len(self.num_tracked_landmarks_list) == 20:
            self.num_tracked_landmarks_list.pop(0)
            self.num_tracked_landmarks_list.append(len(self.inlier_pts_current))

        if self.potential_keys.shape[0] > 1:
            self.triangulate_landmarks(R_CW, t_CW)

        self.feature_adding(img)

        self.transforms.append((R_CW, t_CW))
        self.num_pts.append(len(inliers))        
        self.potential_frame = img
    