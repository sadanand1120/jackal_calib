"""
Jackal frames description: Assume you are watching the robot head on, with the robot facing you.
- Frame 1: World Coordinate System (WCS) / base_link
    At the robot body center but on ground (though attached to robot, so moves with robot)
    +X: Forward of the robot (towards you)
    +Y: Left of the robot (to your right)
    +Z: Upwards (away from the ground, to the sky)
- Frame 2: Intermediate frame
    At the camera lens center, but axes aligned to WCS
    +X: Forward of the robot (towards you)
    +Y: Left of the robot (to your right)
    +Z: Upwards (away from the ground, to the sky)
- Frame 3: Camera Coordinate System (CCS)
    At the camera lens center, with conventional camera axes
    +X: Right of the camera (to your left)
    +Y: Down of the camera (towards the ground, may not be straight down)
    +Z: Coming out of camera along the lens axis (towards you)
- Frame 4: Pixel Coordinate System (PCS)
    At the top left (from your perspective) corner of the image, with conventional image axes
    +X: To your right (left of the image)
    +Y: Downwards

The equation goes like: PCS = M_int * M_perp * M_ext * WCS
    - M_int (3 x 3): Intrinsic matrix, gets PCS from CCS
    - M_perp (3 x 4): Perspective matrix, 3D to 2D projection
    - M_ext (4 x 4): Extrinsic matrix, gets CCS from WCS

Additional info:
    - M_int is constant for a given camera
    - You should NOT directly do M_int multiplication as shown above, but instead use cv2.projectPoints as it takes into account the distortion coefficients as well
    - Sometimes, M_perp * M_ext is combined into a single matrix, called the extrinsic matrix, where then it is (3 x 4) extrinsic matrix
    - (Corrected) lidar frame is such that it only has z displacement from base_link
"""

import cv2
import numpy as np
import os
import yaml
import math
from copy import deepcopy


class JackalCameraCalibration:
    def __init__(self, intrinsics_filepath=None, extrinsics_filepath=None):
        if intrinsics_filepath is None:
            self.intrinsics_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"params/zed_left_rect_intrinsics.yaml")
        else:
            self.intrinsics_filepath = intrinsics_filepath
        if extrinsics_filepath is None:
            self.extrinsics_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "params/baselink_to_zed_left_extrinsics.yaml")
        else:
            self.extrinsics_filepath = extrinsics_filepath
        self.intrinsics_dict = None
        self.extrinsics_dict = None
        self.img_height = None
        self.img_width = None
        self.load_params()

    def load_params(self):
        with open(self.intrinsics_filepath, 'r') as f:
            self.intrinsics_dict = yaml.safe_load(f)
        self.intrinsics_dict['camera_matrix'] = np.array(self.intrinsics_dict['camera_matrix']).reshape((3, 3))
        self.intrinsics_dict['dist_coeffs'] = np.array(self.intrinsics_dict['dist_coeffs']).reshape((1, 5)).squeeze()
        if 'rvecs' in self.intrinsics_dict:
            self.intrinsics_dict['rvecs'] = np.array(self.intrinsics_dict['rvecs']).reshape((-1, 3))
        if 'tvecs' in self.intrinsics_dict:
            self.intrinsics_dict['tvecs'] = np.array(self.intrinsics_dict['tvecs']).reshape((-1, 3))
        self.img_height = self.intrinsics_dict['height']
        self.img_width = self.intrinsics_dict['width']
        with open(self.extrinsics_filepath, 'r') as f:
            self.extrinsics_dict = yaml.safe_load(f)

    def get_M_ext(self):
        """
        Returns the extrinsic matrix (4 x 4) that transforms from WCS to CCS        
        """
        # Transforms from WCS to Intermediate frame
        T1 = JackalCameraCalibration.get_std_trans(cx=self.extrinsics_dict['T12']['T1']['X'] / 100,
                                                   cy=self.extrinsics_dict['T12']['T1']['Y'] / 100,
                                                   cz=self.extrinsics_dict['T12']['T1']['Z'] / 100)

        # Transforms from Intermediate frame to CCS
        T2 = JackalCameraCalibration.get_std_rot(axis=self.extrinsics_dict['T23']['R1']['axis'],
                                                 alpha=np.deg2rad(self.extrinsics_dict['T23']['R1']['alpha']))
        T3 = np.array(self.extrinsics_dict['T23']['R2'])
        T4 = JackalCameraCalibration.get_std_rot(axis=self.extrinsics_dict['T23']['R3']['axis'],
                                                 alpha=np.deg2rad(self.extrinsics_dict['T23']['R3']['alpha']))
        T5 = JackalCameraCalibration.get_std_rot(axis=self.extrinsics_dict['T23']['R4']['axis'],
                                                 alpha=np.deg2rad(self.extrinsics_dict['T23']['R4']['alpha']))
        return T5 @ T4 @ T3 @ T2 @ T1

    @staticmethod
    def general_project_A_to_B(inp, AtoBmat):
        """
        Project inp from A frame to B
        inp: (N x 3) array of points in A frame
        AtoBmat: (4 x 4) transformation matrix from A to B
        Returns: (N x 3) array of points in B frame
        """
        inp = np.array(inp).astype(np.float64)
        inp_4d = JackalCameraCalibration.get_homo_from_ordinary(inp)
        out_4d = (AtoBmat @ inp_4d.T).T
        return JackalCameraCalibration.get_ordinary_from_homo(out_4d)

    def projectWCStoPCS(self, wcs_coords, mode="skip"):
        """
        Projects set of points in WCS to PCS.
        wcs_coords: (N x 3) array of points in WCS
        Returns: (N x 2) array of points in PCS, in FOV of image, and a mask to indicate which ccs locs were preserved during pixel FOV bounding
        """
        ccs_coords = self.projectWCStoCCS(wcs_coords)
        return self.projectCCStoPCS(ccs_coords, mode=mode)

    def projectWCStoCCS(self, wcs_coords):
        """
        Projects set of points in WCS to CCS.
        wcs_coords: (N x 3) array of points in WCS
        Returns: (N x 3) array of points in CCS
        """
        M_ext = self.get_M_ext()
        return JackalCameraCalibration.general_project_A_to_B(wcs_coords, M_ext)

    def projectCCStoPCS(self, ccs_coords, mode="skip"):
        """
        Projects set of points in CCS to PCS.
        ccs_coords: (N x 3) array of points in CCS
        Returns: (N x 2) array of points in PCS, in FOV of image, and a mask to indicate which ccs locs were preserved during pixel FOV bounding
        """
        ccs_coords = np.array(ccs_coords).astype(np.float64)
        ccs_mask = (ccs_coords[:, 2] >= 0)  # mask to filter out points in front of camera (ie, possibly visible in image). This is important and is not taken care of by pixel bounding
        ccs_coords = ccs_coords[ccs_mask]
        if ccs_coords.shape[0] == 0 or ccs_coords is None:
            return None, None
        R = np.zeros((3, 1))
        T = np.zeros((3, 1))
        K = self.intrinsics_dict['camera_matrix']
        d = self.intrinsics_dict['dist_coeffs']
        image_points, _ = cv2.projectPoints(ccs_coords, R, T, K, d)
        image_points = np.swapaxes(image_points, 0, 1).astype(np.int32)

        # Additional processing for the opencv issue https://github.com/opencv/opencv/issues/17768
        image_points_nodist, _ = cv2.projectPoints(ccs_coords, R, T, K, np.zeros((5, 1)).squeeze())
        image_points_nodist = np.swapaxes(image_points_nodist, 0, 1).astype(np.int32)
        _, num_points, _ = image_points_nodist.shape
        safe_image_points_mask = JackalCameraCalibration.get_safe_projs(d, ccs_coords)
        safe_image_points = np.zeros((1, num_points, 2))
        safe_image_points[0, safe_image_points_mask, :] = image_points[0, safe_image_points_mask, :]
        unsafe_image_points_mask = np.delete(np.arange(0, num_points), safe_image_points_mask)
        safe_image_points[0, unsafe_image_points_mask, :] = image_points_nodist[0, unsafe_image_points_mask, :]
        image_points = safe_image_points.astype(np.int32).squeeze().reshape((-1, 2))

        image_points, pcs_mask = JackalCameraCalibration.to_image_fov_bounds(image_points, self.img_width, self.img_height, mode=mode)
        unified_mask = deepcopy(ccs_mask)
        unified_mask[ccs_mask] = pcs_mask
        return image_points, unified_mask

    def projectCCStoWCS(self, ccs_coords):
        """
        Projects set of points in CCS to WCS.
        ccs_coords: (N x 3) array of points in CCS
        Returns: (N x 3) array of points in WCS
        """
        M_ext_inv = np.linalg.inv(self.get_M_ext())
        return JackalCameraCalibration.general_project_A_to_B(ccs_coords, M_ext_inv)

    def projectPCStoWCSground(self, pcs_coords, apply_dist=True, mode="skip"):
        """
        Projects set of points in PCS to WCS, assuming they are on the ground plane (z=0) in WCS.
        pcs_coords: (N x 2) array of points in PCS
        Returns: (N x 3) array of points in WCS, and a mask to indicate which pixel locs were kept during FOV bounding
        """
        pcs_coords = np.array(pcs_coords).astype(np.float64)
        K = self.intrinsics_dict['camera_matrix']
        d = self.intrinsics_dict['dist_coeffs']
        R = np.eye(3)
        undistorted_pcs_coords = cv2.undistortPoints(pcs_coords.reshape(1, -1, 2), K, d, R=R, P=K)
        undistorted_pcs_coords = np.swapaxes(undistorted_pcs_coords, 0, 1).squeeze().reshape((-1, 2))
        undistorted_pcs_coords, pcs_mask = JackalCameraCalibration.to_image_fov_bounds(undistorted_pcs_coords, self.img_width, self.img_height, mode=mode)
        if apply_dist:
            pcs_coords = undistorted_pcs_coords
        pcs_coords_3d = JackalCameraCalibration.get_homo_from_ordinary(pcs_coords)
        M_ext = self.get_M_ext()
        M_ext_short = M_ext[:, [0, 1, 3]][:-1, :].reshape((3, 3))
        H = K @ M_ext_short  # 3x3
        H_inv = np.linalg.inv(H)
        wcs_coords_3d = (H_inv @ pcs_coords_3d.T).T
        wcs_coords = JackalCameraCalibration.get_ordinary_from_homo(wcs_coords_3d)
        zeros = np.zeros((wcs_coords.shape[0], 1))
        wcs_coords_full = np.hstack([wcs_coords, zeros]).reshape((wcs_coords.shape[0], 3))
        ccs_coords_full = self.projectWCStoCCS(wcs_coords_full)
        ccs_mask = (ccs_coords_full[:, 2] >= 0)  # this ccs_mask calculation is to cross-check if projected wcs points make sense or not
        unified_mask = deepcopy(pcs_mask)
        unified_mask[pcs_mask] = ccs_mask
        return wcs_coords_full[ccs_mask], unified_mask

    @staticmethod
    def to_image_fov_bounds(pixels, width, height, mode="skip"):
        """
        Returns the pixels coords in image bounds and a mask for indicating which pixels were kept (ie, relevant for skip mode)
        """
        if mode == "skip":
            mask = (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
            return pixels[mask], mask
        elif mode == "clip":
            pixels[:, 0] = np.clip(pixels[:, 0], 0, width - 1)
            pixels[:, 1] = np.clip(pixels[:, 1], 0, height - 1)
            return pixels, np.ones(pixels.shape[0], dtype=bool)
        elif mode == "none":
            return pixels, np.ones(pixels.shape[0], dtype=bool)
        else:
            raise ValueError("Unknown fov bounds mode!")

    @staticmethod
    def get_safe_projs(distCoeffs, obj_pts):
        """
        Vectorize this for more efficiency (Slow on large point clouds)
        Custom point projection function to fix OpenCV distortion issue for out of bound points:
        https://github.com/opencv/opencv/issues/17768
        """
        # Define a list of booleans to denote if a variable is safe
        obj_pts_safe = np.array([True] * len(obj_pts))

        # First step is to get the location of the points relative to the camera.
        obj_pts_rel_cam = deepcopy(obj_pts)

        # Define the homogenous coordiantes
        x_homo_vals = (obj_pts_rel_cam[:, 0] / obj_pts_rel_cam[:, 2]).astype(np.complex128)
        y_homo_vals = (obj_pts_rel_cam[:, 1] / obj_pts_rel_cam[:, 2]).astype(np.complex128)

        # Define the distortion terms, and vectorize calculating of powers of x_homo_vals
        #   and y_homo_vals
        k1, k2, p1, p2, k3 = distCoeffs
        x_homo_vals_2 = np.power(x_homo_vals, 2)
        y_homo_vals_2 = np.power(y_homo_vals, 2)
        x_homo_vals_4 = np.power(x_homo_vals, 4)
        y_homo_vals_4 = np.power(y_homo_vals, 4)
        x_homo_vals_6 = np.power(x_homo_vals, 6)
        y_homo_vals_6 = np.power(y_homo_vals, 6)

        # Find the bounds on the x_homo coordinate to ensure it is closer than the
        #   inflection point of x_proj as a function of x_homo
        x_homo_min = np.full(x_homo_vals.shape, np.inf)
        x_homo_max = np.full(x_homo_vals.shape, -np.inf)
        for i in range(len(y_homo_vals)):
            # Expanded projection function polynomial coefficients
            x_proj_coeffs = np.array([k3,
                                      0,
                                      k2 + 3 * k3 * y_homo_vals_2[i],
                                      0,
                                      k1 + 2 * k2 * y_homo_vals_2[i] + 3 * k3 * y_homo_vals_4[i],
                                      3 * p2,
                                      1 + k1 * y_homo_vals_2[i] + k2 * y_homo_vals_4[i] + k3 * y_homo_vals_6[i] + 2 * p1 * y_homo_vals[i],
                                      p2 * y_homo_vals_2[i]])

            # Projection function derivative polynomial coefficients
            x_proj_der_coeffs = np.polyder(x_proj_coeffs)

            # Find the root of the derivative
            roots = np.roots(x_proj_der_coeffs)

            # Get the real roots
            # Approximation of real[np.where(np.isreal(roots))]
            real_roots = np.real(roots[np.where(np.abs(np.imag(roots)) < 1e-10)])

            for real_root in real_roots:
                x_homo_min[i] = np.minimum(x_homo_min[i], real_root)
                x_homo_max[i] = np.maximum(x_homo_max[i], real_root)

        # Check that the x_homo values are within the bounds
        obj_pts_safe *= np.where(x_homo_vals > x_homo_min, True, False)
        obj_pts_safe *= np.where(x_homo_vals < x_homo_max, True, False)

        # Find the bounds on the y_homo coordinate to ensure it is closer than the
        #   inflection point of y_proj as a function of y_homo
        y_homo_min = np.full(y_homo_vals.shape, np.inf)
        y_homo_max = np.full(y_homo_vals.shape, -np.inf)
        for i in range(len(x_homo_vals)):
            # Expanded projection function polynomial coefficients
            y_proj_coeffs = np.array([k3,
                                      0,
                                      k2 + 3 * k3 * x_homo_vals_2[i],
                                      0,
                                      k1 + 2 * k2 * x_homo_vals_2[i] + 3 * k3 * x_homo_vals_4[i],
                                      3 * p1,
                                      1 + k1 * x_homo_vals_2[i] + k2 * x_homo_vals_4[i] + k3 * x_homo_vals_6[i] + 2 * p2 * x_homo_vals[i],
                                      p1 * x_homo_vals_2[i]])

            # Projection function derivative polynomial coefficients
            y_proj_der_coeffs = np.polyder(y_proj_coeffs)

            # Find the root of the derivative
            roots = np.roots(y_proj_der_coeffs)

            # Get the real roots
            # Approximation of real[np.where(np.isreal(roots))]
            real_roots = np.real(roots[np.where(np.abs(np.imag(roots)) < 1e-10)])

            for real_root in real_roots:
                y_homo_min[i] = np.minimum(y_homo_min[i], real_root)
                y_homo_max[i] = np.maximum(y_homo_max[i], real_root)

        # Check that the x_homo values are within the bounds
        obj_pts_safe *= np.where(y_homo_vals > y_homo_min, True, False)
        obj_pts_safe *= np.where(y_homo_vals < y_homo_max, True, False)

        # Return the indices where obj_pts is safe to project
        return np.where(obj_pts_safe == True)[0]

    @staticmethod
    def get_ordinary_from_homo(points_higherD):
        # Scales so that last coord is 1 and then removes last coord
        points_higherD = points_higherD / points_higherD[:, -1].reshape(-1, 1)  # scale by the last coord
        return points_higherD[:, :-1]

    @staticmethod
    def get_homo_from_ordinary(points_lowerD):
        # Append 1 to each point
        ones = np.ones((points_lowerD.shape[0], 1))  # create a column of ones
        return np.hstack([points_lowerD, ones])  # append the ones column to points

    @staticmethod
    def get_std_trans(cx=0, cy=0, cz=0):
        """
        cx, cy, cz are the coords of O_M wrt O_F when expressed in F
        Multiplication goes like M_coords = T * F_coords
        """
        mat = [
            [1, 0, 0, -cx],
            [0, 1, 0, -cy],
            [0, 0, 1, -cz],
            [0, 0, 0, 1]
        ]
        return np.array(mat)

    @staticmethod
    def get_std_rot(axis, alpha):
        """
        axis is either "X", "Y", or "Z" axis of F and alpha is positive acc to right hand thumb rule dirn
        Multiplication goes like M_coords = T * F_coords
        """
        if axis == "X":
            mat = [
                [1, 0, 0, 0],
                [0, math.cos(alpha), math.sin(alpha), 0],
                [0, -math.sin(alpha), math.cos(alpha), 0],
                [0, 0, 0, 1]
            ]
        elif axis == "Y":
            mat = [
                [math.cos(alpha), 0, -math.sin(alpha), 0],
                [0, 1, 0, 0],
                [math.sin(alpha), 0, math.cos(alpha), 0],
                [0, 0, 0, 1]
            ]
        elif axis == "Z":
            mat = [
                [math.cos(alpha), math.sin(alpha), 0, 0],
                [-math.sin(alpha), math.cos(alpha), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]
        else:
            raise ValueError("Invalid axis!")
        return np.array(mat)

    @staticmethod
    def pretty_print(v, precision=4, suppress_small=True):
        print(np.array_str(v, precision=precision, suppress_small=suppress_small))


if __name__ == "__main__":
    c = JackalCameraCalibration()

    gt_wcs_coords = np.array([
        [359, 108, 0],
        [274, -69, 29],
    ])
    gt_wcs_coords = gt_wcs_coords / 100
    gt_pcs_coords = np.array([
        [331, 397],
        [631, 372],
    ])

    est_pcs_coords, _ = c.projectWCStoPCS(gt_wcs_coords)

    _3dto2d_error = est_pcs_coords - gt_pcs_coords
    print("--------------------- 3D to 2D error (pixels):")
    JackalCameraCalibration.pretty_print(_3dto2d_error)
    print(np.mean(np.abs(_3dto2d_error)))
    print("-------------------------------------------------------")
