import cv2
import numpy as np
np.float = np.float64  # temp fix for following import https://github.com/eric-wieser/ros_numpy/issues/37
import ros_numpy
import os
from sensor_msgs.msg import PointCloud2, CompressedImage
import matplotlib.pyplot as plt
import rospy
import time
from cv_bridge import CvBridge
from copy import deepcopy
import yaml
from scipy.interpolate import griddata
from copy import deepcopy

if __name__ == "__main__":
    from cam_calib import JackalCameraCalibration
else:
    from .cam_calib import JackalCameraCalibration


class JackalLidarCamCalibration:
    COLMAP = [(0, 0, 0.5385), (0, 0, 0.6154),
              (0, 0, 0.6923), (0, 0, 0.7692),
              (0, 0, 0.8462), (0, 0, 0.9231),
              (0, 0, 1.0000), (0, 0.0769, 1.0000),
              (0, 0.1538, 1.0000), (0, 0.2308, 1.0000),
              (0, 0.3846, 1.0000), (0, 0.4615, 1.0000),
              (0, 0.5385, 1.0000), (0, 0.6154, 1.0000),
              (0, 0.6923, 1.0000), (0, 0.7692, 1.0000),
              (0, 0.8462, 1.0000), (0, 0.9231, 1.0000),
              (0, 1.0000, 1.0000), (0.0769, 1.0000, 0.9231),
              (0.1538, 1.0000, 0.8462), (0.2308, 1.0000, 0.7692),
              (0.3077, 1.0000, 0.6923), (0.3846, 1.0000, 0.6154),
              (0.4615, 1.0000, 0.5385), (0.5385, 1.0000, 0.4615),
              (0.6154, 1.0000, 0.3846), (0.6923, 1.0000, 0.3077),
              (0.7692, 1.0000, 0.2308), (0.8462, 1.0000, 0.1538),
              (0.9231, 1.0000, 0.0769), (1.0000, 1.0000, 0),
              (1.0000, 0.9231, 0), (1.0000, 0.8462, 0),
              (1.0000, 0.7692, 0), (1.0000, 0.6923, 0),
              (1.0000, 0.6154, 0), (1.0000, 0.5385, 0),
              (1.0000, 0.4615, 0), (1.0000, 0.3846, 0),
              (1.0000, 0.3077, 0), (1.0000, 0.2308, 0),
              (1.0000, 0.1538, 0), (1.0000, 0.0769, 0),
              (1.0000, 0, 0), (0.9231, 0, 0),
              (0.8462, 0, 0), (0.7692, 0, 0),
              (0.6923, 0, 0), (0.6154, 0, 0)]

    def __init__(self, ros_flag=True, lidar_extrinsics_filepath=None, lidar_actual_extrinsics_filepath=None, cam_intrinsics_filepath=None, cam_extrinsics_filepath=None):
        self.jackal_cam_calib = JackalCameraCalibration(intrinsics_filepath=cam_intrinsics_filepath, extrinsics_filepath=cam_extrinsics_filepath)
        self.latest_img = None
        self.latest_vlp_points = None
        self.ros_flag = ros_flag
        self.img_height = self.jackal_cam_calib.img_height
        self.img_width = self.jackal_cam_calib.img_width
        if lidar_extrinsics_filepath is None:
            self.extrinsics_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "params/baselink_to_lidar_extrinsics.yaml")
        else:
            self.extrinsics_filepath = lidar_extrinsics_filepath
        if lidar_actual_extrinsics_filepath is None:
            self._actual_extrinsics_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "params/baselink_to_actual_lidar_extrinsics.yaml")
        else:
            self._actual_extrinsics_filepath = lidar_actual_extrinsics_filepath
        self.extrinsics_dict = None
        self._actual_extrinsics_dict = None
        self.load_params()
        if self.ros_flag:
            self.cv_bridge = CvBridge()
            rospy.Subscriber("/zed2i/zed_node/left/image_rect_color/compressed", CompressedImage, self.image_callback, queue_size=1, buff_size=2**32)
            rospy.Subscriber("/ouster/points", PointCloud2, self.pc_callback, queue_size=10)
            self.pub = rospy.Publisher("/lidar_cam/compressed", CompressedImage, queue_size=1)
            rospy.Timer(rospy.Duration(1 / 10), lambda event: self.main(self.latest_img, self.latest_vlp_points))

    def load_params(self):
        with open(self.extrinsics_filepath, 'r') as f:
            self.extrinsics_dict = yaml.safe_load(f)

        with open(self._actual_extrinsics_filepath, 'r') as f:
            self._actual_extrinsics_dict = yaml.safe_load(f)

    def pc_callback(self, msg):
        pc_cloud = ros_numpy.point_cloud2.pointcloud2_to_array(msg).reshape((1, -1))
        pc_np = np.zeros((pc_cloud.shape[0], pc_cloud.shape[1], 3), dtype=np.float32)
        pc_np[..., 0] = pc_cloud['x']
        pc_np[..., 1] = pc_cloud['y']
        pc_np[..., 2] = pc_cloud['z']
        latest_vlp_points = pc_np.reshape((-1, 3))
        self.latest_vlp_points = self._correct_pc(latest_vlp_points)

    def _correct_pc(self, vlp_points):
        """
        Corrects actual pc to desired lidar location pc
        vlp_points: (N x K) numpy array of points in VLP frame
        returns: (N x K) numpy array of points in (corrected) VLP frame
        """
        vlp_points_copy = deepcopy(vlp_points)
        pc_np_xyz = vlp_points[:, :3].reshape((-1, 3)).astype(np.float64)
        real_lidar_to_wcs = np.linalg.inv(self._get_actual_M_ext())
        wcs_to_lidar = self.get_M_ext()
        real_lidar_to_lidar = wcs_to_lidar @ real_lidar_to_wcs  # lidar_from_real_lidar = lidar_from_wcs @ wcs_from_real_lidar
        pc_np_xyz_4d = JackalCameraCalibration.get_homo_from_ordinary(pc_np_xyz)
        lidar_coords_xyz_4d = (real_lidar_to_lidar @ pc_np_xyz_4d.T).T
        lidar_coords_xyz = JackalCameraCalibration.get_ordinary_from_homo(lidar_coords_xyz_4d)
        vlp_points_copy[:, :3] = lidar_coords_xyz
        return vlp_points_copy

    def image_callback(self, msg):
        self.latest_img = self.cv_bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")

    @staticmethod
    def plot_points_on_image(img, corresponding_dists, corresponding_pcs_coords, resize=False):
        """
        img: cv2 image (BGR) to plot on
        corresponding_dists: a list of distances / 1d np array of distances
        corresponding_pcs_coords: N x 2 pixel coordinates corresponding to the distances
        """
        if isinstance(corresponding_dists, np.ndarray):
            corresponding_dists = list(corresponding_dists.squeeze())
        colors = JackalLidarCamCalibration.get_depth_colors(corresponding_dists)
        for i in range(corresponding_pcs_coords.shape[0]):
            cv2.circle(img, tuple(corresponding_pcs_coords[i, :].astype(np.int32)), radius=1, color=colors[i], thickness=-1)
        if resize:
            img = cv2.resize(img, None, fx=0.35, fy=0.35)
        return img

    @staticmethod
    def interp(a, b, x, method="nearest"):
        y = griddata(a, b, x, method=method)
        # for coord, value in zip(a, b):
        #     index = np.where((x == coord).all(axis=1))[0]
        #     y[index] = value
        return y

    def double_interp(self, a, b, x, do_nearest=True, firstmethod="linear"):
        y = self.interp(a, b, x, method=firstmethod)
        single_mask = np.isnan(y[:, 0]).squeeze()
        if do_nearest:
            x2 = x[~single_mask]
            y2 = y[~single_mask]
            y = self.interp(a=x2, b=y2, x=x, method="nearest")
            return y, np.ones(x.shape[0], dtype=bool)
        else:
            return y[~single_mask], ~single_mask

    def projectPCtoImage(self, pc_np, img):
        """
        img: (H x W x 3) numpy array, cv2 based (BGR)
        pc_np: (N x 3) numpy array of points in VLP frame
        """
        side_by_side, pcs_coords, vlp_points, ccs_dists = self.main(img, pc_np)
        return side_by_side, pcs_coords, vlp_points, ccs_dists

    def projectPCtoImageFull(self, pc_np, img, ret_imgs=False, do_nearest=True, firstmethod="linear", resize=True):
        """
        img: (H x W x 3) numpy array, cv2 based (BGR)
        pc_np: (N x 3) numpy array of points in VLP frame
        """
        _, corresponding_pcs_coords, corresponding_vlp_coords, corresponding_ccs_dists = self.projectPCtoImage(pc_np, img)
        all_ys, all_xs = np.meshgrid(np.arange(self.img_height), np.arange(self.img_width))
        all_pixel_locs = np.stack((all_xs.flatten(), all_ys.flatten()), axis=-1)  # K x 2
        all_vlp_coords, interp_mask = self.double_interp(a=corresponding_pcs_coords, b=corresponding_vlp_coords, x=all_pixel_locs, do_nearest=do_nearest, firstmethod=firstmethod)

        all_ccs_dists, interp_mask = self.double_interp(a=corresponding_pcs_coords, b=corresponding_ccs_dists, x=all_pixel_locs, do_nearest=do_nearest, firstmethod=firstmethod)
        all_pixel_locs = all_pixel_locs[interp_mask]
        all_vlp_zs = all_vlp_coords[:, 2].reshape((-1, 1))
        corresponding_vlp_zs = corresponding_vlp_coords[:, 2].reshape((-1, 1))
        if ret_imgs:
            img_vlp_ccs_dists = deepcopy(img)
            img_full_ccs_dists = deepcopy(img)
            img_vlp_vlp_zs = deepcopy(img)
            img_full_vlp_zs = deepcopy(img)
            img_vlp_ccs_dists = JackalLidarCamCalibration.plot_points_on_image(img_vlp_ccs_dists, corresponding_ccs_dists, corresponding_pcs_coords, resize=resize)
            img_full_ccs_dists = JackalLidarCamCalibration.plot_points_on_image(img_full_ccs_dists, all_ccs_dists, all_pixel_locs, resize=resize)
            img_vlp_vlp_zs = JackalLidarCamCalibration.plot_points_on_image(img_vlp_vlp_zs, corresponding_vlp_zs, corresponding_pcs_coords, resize=resize)
            img_full_vlp_zs = JackalLidarCamCalibration.plot_points_on_image(img_full_vlp_zs, all_vlp_zs, all_pixel_locs, resize=resize)
            side_by_side_ccs_dists = np.hstack((img_vlp_ccs_dists, img_full_ccs_dists))
            side_by_side_vlp_zs = np.hstack((img_vlp_vlp_zs, img_full_vlp_zs))
            full_img = np.vstack((side_by_side_ccs_dists, side_by_side_vlp_zs))
            imgs = [full_img, img_vlp_ccs_dists, img_full_ccs_dists, img_vlp_vlp_zs, img_full_vlp_zs]
            return all_pixel_locs, all_vlp_coords, all_ccs_dists, all_vlp_zs, interp_mask, imgs
        return all_pixel_locs, all_vlp_coords, all_ccs_dists, all_vlp_zs, interp_mask

    def main(self, img, vlp_points, event=None):
        """
        img: (H x W x 3) numpy array, cv2 based (BGR)
        vlp_points: (N x 3) numpy array of points in (corrected) VLP frame
        """
        if img is None or vlp_points is None:
            return
        cur_img = deepcopy(img)
        cur_vlp_points = deepcopy(vlp_points)
        pcs_coords, mask, ccs_dists = self.projectVLPtoPCS(cur_vlp_points)
        cur_img = JackalLidarCamCalibration.plot_points_on_image(cur_img, ccs_dists, pcs_coords)
        if self.ros_flag:
            img2 = cv2.resize(cur_img, None, fx=0.75, fy=0.75)
            msg = CompressedImage()
            msg.header.stamp = rospy.Time.now()
            msg.format = "jpeg"
            msg.data = np.array(cv2.imencode('.jpg', img2)[1]).tobytes()
            self.pub.publish(msg)
        else:
            cur_img = cv2.resize(cur_img, None, fx=0.75, fy=0.75)
            img = cv2.resize(img, None, fx=0.75, fy=0.75)
            side_by_side = np.hstack((img, cur_img))
            return side_by_side, pcs_coords, cur_vlp_points[mask], np.asarray(ccs_dists.squeeze()).reshape((-1, 1))

    @staticmethod
    def get_depth_colors(dists):
        """
        Gives colors for depth values
        dists: list of distances
        Returns: list of colors in BGR format
        """
        COLMAP = JackalLidarCamCalibration.COLMAP
        colors = []
        for i in range(len(dists)):
            range_val = max(min(round((dists[i] / 30.0) * 49), 49), 0)
            color = (255 * COLMAP[49 - range_val][2], 255 * COLMAP[49 - range_val][1], 255 * COLMAP[49 - range_val][0])
            colors.append(color)
        return colors

    def get_M_ext(self):
        """
        Returns the extrinsic matrix (4 x 4) that transforms from WCS to VLP frame
        """
        T1 = JackalCameraCalibration.get_std_trans(cx=self.extrinsics_dict['T1']['Trans1']['X'] / 100,
                                                   cy=self.extrinsics_dict['T1']['Trans1']['Y'] / 100,
                                                   cz=self.extrinsics_dict['T1']['Trans1']['Z'] / 100)
        T2 = JackalCameraCalibration.get_std_rot(axis=self.extrinsics_dict['T2']['Rot1']['axis'],
                                                 alpha=np.deg2rad(self.extrinsics_dict['T2']['Rot1']['alpha']))
        T3 = JackalCameraCalibration.get_std_rot(axis=self.extrinsics_dict['T2']['Rot2']['axis'],
                                                 alpha=np.deg2rad(self.extrinsics_dict['T2']['Rot2']['alpha']))
        T4 = JackalCameraCalibration.get_std_rot(axis=self.extrinsics_dict['T2']['Rot3']['axis'],
                                                 alpha=np.deg2rad(self.extrinsics_dict['T2']['Rot3']['alpha']))
        return T4 @ (T2 @ T3) @ T1

    def _get_actual_M_ext(self):
        """
        Returns the actual extrinsic matrix (4 x 4) that transforms from WCS to real VLP frame
        """
        T1 = JackalCameraCalibration.get_std_trans(cx=self._actual_extrinsics_dict['T1']['Trans1']['X'] / 100,
                                                   cy=self._actual_extrinsics_dict['T1']['Trans1']['Y'] / 100,
                                                   cz=self._actual_extrinsics_dict['T1']['Trans1']['Z'] / 100)
        T2 = JackalCameraCalibration.get_std_rot(axis=self._actual_extrinsics_dict['T2']['Rot1']['axis'],
                                                 alpha=np.deg2rad(self._actual_extrinsics_dict['T2']['Rot1']['alpha']))
        T3 = JackalCameraCalibration.get_std_rot(axis=self._actual_extrinsics_dict['T2']['Rot2']['axis'],
                                                 alpha=np.deg2rad(self._actual_extrinsics_dict['T2']['Rot2']['alpha']))
        T4 = JackalCameraCalibration.get_std_rot(axis=self._actual_extrinsics_dict['T2']['Rot3']['axis'],
                                                 alpha=np.deg2rad(self._actual_extrinsics_dict['T2']['Rot3']['alpha']))
        return T4 @ (T2 @ T3) @ T1

    def projectVLPtoWCS(self, vlp_points):
        """
        Project VLP points to WCS
        vlp_points: (N x 3) numpy array of points in VLP frame
        Returns: (N x 3) numpy array of points in WCS
        """
        M_ext_inv = np.linalg.inv(self.get_M_ext())
        return JackalCameraCalibration.general_project_A_to_B(vlp_points, M_ext_inv)

    def projectVLPtoPCS(self, vlp_points, mode="skip"):
        """
        Project VLP points to PCS
        vlp_points: (N x 3) numpy array of points in VLP frame
        Returns: (N x 2) numpy array of points in PCS, in FOV of image, and a mask to indicate which ccs locs were preserved during pixel FOV bounding
        """
        vlp_points = np.array(vlp_points).astype(np.float64)
        wcs_coords = self.projectVLPtoWCS(vlp_points)
        ccs_coords = self.jackal_cam_calib.projectWCStoCCS(wcs_coords)
        pcs_coords, mask = self.jackal_cam_calib.projectCCStoPCS(ccs_coords, mode=mode)
        ccs_dists = np.linalg.norm(ccs_coords, axis=1).reshape((-1, 1))
        ccs_dists = ccs_dists[mask]
        return pcs_coords, mask, ccs_dists

    def projectPCStoWCSusingZ(self, corresponding_pcs_coords, corresponding_vlp_zs, apply_dist=True, mode="skip"):
        """
        Projects set of points in PCS to WCS, using z information from lidar.
        corresponding_pcs_coords: (N x 2) array of points in PCS
        corresponding_vlp_zs: (N x 1) array of z heights (i.e., +Z axis) in (corrected) VLP frame
        Returns: (N x 3) array of points in WCS, and a mask to indicate which pixel locs were kept during FOV bounding
        """
        # Converting the corresponding_vlp_zs to corresponding_wcs_zs
        zeros = np.zeros((corresponding_vlp_zs.shape[0], 2))
        pseudo_vlp_coords = np.hstack([zeros, corresponding_vlp_zs])  # Note VLP frame is only z translated from WCS frame
        pseudo_wcs_coords = self.projectVLPtoWCS(pseudo_vlp_coords)
        corresponding_wcs_zs = pseudo_wcs_coords[:, 2].reshape((-1, 1))

        corresponding_pcs_coords = np.array(corresponding_pcs_coords).astype(np.float64)
        K = self.jackal_cam_calib.intrinsics_dict['camera_matrix']  # 3 x 3
        d = self.jackal_cam_calib.intrinsics_dict['dist_coeffs']
        R = np.eye(3)
        undistorted_pcs_coords = cv2.undistortPoints(corresponding_pcs_coords.reshape(1, -1, 2), K, d, R=R, P=K)
        undistorted_pcs_coords = np.swapaxes(undistorted_pcs_coords, 0, 1).squeeze().reshape((-1, 2))
        undistorted_pcs_coords, pcs_mask = JackalCameraCalibration.to_image_fov_bounds(undistorted_pcs_coords, self.img_width, self.img_height, mode=mode)
        if apply_dist:
            corresponding_pcs_coords = undistorted_pcs_coords
        pcs_coords_3d = JackalCameraCalibration.get_homo_from_ordinary(corresponding_pcs_coords)
        pcs_coords_3d_T = pcs_coords_3d.T  # 3 x N
        M_ext = self.jackal_cam_calib.get_M_ext()  # 4 x 4
        M_ext_short = M_ext[:, [0, 1, 3]][:-1, :].reshape((3, 3))
        M_col3 = M_ext[:, [2]][:-1, :].reshape((3, 1))
        wcs_zs = corresponding_wcs_zs.T  # 1 x N
        lhs = pcs_coords_3d_T - K @ M_col3 @ wcs_zs
        rhs_mat = K @ M_ext_short
        wcs_coords_3d = (np.linalg.inv(rhs_mat) @ lhs).T  # N x 3, Note this has x, y, 1
        wcs_coords = JackalCameraCalibration.get_ordinary_from_homo(wcs_coords_3d)  # N x 2, this has x, y
        wcs_coords_full = np.hstack([wcs_coords, corresponding_wcs_zs]).reshape((wcs_coords.shape[0], 3))
        ccs_coords_full = self.jackal_cam_calib.projectWCStoCCS(wcs_coords_full)
        ccs_mask = (ccs_coords_full[:, 2] >= 0)  # this ccs_mask calculation is to cross-check if projected wcs points make sense or not
        unified_mask = deepcopy(pcs_mask)
        unified_mask[pcs_mask] = ccs_mask
        return wcs_coords_full[ccs_mask], unified_mask


if __name__ == "__main__":
    rospy.init_node('lidar_cam_calib_testing', anonymous=False)
    e = JackalLidarCamCalibration()
    time.sleep(1)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS lidar cam calib testing module")
