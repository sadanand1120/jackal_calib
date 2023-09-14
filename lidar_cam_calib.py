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

from cam_calib import JackalCameraCalibration


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

    def __init__(self, ros_flag=True):
        self.jackal_cam_calib = JackalCameraCalibration()
        self.latest_img = None
        self.latest_vlp_points = None
        self.ros_flag = ros_flag
        self.extrinsics_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "params/baselink_to_lidar_extrinsics.yaml")
        self._actual_extrinsics_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "params/baselink_to_actual_lidar_extrinsics.yaml")
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
        colors = JackalLidarCamCalibration.get_depth_colors(list(ccs_dists.squeeze()))
        for i in range(pcs_coords.shape[0]):
            cv2.circle(cur_img, tuple(pcs_coords[i, :].astype(np.int32)), radius=1, color=colors[i], thickness=-1)
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
        return T3 @ T2 @ T1

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
        return T3 @ T2 @ T1

    def projectVLPtoWCS(self, vlp_points):
        """
        Project VLP points to WCS
        vlp_points: (N x 3) numpy array of points in VLP frame
        Returns: (N x 3) numpy array of points in WCS
        """
        vlp_points = np.array(vlp_points).astype(np.float64)
        vlp_points_4d = JackalCameraCalibration.get_homo_from_ordinary(vlp_points)
        M_ext = np.linalg.inv(self.get_M_ext())
        wcs_coords_4d = (M_ext @ vlp_points_4d.T).T
        return JackalCameraCalibration.get_ordinary_from_homo(wcs_coords_4d)

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


if __name__ == "__main__":
    rospy.init_node('lidar_cam_calib_testing', anonymous=False)
    e = JackalLidarCamCalibration()
    time.sleep(1)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS lidar cam calib testing module")
