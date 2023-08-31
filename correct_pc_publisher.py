#!/usr/bin/env python
# Corrects actual pc to what tf frames say (ie, 0.85 m z translation from baselink only)
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import yaml
import os
from copy import deepcopy
import argparse
from cam_calib import JackalCameraCalibration
from lidar_cam_calib import JackalLidarCamCalibration


def get_M_ext():
    """
    Returns the extrinsic matrix (4 x 4) that transforms from WCS to real VLP frame
    """
    extrinsics_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "params/baselink_to_actual_lidar_extrinsics.yaml")
    with open(extrinsics_filepath, 'r') as f:
        extrinsics_dict = yaml.safe_load(f)
    T1 = JackalCameraCalibration.get_std_trans(cx=extrinsics_dict['T1']['Trans1']['X'] / 100,
                                               cy=extrinsics_dict['T1']['Trans1']['Y'] / 100,
                                               cz=extrinsics_dict['T1']['Trans1']['Z'] / 100)
    T2 = JackalCameraCalibration.get_std_rot(axis=extrinsics_dict['T2']['Rot1']['axis'],
                                             alpha=np.deg2rad(extrinsics_dict['T2']['Rot1']['alpha']))
    T3 = JackalCameraCalibration.get_std_rot(axis=extrinsics_dict['T2']['Rot2']['axis'],
                                             alpha=np.deg2rad(extrinsics_dict['T2']['Rot2']['alpha']))
    return T3 @ T2 @ T1


def correct_pc(msg):
    gen = pc2.read_points(msg, field_names=None, skip_nans=True)
    pc_np = np.array(list(gen))
    lidar_coords = deepcopy(pc_np)
    pc_np_xyz = pc_np[:, :3].reshape((-1, 3)).astype(np.float64)
    real_lidar_to_wcs = np.linalg.inv(get_M_ext())
    wcs_to_lidar = c.get_M_ext()

    # lidar_from_real_lidar = lidar_from_wcs @ wcs_from_real_lidar
    real_lidar_to_lidar = wcs_to_lidar @ real_lidar_to_wcs

    pc_np_xyz_4d = JackalCameraCalibration.get_homo_from_ordinary(pc_np_xyz)
    lidar_coords_xyz_4d = (real_lidar_to_lidar @ pc_np_xyz_4d.T).T
    lidar_coords_xyz = JackalCameraCalibration.get_ordinary_from_homo(lidar_coords_xyz_4d)
    lidar_coords[:, :3] = lidar_coords_xyz
    lidar_coords_list = [tuple(int(el) if idx >= len(row) - 5 else el for idx, el in enumerate(row)) for row in lidar_coords.tolist()]
    lidar_cloud = pc2.create_cloud(msg.header, msg.fields, lidar_coords_list)
    lidar_cloud.is_dense = True
    pub.publish(lidar_cloud)


if __name__ == "__main__":
    c = JackalLidarCamCalibration(ros_flag=False)
    rospy.init_node('pointcloud_correction', anonymous=False)
    rospy.Subscriber("/ouster/points", PointCloud2, correct_pc, queue_size=1)
    pub = rospy.Publisher('/corrected_ouster_points', PointCloud2, queue_size=1)
    rospy.spin()
