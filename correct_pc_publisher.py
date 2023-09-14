#!/usr/bin/env python
# Corrects actual pc to what tf frames say (ie, 0.85 m z translation from baselink only)
# UPDATE: NOT REQUIRED anymore, as a correct_pc method included in lidar_cam_calib.py that corrects from original pc on-the-fly
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


def correct_pc(msg):
    gen = pc2.read_points(msg, field_names=None, skip_nans=True)
    pc_np = np.array(list(gen))
    lidar_coords = c._correct_pc(pc_np)
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
