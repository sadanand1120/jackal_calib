#!/usr/bin/env python

import rospy
import numpy as np
import argparse
from sensor_msgs.msg import CameraInfo
from spot_calib import SpotCameraCalibration


def get_cam_intrinsic(c):
    return (c.intrinsics_dict['camera_matrix'], c.intrinsics_dict['dist_coeffs'])


def publish_camera_info(res):
    c = SpotCameraCalibration(resolution=res)
    (mtx, dist) = get_cam_intrinsic(c)
    rospy.init_node('camera_info_publisher')
    publisher = rospy.Publisher("/camera/kinect/camera_info", CameraInfo, queue_size=10)
    camera_info_msg = CameraInfo()

    # Header
    camera_info_msg.header.stamp = rospy.Time.now()
    camera_info_msg.header.frame_id = "kinect_color"

    # Image dimensions
    camera_info_msg.height = c.img_height
    camera_info_msg.width = c.img_width

    # Distortion model
    camera_info_msg.distortion_model = "plumb_bob"

    # Distortion parameters (D)
    camera_info_msg.D = list(dist.reshape((1, 5)).squeeze())

    # Intrinsic camera matrix (K)
    camera_info_msg.K = list(mtx.reshape((1, 9)).squeeze())

    # Rectification matrix (R)
    camera_info_msg.R = list(np.eye(3).reshape((1, 9)).squeeze())

    # Projection/camera matrix (P)
    # For monocular cameras, P is [fx 0 cx 0, 0 fy cy 0, 0 0 1 0]
    P = np.hstack((mtx, np.zeros((3, 1))))
    camera_info_msg.P = list(P.reshape((1, 12)).squeeze())

    rate = rospy.Rate(30)  # 10Hz

    while not rospy.is_shutdown():
        # Update the timestamp
        camera_info_msg.header.stamp = rospy.Time.now()

        # Publish the message
        publisher.publish(camera_info_msg)

        # Sleep enough to maintain the desired rate
        rate.sleep()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", default=1440, type=int, help="Camera resolution 1440 or 1536")
    args = parser.parse_args(rospy.myargv()[1:])
    try:
        publish_camera_info(args.res)
    except rospy.ROSInterruptException:
        pass
