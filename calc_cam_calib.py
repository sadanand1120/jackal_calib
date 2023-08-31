#!/usr/bin/env python

import cv2
import numpy as np
import os
import glob
import yaml
from cam_calib import JackalCameraCalibration

images_dir = "/home/dynamo/AMRL_Research/repos/jackal_calib/cam_calib_imgs"
save_calib = "/home/dynamo/AMRL_Research/repos/jackal_calib/params/zed_left_rect_intrinsics.yaml"
MODE = "calib"  # "filter" or "calib" or "reproj_err"
VIEW_MODE = False

if MODE == "reproj_err":
    spot_cam_calib = JackalCameraCalibration()
    loaded_mtx = np.array(spot_cam_calib.intrinsics_dict['camera_matrix'])
    loaded_dist = np.array(spot_cam_calib.intrinsics_dict['dist_coeffs'])
    loaded_rvecs = np.array(spot_cam_calib.intrinsics_dict['rvecs'])
    loaded_tvecs = np.array(spot_cam_calib.intrinsics_dict['tvecs'])


if MODE not in ["filter", "calib", "reproj_err"]:
    raise ValueError("Invalid mode")

# Defining the dimensions of checkerboard
CHECKERBOARD = (7, 10)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 500, 1e-8)

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = []

# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

# Extracting path of individual image stored in a given directory
images = glob.glob(images_dir + '/*.png')


def corners_are_good_quality(corners):
    # Define your criteria for good quality corners here
    # This can be based on the distance between corners, the position of the corners in the image etc.
    # This is an example check where we reject corners too close to image edges
    # You should add more checks based on your requirements
    for corner in corners:
        x, y = corner.ravel()
        PIX_THRESH = 80
        if x < PIX_THRESH or y < PIX_THRESH or x > img.shape[1] - PIX_THRESH or y > img.shape[0] - PIX_THRESH:
            return False
    return True


def calculate_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    return total_error / len(objpoints)


def resize_img(img, scale_percent=60):
    _width = int(img.shape[1] * scale_percent / 100)
    _height = int(img.shape[0] * scale_percent / 100)
    dim = (_width, _height)
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def pretty_print(v):
    print(np.array_str(v, precision=4, suppress_small=True))


if MODE == "filter":
    tot_removed = 0

print(f"To process {len(images)} images")
for i, fname in enumerate(images):
    print(f"Processing image: {i}")
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners. If desired number of corners are found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if MODE == "calib" or MODE == "reproj_err":
        # ASSUMES you did the filtering already
        # Perform calib and show the images with corners where you could find all corners
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)  # refining pixel coordinates for given 2d points.
        imgpoints.append(corners2)
        if VIEW_MODE:
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)  # Draw and display the corners
    elif MODE == "filter":
        # Perform filtering and remove images where you could not find all corners. Show the images which you are removing
        if ret == False or not corners_are_good_quality(corners):
            print(f"Removing image: {i}")
            os.remove(fname)
            tot_removed += 1

    if VIEW_MODE:
        cv2.imshow('img', resize_img(img))
        cv2.waitKey(0)


cv2.destroyAllWindows()

if MODE == "filter":
    print(f"Total removed: {tot_removed}")
elif MODE == "calib":
    """
    Performing camera calibration by passing the value of known 3D points (objpoints) and corresponding pixel 
    coordinates of the detected corners (imgpoints)
    """
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    reprojection_error = calculate_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist)
    print(f"Reprojection Error: {reprojection_error}")

    print("camera_matrix:")
    pretty_print(mtx)
    print("dist_coeffs:")
    pretty_print(dist)

    # Saving the results into yaml file
    mtx_list = np.array(mtx).squeeze().tolist()
    dist_list = np.array(dist).squeeze().tolist()
    rvecs_list = np.array(rvecs).squeeze().tolist()
    tvecs_list = np.array(tvecs).squeeze().tolist()

    calibration_data = {
        "camera_matrix": mtx_list,
        "dist_coeffs": dist_list,
        "height": img.shape[0],
        "width": img.shape[1],
        "rvecs": rvecs_list,
        "tvecs": tvecs_list
    }
    with open(save_calib, 'w') as outfile:
        yaml.dump(calibration_data, outfile)
elif MODE == "reproj_err":
    reprojection_error = calculate_reprojection_error(objpoints, imgpoints, loaded_rvecs, loaded_tvecs, loaded_mtx, loaded_dist)
    print(f"Reprojection Error: {reprojection_error}")
