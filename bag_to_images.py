# system
import os
import argparse
import signal
import time
import sys

# images
import cv2

# ros
from cv_bridge import CvBridge
import rosbag

IMAGE_FILEEXT = ".png"
IMAGE_START_INDEX = 0
IMAGE_TOPICNAME = "/camera/rgb/image_raw/compressed"


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bagfile", default="/home/dynamo/AMRL_Research/repos/spot_calib/calib_1440.bag", type=str, help="Bagfile path")
    parser.add_argument("--outputdir", default="/home/dynamo/AMRL_Research/repos/spot_calib/calib_1440_images/", type=str, help="Output root directory path")
    args = parser.parse_args()
    if not os.path.exists(args.bagfile):
        raise FileNotFoundError("Input bagfile " + args.bagfile + " doesn't exist")
    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir, exist_ok=True)
    return args


def parse_bag(args):
    def signal_handler(sig, frame):
        print("Ctrl+C detected! Killing script...")
        bag.close()
        time.sleep(2)
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    bag = rosbag.Bag(args.bagfile, "r")
    image_dirpath = args.outputdir
    if not os.path.exists(image_dirpath):
        os.makedirs(image_dirpath, exist_ok=True)
    img_idx = IMAGE_START_INDEX
    bridge = CvBridge()
    for topic, msg, t in bag.read_messages(topics=[IMAGE_TOPICNAME]):
        if topic == IMAGE_TOPICNAME:
            print(f"Saved image {img_idx}")
            img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")
            filename = f'{img_idx:06}' + IMAGE_FILEEXT
            filepath = os.path.join(image_dirpath, filename)
            cv2.imwrite(filepath, img)
            img_idx += 1
        else:
            pass
    bag.close()


if __name__ == "__main__":
    args = parse_opt()
    print(args)
    parse_bag(args)
