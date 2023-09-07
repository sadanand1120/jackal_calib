import os
import cv2
import rosbag
from cv_bridge import CvBridge, CvBridgeError
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Convert a ROS bag to an MP4 video.")
    parser.add_argument("--bagfile", required=True, type=str, help="Path to the ROS bag file.")
    parser.add_argument("--output", required=True, type=str, help="Path to the output MP4 video.")
    parser.add_argument("--topic", default="/zed2i/zed_node/left/image_rect_color/compressed", type=str, help="Image topic in the ROS bag.")
    parser.add_argument("--fps", default=5, type=int, help="Frames per second for the output video.")
    return parser.parse_args()


def bag_to_video(bagfile, output, image_topic, fps):
    # Open the ROS bag file
    bag = rosbag.Bag(bagfile, "r")
    bridge = CvBridge()

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = None

    for topic, msg, t in bag.read_messages(topics=[image_topic]):
        try:
            # Convert the image message to an OpenCV frame
            frame = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError as e:
            print(f"Error: {e}")
            continue

        # Initialize the video writer
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(output, fourcc, fps, (w, h), True)

        # Write the frame to the video file
        writer.write(frame)

    # Release resources
    writer.release()
    bag.close()


if __name__ == "__main__":
    args = parse_args()
    bag_to_video(args.bagfile, args.output, args.topic, args.fps)
