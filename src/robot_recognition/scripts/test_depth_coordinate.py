#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import numpy as np

class DepthImageProcessor(Node):

    def __init__(self):
        super().__init__('depth_image_processor')
        self.bridge = CvBridge()
        self.depth_image = None

        self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.image_callback,
            10)
        self.create_subscription(
            CameraInfo,
            '/camera/depth/camera_info',
            self.camera_info_callback,
            10)

        self.publisher_ = self.create_publisher(PointStamped, 'point_3d', 10)
        self.camera_info = None

    def image_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")

    def camera_info_callback(self, msg):
        self.camera_info = msg

    def process_depth_image(self):
        if self.depth_image is None or self.camera_info is None:
            return

        # Get the depth value at a specific point (e.g., center of the image)
        x = int(self.depth_image.shape[1] / 2)
        y = int(self.depth_image.shape[0] / 2)
        depth = self.depth_image[y, x]

        # Intrinsic camera matrix for the raw (distorted) images
        K = np.array(self.camera_info.k).reshape(3, 3)

        # Deproject the depth image to 3D point
        point_3d = self.deproject_pixel_to_point(K, [x, y], depth)

        self.publish_point(point_3d)

    def deproject_pixel_to_point(self, K, pixel, depth):
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        x = (pixel[0] - cx) * depth / fx
        y = (pixel[1] - cy) * depth / fy
        z = float(depth)

        return [x, y, z]

    def publish_point(self, point_3d):
        point_msg = PointStamped()
        point_msg.header.stamp = self.get_clock().now().to_msg()
        point_msg.header.frame_id = 'depth_camera'
        point_msg.point.x = point_3d[0]
        point_msg.point.y = point_3d[1]
        point_msg.point.z = point_3d[2]
        self.publisher_.publish(point_msg)
        self.get_logger().info(f"Published Point: {point_msg}")

def main(args=None):
    rclpy.init(args=args)
    processor = DepthImageProcessor()

    try:
        while rclpy.ok():
            rclpy.spin_once(processor)
            processor.process_depth_image()
    except KeyboardInterrupt:
        pass
    finally:
        processor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
