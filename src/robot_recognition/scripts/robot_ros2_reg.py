#!/usr/bin/env python3


import os
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from cv_bridge import CvBridge
from ultralytics import YOLO
import torch
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Twist, PointStamped, PoseStamped
from std_msgs.msg import String
from articubot_msgs.msg import InferenceResult, Yolov8Inference
from articubot_msgs.action import ArticubotTask
from ament_index_python.packages import get_package_share_directory
from tf_transformations import quaternion_from_euler

class CameraSubscriber(Node):
    def __init__(self):
        super().__init__('camera_subscriber')
        self.setup_parameters()
        self.setup_publishers_subscribers()
        self.setup_model()
        self.setup_variables()

    def setup_parameters(self):
        self.declare_parameters(
            namespace='',
            parameters=[
                ('rcv_timeout_secs', 1.0),
                ('angular_chase_multiplier', 0.2),
                ('forward_chase_speed', 0.1),
                ('search_angular_speed', 0.15),
                ('max_size_thresh', 0.1),
                ('filter_value', 0.9)
            ]
        )
        self.rcv_timeout_secs = self.get_parameter('rcv_timeout_secs').value
        self.angular_chase_multiplier = self.get_parameter('angular_chase_multiplier').value
        self.forward_chase_speed = self.get_parameter('forward_chase_speed').value
        self.search_angular_speed = self.get_parameter('search_angular_speed').value
        self.max_size_thresh = self.get_parameter('max_size_thresh').value
        self.filter_value = self.get_parameter('filter_value').value

    def setup_publishers_subscribers(self):
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.yolov8_pub = self.create_publisher(Yolov8Inference, "/Yolov8_Inference", 5)
        self.img_pub = self.create_publisher(Image, "/inference_result", 5)
        self.publisher_point = self.create_publisher(PointStamped, 'point_3d', 5)

        self.create_subscription(Image, 'camera/camera/color/image_raw', self.camera_callback, 1)
        self.create_subscription(Image, 'camera/camera/depth/image_rect_raw', self.depth_camera_callback, 1)
        self.create_subscription(CameraInfo, 'camera/camera/depth/camera_info', self.camera_info_callback, 10)
        self.create_subscription(String, 'find_ball', self.findball_callback, 5)

        self._action_client = ActionClient(self, ArticubotTask, 'test_server')

        self.timer = self.create_timer(0.1, self.timer_callback)

    def setup_model(self):
        package_share_dir = get_package_share_directory("robot_recognition")
        # model_dir = os.path.join(package_share_dir, "scripts", "yolov8m.pt")
        model_engine_onnx_dir = os.path.join(package_share_dir, "scripts", "yolov8m.engine")
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model = YOLO(model_dir).to(self.device)
        self.model=YOLO(model_engine_onnx_dir,task='detect')

    def setup_variables(self):
        self.bridge = CvBridge()
        self.target_val = 0.0
        self.target_dist = 0.0
        self.pixel_x = 0
        self.pixel_y = 0
        self.detect = False
        self.camera_info = None
        self.inference_result = InferenceResult()
        self.frame_height = 480
        self.frame_width = 640
        self.depth_image = None
        self.findball = True

    def camera_info_callback(self, msg):
        self.camera_info = msg

    def timer_callback(self):
        msg = Twist()
        if self.findball and (self.inference_result.class_name in ["sports ball", "frisbee"]):
            if self.target_dist > self.max_size_thresh:
                msg.linear.x = self.forward_chase_speed
                self.detect = True
            msg.angular.z = -self.angular_chase_multiplier * self.target_val
        elif self.findball and self.pixel_x < self.frame_height/2:
            msg.angular.z = self.search_angular_speed
        self.publisher_.publish(msg)

    def deproject_pixel_to_point(self, K, pixel, depth):
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        x = (pixel[0] - cx) * depth / fx
        y = (pixel[1] - cy) * depth / fy
        return [x, y, float(depth)]

    def publish_point(self, point_3d):
        point_msg = PointStamped()
        pose_msg = PoseStamped()
        point_msg.header.stamp = self.get_clock().now().to_msg()
        point_msg.header.frame_id = 'depth_camera'
        point_msg.point.x, point_msg.point.y, point_msg.point.z = point_3d

        pose_msg.pose.position.x = abs(point_3d[0] / 10)
        pose_msg.pose.position.y = abs(point_3d[1] / 10)
        pose_msg.pose.position.z = abs(point_3d[2] / 10)
        quat = quaternion_from_euler(0, 160, 0)
        pose_msg.pose.orientation.x, pose_msg.pose.orientation.y, pose_msg.pose.orientation.z, pose_msg.pose.orientation.w = quat

        self.publisher_point.publish(point_msg)
        return pose_msg

    def pixel_to_3d(self, x, y, depth):
        fx, fy = self.camera_info.k[0], self.camera_info.k[4]
        cx, cy = self.camera_info.k[2], self.camera_info.k[5]
        x3d = ((x - cx) * depth / fx) * 0.001
        y3d = ((y - cy) * depth / fy) * 0.001
        z3d = depth * 0.001
        return x3d, y3d, z3d

    def camera_callback(self, data):
        if self.depth_image is None or self.camera_info is None:
            return

        img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        resized_image = cv2.resize(img, (self.frame_width, self.frame_height))
        
        results = self.model(resized_image, conf=0.5, verbose=False)
        self.yolov8_inference = Yolov8Inference()
        self.yolov8_inference.header.frame_id = "inference"
        self.yolov8_inference.header.stamp = self.get_clock().now().to_msg()

        for r in results:
            for box in r.boxes:
                self.process_detection(box)

        annotated_frame = results[0].plot()
        img_msg = self.bridge.cv2_to_imgmsg(annotated_frame)  
        self.img_pub.publish(img_msg)
        self.yolov8_pub.publish(self.yolov8_inference)

    def process_detection(self, box):
        b = box.xyxy[0].to('cpu').detach().numpy().copy()
        c = box.cls
        self.inference_result = InferenceResult()
        self.inference_result.class_name = self.model.names[int(c)]
        self.inference_result.top, self.inference_result.left, self.inference_result.bottom, self.inference_result.right = map(int, b)
        self.pixel_x = int((self.inference_result.top + self.inference_result.bottom) / 2)
        self.pixel_y = int((self.inference_result.right + self.inference_result.left) / 2)
        self.yolov8_inference.yolov8_inference.append(self.inference_result)

        if self.inference_result.class_name in ["sports ball", "frisbee"]:
            self.process_ball_detection()

    def process_ball_detection(self):
        self.target_dist = self.depth_image[self.pixel_y, self.pixel_x]
        self.target_val = 2 * self.pixel_y / self.frame_width - 1
        K = np.array(self.camera_info.k).reshape(3, 3)
        point_3d = self.deproject_pixel_to_point(K, [self.pixel_x / 1000, self.pixel_y / 1000], self.target_dist / 1000)
        point_position = self.publish_point(point_3d)
        
        self.get_logger().info(f'Position: x={point_position.pose.position.x}, y={point_position.pose.position.y}, z={point_position.pose.position.z}')
        self.get_logger().info(f'Orientation: x={point_position.pose.orientation.x}, y={point_position.pose.orientation.y}, z={point_position.pose.orientation.z}')
        
        if 0 < self.target_dist < 445 and self.detect:
            self.send_pickup_goal(point_position)

    def send_pickup_goal(self, point_position):
        self.findball = False
        self.detect = False
        self.get_logger().info("Sending pickup goal")
        self.send_goal(
            point_position.pose.position.x,
            point_position.pose.position.y,
            point_position.pose.position.z,
            0.0, 150.0, 0.0, 5
        )

    def send_goal(self, p_x, p_y, p_z, or_x, or_y, or_z, order):
        goal_msg = ArticubotTask.Goal()
        goal_msg.task = order
        goal_msg.p_x, goal_msg.p_y, goal_msg.p_z = p_x, p_y, p_z
        goal_msg.or_x, goal_msg.or_y, goal_msg.or_z = or_x, or_y, or_z
        
        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def get_result_callback(self, future):
        result = future.result().result
        status = future.result().status
        self.get_logger().info(f'Result: {result.success}')
        self.get_logger().info(f'Status: {status}')
        self.detect = True

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Received feedback: {feedback}')

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return
        self.get_logger().info('Goal accepted')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def depth_camera_callback(self, data):
        self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")

    def findball_callback(self, msg):
        self.findball = "true" in msg.data.lower()

def main(args=None):
    rclpy.init(args=args)
    camera_subscriber = CameraSubscriber()
    rclpy.spin(camera_subscriber)
    rclpy.shutdown()

if __name__ == '__main__':
    main()