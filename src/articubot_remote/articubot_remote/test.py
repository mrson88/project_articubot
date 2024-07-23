#!/usr/bin/env python3
import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QSlider, QTextEdit
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from cv_bridge import CvBridge
from PyQt5.QtCore import QLibraryInfo
import os
from geometry_msgs.msg import Twist
from pathlib import Path
import signal
import subprocess
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(
    QLibraryInfo.PluginsPath
)
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import TransformStamped
import math
from tf_transformations import euler_from_quaternion
class ArmRobotControllerNode(QMainWindow):
    def __init__(self):
        super().__init__('link_pose_getter')
        self.initUI()
        
        # Initialize ROS2 node
        rclpy.init()
        self.node = Node('joint_controller')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.position_and_orientation=[0,0,0,0,0,0,0]

        # Replace 'your_link_name' with the name of the link you want to track
        self.link_name = 'claw_support'
        # Create a timer that will trigger the callback every 1 second

        self.publisher_joint = self.node.create_publisher(JointState, 'arm_robot', 10)
        self.publisher_twist = self.node.create_publisher(Twist, '/cmd_vel', 10)
        self.publisher_joint_commend = self.node.create_publisher(
            JointTrajectory, 
            '/arm_controller/follow_joint_trajectory', 
            10
        )
        self.action_client = ActionClient(self.node, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory')
        self.bridge = CvBridge()       
        # Initialize camera subscriber
        self.camera_subscriber = self.node.create_subscription(
            Image,
            '/camera/image_raw',  # Adjust this topic to match your camera's topic
            self.camera_callback,
            10)

    def timer_callback(self):
        try:
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                'base_link',
                self.link_name,
                rclpy.time.Time())

            # Extract the position
            position = transform.transform.translation

            # Extract the orientation
            orientation = transform.transform.rotation

            # Convert quaternion to Euler angles (roll, pitch, yaw)
            quaternion = [
                orientation.x,
                orientation.y,
                orientation.z,
                orientation.w
            ]
            roll, pitch, yaw = euler_from_quaternion(quaternion)

            print(
                f'Pose of {self.link_name}:\n'
                f'Position: x={position.x:.2f}, y={position.y:.2f}, z={position.z:.2f}\n'
                f'Orientation (quaternion): x={orientation.x:.2f}, y={orientation.y:.2f}, '
                f'z={orientation.z:.2f}, w={orientation.w:.2f}\n'
                f'Orientation (RPY radians): roll={roll:.2f}, pitch={pitch:.2f}, yaw={yaw:.2f}\n'
                f'Orientation (RPY degrees): roll={math.degrees(roll):.2f}, '
                f'pitch={math.degrees(pitch):.2f}, yaw={math.degrees(yaw):.2f}'
            )
            self.position_and_orientation  = [position.x,position.y,position.z,orientation.x,orientation.y,orientation.z,orientation.w]

        except TransformException as ex:
            print(
                f'Could not transform {self.link_name} to world: {ex}')    
        return self.position_and_orientation
        # self.bridge = CvBridge()
# class MainWindow(QMainWindow,ArmRobotControllerNode):

    


    def initUI(self):
        self.setWindowTitle('Arm Robot Controller')
        self.setGeometry(100, 100, 1280, 768)
        self.joint_names=['arm_base_forearm_joint', 'forearm_hand_1_joint', 'forearm_hand_2_joint', 'forearm_hand_3_joint', 'forearm_claw_joint']
        self.position_and_orientation_str=['0','0','0','0','0','0','0']
        main_layout = QHBoxLayout()
    
        # Control buttons layout
        control_layout = QVBoxLayout()
        self.btn_forward = QPushButton('Move Forward', self)
        self.btn_backward = QPushButton('Move Backward', self)
        self.btn_left = QPushButton('Move Left', self)
        self.btn_right = QPushButton('Move Right', self)
        self.btn_stop = QPushButton('Stop', self)
        
        
        self.btn_home = QPushButton('Move to Home', self)
        self.btn_position1 = QPushButton('Move to Position 1', self)
        self.btn_position2 = QPushButton('Move to Position 2', self)
        self.launch_button = QPushButton("Launch Robot Recogntion")
        self.close_launch_button = QPushButton("Close Robot Recogntion")

        self.btn_forward.clicked.connect(self.move_forward)
        self.btn_backward.clicked.connect(self.move_backward)
        self.btn_left.clicked.connect(self.rotate_left)
        self.btn_right.clicked.connect(self.rotate_right)
        self.btn_stop.clicked.connect(self.stop)
        self.btn_home.clicked.connect(self.move_to_home)
        self.btn_position1.clicked.connect(self.move_to_position1)
        self.btn_position2.clicked.connect(self.move_to_position2)
        self.launch_button.clicked.connect(self.launch_robot_recognition)
        self.close_launch_button.clicked.connect(self.close_robot_recognition)
        
        
        control_layout.addWidget(self.btn_home)
        control_layout.addWidget(self.btn_position1)
        control_layout.addWidget(self.btn_position2)
        control_layout.addWidget(self.btn_forward)
        control_layout.addWidget(self.btn_backward)
        control_layout.addWidget(self.btn_left)
        control_layout.addWidget(self.btn_right)
        control_layout.addWidget(self.btn_stop)
        control_layout.addWidget(self.launch_button)
        control_layout.addWidget(self.close_launch_button)  
        # control_layout.addStretch(1)

        self.joint_sliders = []
        self.joint_labels = []
        self.position_labels = []

        for i in self.joint_names:  # Assuming 6 joints, adjust as needed
            layout = QHBoxLayout()
            
            label = QLabel(f"{i}:")
            control_layout.addWidget(label)

            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(-100)
            slider.setMaximum(100)
            slider.setValue(0)
            slider.setTickPosition(QSlider.TicksBelow)
            slider.setTickInterval(10)
            slider.valueChanged.connect(self.update_label)
            self.joint_sliders.append(slider)
            control_layout.addWidget(slider)

            value_label = QLabel("0.00")
            self.joint_labels.append(value_label)
            control_layout.addWidget(value_label)
        for i in range(7):  # Assuming 6 joints, adjust as needed
            label = QLabel(f"Position {i+1}: {self.position_and_orientation_str[i]}")
            self.position_labels.append(label)
            control_layout.addWidget(label)
        send_button = QPushButton("Send Command")
        send_button.clicked.connect(self.send_joint_command)
        control_layout.addWidget(send_button)





        # Video display
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)

        main_layout.addLayout(control_layout)
        main_layout.addWidget(self.video_label)

        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        control_layout.addWidget(self.status_text)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def update_label(self):
        for i, slider in enumerate(self.joint_sliders):
            value = slider.value() / 100.0  # Convert to -1.0 to 1.0 range
            self.joint_labels[i].setText(f"{value:.2f}")
    def send_joint_command(self):
        goal_msg = FollowJointTrajectory.Goal()
        
        trajectory = JointTrajectory()
        trajectory.joint_names = ['arm_base_forearm_joint', 'forearm_hand_1_joint', 'forearm_hand_2_joint', 'forearm_hand_3_joint', 'forearm_claw_joint']   # Adjust joint names as needed
        
        point = JointTrajectoryPoint()
        point.positions = [slider.value() / 100.0 for slider in self.joint_sliders]
        point.time_from_start.sec = 2  # Move to position in 2 seconds
        
        trajectory.points = [point]
        goal_msg.trajectory = trajectory

        self.action_client.wait_for_server()
        
        self.status_text.append("Sending goal...")
        send_goal_future = self.action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        send_goal_future.add_done_callback(self.goal_response_callback)
    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.status_text.append("Goal rejected")
            return

        self.status_text.append("Goal accepted")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.status_text.append(f"Result: {result}")

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.status_text.append(f"Feedback: {feedback}")

    def launch_robot_recognition(self):
        try:
            self.launch_process = subprocess.Popen(['ros2', 'launch', 'robot_recognition', 'launch_yolov8.launch.py'])
            print("Launched robot arm successfully")
        except subprocess.CalledProcessError as e:
            print(f"Failed to launch robot arm: {e}")
    def move_to_home(self):
        self.publish_joint_states([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def move_to_position1(self):
        self.publish_joint_states([0.5, -0.3, 0.2, 0.1, 0.0, 0.0])

    def move_to_position2(self):
        self.publish_joint_states([-0.2, 0.4, -0.1, 0.3, 0.1, 0.0])

    def publish_joint_states(self, positions):
        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = ['arm_base_forearm_joint', 'forearm_hand_1_joint', 'forearm_hand_2_joint', 'forearm_hand_3_joint', 'forearm_claw_joint'] 
        msg.position = positions
        self.publisher_joint.publish(msg)

    def camera_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.display_image(cv_image)
        except Exception as e:
            print(f'Error processing image: {str(e)}')

    def display_image(self, cv_image):
        height, width, channel = cv_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(cv_image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_image)
        self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def process_ros_events(self):
        rclpy.spin_once(self.node, timeout_sec=0)

    def closeEvent(self, event):
        self.node.destroy_node()
        rclpy.shutdown()
    def move_forward(self):
        self.publish_command(1.0, 0.0)

    def move_backward(self):
        self.publish_command(-1.0, 0.0)
        
    def rotate_right(self):
        self.publish_command(0, -1.0)
        
    def rotate_left(self):
        self.publish_command(0, 1.0)
        
    def stop(self):
        self.publish_command(0.0, 0.0)

    def publish_command(self, linear, angular):
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        self.publisher_twist.publish(msg)
    def close_robot_recognition(self):
        if self.launch_process and self.launch_process.poll() is None:
            
            self.launch_process.send_signal(signal.SIGINT)
            print("Closing robot recognition...")
            try:
                self.launch_process.wait(timeout=10)
                print("Robot arm closed successfully")
            except subprocess.TimeoutExpired:
                print("Timeout while waiting for robot arm to close, forcing termination")
                self.launch_process.kill()
            self.launch_process = None
        else:
            print("No robot arm process to close")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ArmRobotControllerNode()
    ex.show()
    sys.exit(app.exec_())
