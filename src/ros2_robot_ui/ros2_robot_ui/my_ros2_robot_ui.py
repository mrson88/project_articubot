#!/usr/bin/env python3
import sys
import threading
import time
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QSlider, QTextEdit, QListWidget, QMessageBox, QLineEdit
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
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
from articubot_msgs.action import ArticubotTask
from nav2_simple_commander.robot_navigator import BasicNavigator
from geometry_msgs.msg import PoseStamped
import tf_transformations
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import String
import pyttsx3 

class ArmRobotControllerNode(Node):
    def __init__(self):
        super().__init__('arm_robot_controller')
        self.get_logger().info("ArmRobotController initialized")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.position_and_orientation=[0,0,0,0,0,0,0]
        self._action_client = ActionClient(self, ArticubotTask, 'test_server')
        # Replace 'your_link_name' with the name of the link you want to track
        self.link_name = 'claw_support'
        # Create a timer that will trigger the callback every 1 second
        
        

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

            # print(
            #     f'Pose of {self.link_name}:\n'
            #     f'Position: x={position.x:.2f}, y={position.y:.2f}, z={position.z:.2f}\n'
            #     f'Orientation (quaternion): x={orientation.x:.2f}, y={orientation.y:.2f}, '
            #     f'z={orientation.z:.2f}, w={orientation.w:.2f}\n'
            #     f'Orientation (RPY radians): roll={roll:.2f}, pitch={pitch:.2f}, yaw={yaw:.2f}\n'
            #     f'Orientation (RPY degrees): roll={math.degrees(roll):.2f}, '
            #     f'pitch={math.degrees(pitch):.2f}, yaw={math.degrees(yaw):.2f}'
            # )
            # self.position_and_orientation  = [position.x,position.y,position.z,math.degrees(roll),math.degrees(pitch),math.degrees(yaw),orientation.w]
            self.position_and_orientation  = [position.x,position.y,position.z,orientation.x,orientation.y,orientation.z,orientation.w]

        except TransformException as ex:
            print(
                f'Could not transform {self.link_name} to world: {ex}')    
        return self.position_and_orientation
        # self.bridge = CvBridge()

    def send_goal_robot_arm(self, p_x,p_y,p_z,or_x,or_y,or_z,or_w,order):
        self.detect=False
        goal_msg = ArticubotTask.Goal()
        goal_msg.task = order
        goal_msg.p_x=p_x
        goal_msg.p_y=p_y
        goal_msg.p_z=p_z
        goal_msg.or_x=or_x
        goal_msg.or_y=or_y
        goal_msg.or_z=or_z
        goal_msg.or_w=or_w
        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_robot_arm_callback)
        self.get_logger().info('Result: {0}'.format(self._send_goal_future.add_done_callback(self.goal_response_robot_arm_callback)))
        self._send_goal_future.add_done_callback(self.goal_response_robot_arm_callback)
    def get_result_robot_arm_callback(self, future):
        result = future.result().result
        status = future.result().status
        self.get_logger().info('Result: {0}'.format(result.success))
        self.get_logger().info('Status: {0}'.format(status))
        self.detect=True
        



    def feedback_robot_arm_callback(self, feedback_msg):
            feedback = feedback_msg.feedback
            self.get_logger().info('Received feedback:',feedback)

    def goal_response_robot_arm_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return
        self.get_logger().info('Goal accepted :)')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_robot_arm_callback)  
        
class MainWindow(QMainWindow):
    def __init__(self, ros2_node):
        super().__init__()
        self.node = ros2_node
        self.initUI()
        self.point=[]
        self.publisher_joint = self.node.create_publisher(JointState, 'arm_robot', 10)
        self.publisher_twist = self.node.create_publisher(Twist, '/cmd_vel', 10)
        self.publisher_joint_commend = self.node.create_publisher(
            JointTrajectory, 
            '/arm_controller/follow_joint_trajectory', 
            10
        )
        self.action_client = ActionClient(self.node, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory')
        self.action_client_navigation = ActionClient(self.node, NavigateToPose, 'navigate_to_pose')
        self.bridge = CvBridge()       
        # Initialize camera subscriber
        self.camera_subscriber = self.node.create_subscription(
            Image,
            '/camera/camera/color/image_raw',  # Adjust this topic to match your camera's topic
            self.camera_callback,
            10)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_label_position)
        self.timer.start(100)  # Update every 1000 ms (1 second)  
        self.tts_engine = pyttsx3.init()
        self.processes = {}
        self.lock = threading.Lock()
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)


    def start_launches(self):
        for launch_cmd in self.launch_files:
            process = subprocess.Popen(launch_cmd)
            self.processes.append(process)
            print(f"Started launch: {' '.join(launch_cmd)}")

    def shutdown(self):
        for process in self.processes:
            process.terminate()
        for process in self.processes:
            process.wait()
        print("All launches have been terminated")

    def initUI(self):
        self.setWindowTitle('Arm Robot Controller')
        self.setGeometry(100, 100, 1280, 768)
        self.joint_names=['arm_base_forearm_joint', 'forearm_hand_1_joint', 'forearm_hand_2_joint', 'forearm_hand_3_joint', 'forearm_claw_joint']
        self.position_and_orientation_name=['Px','Py','Pz','Orx','Ory','Orz','Orw']
        main_layout = QHBoxLayout()
    
        # Control buttons layout
        control_layout = QVBoxLayout()
        video_layout = QVBoxLayout()
        position_layout=QHBoxLayout()
        main_button_layout=QHBoxLayout()
        button_control_layout= QVBoxLayout()
        button_control_navigation_layout= QVBoxLayout()
        control_position_arm_layout= QVBoxLayout()
        value_position_layout = QVBoxLayout()
        self.btn_forward = QPushButton('Move Forward', self)
        self.btn_backward = QPushButton('Move Backward', self)
        self.btn_left = QPushButton('Move Left', self)
        self.btn_right = QPushButton('Move Right', self)
        self.btn_stop = QPushButton('Stop', self)
        self.textbox_x = QLineEdit(self)
        self.textbox_x.resize(100,40)
        self.textbox_y = QLineEdit(self)
        self.textbox_y.resize(100,40)
        self.textbox_theta = QLineEdit(self)
        self.textbox_theta.resize(100,40)
        self.btn_run_navigation = QPushButton('Run Navigation', self)
        
        
        self.btn_home = QPushButton('Move to Home', self)
        self.btn_position1 = QPushButton('Close Gripper', self)
        self.btn_position2 = QPushButton('Open Gripper', self)
        self.launch_button = QPushButton("Launch Robot Recogntion")
        self.open_navigation_button = QPushButton("Open Robot Navigation")
        self.save_position_arm = QPushButton('Save Arm Position', self)
        self.btn_move_position_arm = QPushButton('Move Arm Position', self)
        self.button_quit = QPushButton("Quit", self)

        self.btn_forward.clicked.connect(self.move_forward)
        self.btn_backward.clicked.connect(self.move_backward)
        self.btn_left.clicked.connect(self.rotate_left)
        self.btn_right.clicked.connect(self.rotate_right)
        self.btn_stop.clicked.connect(self.stop)
        self.btn_home.clicked.connect(self.move_to_home)
        self.btn_position1.clicked.connect(self.move_to_position1)
        self.btn_position2.clicked.connect(self.move_to_position2)
        self.launch_button.clicked.connect(self.launch_robot_recognition)
        self.open_navigation_button.clicked.connect(self.launch_robot_navigation)
        self.save_position_arm.clicked.connect(self.save_position_arm_to_position)
        self.btn_move_position_arm.clicked.connect(self.control_arm_to_position)
        self.button_quit.clicked.connect(self.close)
        self.btn_run_navigation.clicked.connect(self.send_navigation_goal)
        
        
        
        button_control_layout.addWidget(self.btn_home)
        button_control_layout.addWidget(self.btn_position1)
        button_control_layout.addWidget(self.btn_position2)
        button_control_layout.addWidget(self.btn_forward)
        button_control_layout.addWidget(self.btn_backward)
        button_control_layout.addWidget(self.btn_left)
        button_control_layout.addWidget(self.btn_right)
        button_control_layout.addWidget(self.btn_stop)
        button_control_layout.addWidget(self.launch_button)
        button_control_layout.addWidget(self.open_navigation_button)  

        # control_layout.addStretch(1)

        control_position_arm_layout.addWidget(self.save_position_arm)  
        control_position_arm_layout.addWidget(self.btn_move_position_arm)  
        self.joint_sliders = []
        self.joint_labels = []
        self.position_labels = []
        control_layout.addLayout(position_layout)
        position_layout.addLayout(value_position_layout)
        position_layout.addLayout(control_position_arm_layout)

        for i in self.joint_names:  # Assuming 6 joints, adjust as needed

            
            label = QLabel(f"{i}:")
            control_layout.addWidget(label)

            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(-150)
            slider.setMaximum(150)
            slider.setValue(0)
            slider.setTickPosition(QSlider.TicksBelow)
            slider.setTickInterval(5)
            slider.valueChanged.connect(self.update_label)
            self.joint_sliders.append(slider)
            control_layout.addWidget(slider)

            value_label = QLabel("0.00")
            self.joint_labels.append(value_label)
            control_layout.addWidget(value_label)
            
        for i in range(7):  # Assuming 6 joints, adjust as needed

            self.label_position = QLabel(f"{self.position_and_orientation_name[i]}: ")
            self.position_labels.append(self.label_position )
            value_position_layout.addWidget(self.label_position )

        send_button = QPushButton("Send Command")
        send_button.clicked.connect(self.send_joint_command)
        control_layout.addWidget(send_button)





        # Video display
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)

        main_layout.addLayout(control_layout)
        control_layout.addLayout(main_button_layout)
        main_button_layout.addLayout(button_control_layout)
        main_button_layout.addLayout(button_control_navigation_layout)
        main_layout.addLayout(video_layout)
        video_layout.addWidget(self.video_label)

        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        video_layout.addWidget(self.status_text)
        video_layout.addWidget(self.button_quit)
        button_control_navigation_layout.addWidget(self.textbox_x)
        button_control_navigation_layout.addWidget(self.textbox_y)
        button_control_navigation_layout.addWidget(self.textbox_theta)
        button_control_navigation_layout.addWidget(self.btn_run_navigation)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
    
    def update_label_position(self):
        for i in range(7):  # Assuming 6 joints, adjust as needed

            self.position_labels[i].setText((f"{self.position_and_orientation_name[i]}: {self.node.position_and_orientation[i]:.4f}"))

    def save_position_arm_to_position(self):
        self.point=self.node.position_and_orientation
        # print(self.point)
        print(f"px={self.point[0]}, py={self.point[1]}, pz={self.point[2]}, orx={self.point[3]}, ory={self.point[4]}, orz={self.point[5]}, orw={self.point[6]}")
    def control_arm_to_position(self):

        self.node.send_goal_robot_arm(self.point[0],self.point[1],self.point[2],self.point[3],self.point[4],self.point[5],self.point[6],0)
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
        point.time_from_start.sec = 4  # Move to position in 2 seconds
        
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


    def move_to_home(self):
        self.node.send_goal_robot_arm(0.5,1.2,1.2,1.2,0.0,0.0,0.0,1)

    def move_to_position1(self):
        self.node.send_goal_robot_arm(self.point[0],self.point[1],self.point[2],self.point[3],self.point[4],self.point[5],self.point[6],2)

    def move_to_position2(self):
        self.node.send_goal_robot_arm(self.point[0],self.point[1],self.point[2],self.point[3],self.point[4],self.point[5],self.point[6],3)

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

    def start_node(self, package_name, node_name, namespace=None):
        cmd = ["ros2", "launch", package_name, node_name]
        if namespace:
            cmd = ["ros2", "launch", "--namespace", namespace, package_name, node_name]
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        with self.lock:
            self.processes[node_name] = process
        
        # Start threads to read output
        threading.Thread(target=self.read_output, args=(process.stdout, f"{node_name} [OUT]"), daemon=True).start()
        threading.Thread(target=self.read_output, args=(process.stderr, f"{node_name} [ERR]"), daemon=True).start()

        print(f"Started {node_name}")

    def read_output(self, pipe, prefix):
        for line in iter(pipe.readline, b''):
            print(f"{prefix}: {line.decode().strip()}")

    def stop_all_nodes(self):
        with self.lock:
            for name, process in self.processes.items():
                print(f"Stopping {name}")
                process.terminate()
            
            # Wait for processes to terminate
            for name, process in self.processes.items():
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"Force killing {name}")
                    process.kill()
            
            self.processes.clear()

    def signal_handler(self, signum, frame):
        print("Received signal to terminate")
        self.stop_all_nodes()
        sys.exit(0)

    def wait_for_nodes(self):
        while True:
            with self.lock:
                if not self.processes:
                    break
                for name, process in list(self.processes.items()):
                    if process.poll() is not None:
                        print(f"{name} has exited with return code {process.returncode}")
                        del self.processes[name]
            time.sleep(1)

    def launch_robot_recognition(self):
        try:

            self.start_node("robot_recognition", "launch_yolov8.launch.py")
            self.start_node("articubot_one", "online_async_launch.launch.py")
            
            self.start_launches()
            print("All Launched robot arm successfully")
        except subprocess.CalledProcessError as e:
            print(f"Failed to launch robot arm: {e}")
    def launch_robot_navigation(self):
        try:
            self.start_node("articubot_one", "navigation1.launch.py")
            print("All Launched robot arm successfully")
        except subprocess.CalledProcessError as e:
            print(f"Failed to launch robot arm: {e}")
    def close_robot_recognition(self):
        try:
            self.wait_for_nodes()
        except KeyboardInterrupt:
            print("Keyboard interrupt received")
        finally:
            self.stop_all_nodes()

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Window Close', 'Are you sure you want to close the window?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.close_robot_recognition()
            event.accept()
            QApplication.instance().quit()
            self.node.destroy_node()
            rclpy.shutdown()
        else:
            event.ignore()

    def send_navigation_goal(self) -> None:
        """
        Sends a navigation goal to the action server.

        :param location: A dictionary containing the location information (name, x, y, theta).
        """
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.stamp = self.node.get_clock().now().to_msg()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.pose.position.x = float(self.textbox_x.text())
        goal_msg.pose.pose.position.y = float(self.textbox_y.text())
        goal_msg.pose.pose.orientation.z = float(self.textbox_theta.text())

        print(f'Sending navigation goal: {"Position"},x= {self.textbox_x.text()},y= {self.textbox_x.text()}')
        self.tts_engine.say("Ok Mrson")
        self.tts_engine.runAndWait()
        self.action_client_navigation.wait_for_server()
        goal_handle = self.action_client_navigation.send_goal_async(goal_msg)
        goal_handle.add_done_callback(self.navigation_goal_done_callback)

    def navigation_goal_done_callback(self, future) -> None:
        """
        Callback function that is called when a navigation goal is done.

        :param future: A Future object that contains the result of the goal.
        """
        goal_handle = future.result()
        if not goal_handle.accepted:
            print('Goal rejected :(')
            return

        print('Goal accepted :)')
def main(args=None):
    rclpy.init(args=args)
    ros2_node = ArmRobotControllerNode()

    app = QApplication(sys.argv)
    main_window = MainWindow(ros2_node)
    main_window.show()

    # Set up a timer to handle ROS2 callbacks
    timer = QTimer()
    timer.timeout.connect(lambda: rclpy.spin_once(ros2_node, timeout_sec=0))
    timer.start(100)  # 100ms

    try:
        main_window.wait_for_nodes()
        sys.exit(app.exec_())
    finally:
        main_window.stop_all_nodes()
        ros2_node.destroy_node()

        rclpy.shutdown()

if __name__ == '__main__':
    main()