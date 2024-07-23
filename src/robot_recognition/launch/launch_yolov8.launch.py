from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node

def generate_launch_description():

    return LaunchDescription([
        # Node(package='robot_recognition', executable='yolov8_ros2_pt.py', output='screen'),
        # Node(package='robot_recognition', executable='yolo_ros2_depth_test.py', output='screen'),
        Node(package='robot_recognition', executable='robot_ros2_reg.py', output='screen'),
        # Node(package='robot_recognition', executable='webcam.py', output='screen'),
    ])