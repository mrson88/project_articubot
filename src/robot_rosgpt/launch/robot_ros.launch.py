from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node

def generate_launch_description():

    return LaunchDescription([

        Node(package='robot_rosgpt', executable='robot_navigation_node', output='screen'),
    ])