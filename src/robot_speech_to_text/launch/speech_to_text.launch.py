from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node

def generate_launch_description():

    return LaunchDescription([

        # Node(package='robot_speech_to_text', executable='speech_to_text_node', output='screen'),
        Node(package='robot_speech_to_text', executable='speech_to_text_whisper_node', output='screen'),
    ])