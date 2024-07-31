from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    qos_override_path = os.path.join(
        get_package_share_directory('articubot_remote'),
        'config',
        'qos_override.yaml'
    )
    # task_server_node = Node(
    #     package="articubot_remote",
    #     executable="move_to_point_action_server.py",
    # )
    task_server_node_py = Node(
        package="articubot_remote",
        executable="task_server.py",
    )   
    test_server_node = Node(
        package="articubot_remote",
        executable="test_server_node",
        parameters=[qos_override_path]
    )
    articubot_server_node = Node(
        package="articubot_remote",
        executable="articubot_server_node",
    )
    alexa_interface_node = Node(
        package="articubot_remote",
        executable="alexa_interface.py",
    )

    return LaunchDescription([
        test_server_node,
        # task_server_node_py,
        # alexa_interface_node,
        # articubot_server_node
    ])
