

from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import yaml
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def load_yaml(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)

    try:
        with open(absolute_file_path, 'r') as file:
            return yaml.safe_load(file)
    except EnvironmentError:  # parent of IOError, OSError *and* WindowsError where available
        return None
def load_file(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)

    try:
        with open(absolute_file_path, 'r') as file:
            return file.read()
    except EnvironmentError:  # parent of IOError, OSError *and* WindowsError where available
        return None

def generate_launch_description():

    use_sim_time = LaunchConfiguration('is_sim')
    
    is_sim_arg = DeclareLaunchArgument(
        'is_sim',
        default_value='False'
    )
    declare_loaded_description = DeclareLaunchArgument(
        'loaded_description',
        default_value='',
        description='Set robot_description text.  \
                     It is recommended to use RobotDescriptionLoader() in crane_plus_description.'
    )
    robot_description_content = Command(
        [
            FindExecutable(name="xacro"), " ",
            PathJoinSubstitution([FindPackageShare("articubot_one"), "description", "robot.urdf.xacro"]),
        ]
    )
    robot_description_semantic_config = load_file(
        'articubot_moveit', 'config/articubot.srdf')
    robot_description_semantic = {
        'robot_description_semantic': robot_description_semantic_config}
    trajectory_execution = {'moveit_manage_controllers': True,
                            'trajectory_execution.allowed_execution_duration_scaling': 1.2,
                            'trajectory_execution.allowed_goal_duration_margin': 0.5,
                            'trajectory_execution.allowed_start_tolerance': 0.1}

    planning_scene_monitor_parameters = {'publish_planning_scene': True,
                                         'publish_geometry_updates': True,
                                         'publish_state_updates': True,
                                         'publish_transforms_updates': True,
                                         "publish_transforms_updates": True,
                                         "monitor_dynamics": False,}
  
    controllers_yaml = load_yaml('articubot_moveit', 'config/moveit_controllers.yaml')
    moveit_controllers = {
        'moveit_simple_controller_manager': controllers_yaml,
        'moveit_controller_manager':
            'moveit_simple_controller_manager/MoveItSimpleControllerManager'}
    kinematics_yaml = load_yaml('articubot_moveit', 'config/kinematics.yaml')
    # Planning Functionality
    ompl_planning_pipeline_config = {'move_group': {
        'planning_plugin': 'ompl_interface/OMPLPlanner',
        'request_adapters': 'default_planner_request_adapters/AddTimeOptimalParameterization \
                               default_planner_request_adapters/FixWorkspaceBounds \
                               default_planner_request_adapters/FixStartStateBounds \
                               default_planner_request_adapters/FixStartStateCollision \
                               default_planner_request_adapters/FixStartStatePathConstraints',
        'start_state_max_bounds_error': 0.1}}
    ompl_planning_yaml = load_yaml('articubot_moveit', 'config/ompl_planning.yaml')
    ompl_planning_pipeline_config['move_group'].update(ompl_planning_yaml)
    # robot_description = {'robot_description': LaunchConfiguration('loaded_description')}
    robot_description = {'robot_description': robot_description_content}
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
                    # moveit_config.to_dict(), 
                    robot_description,
                    {'use_sim_time': False},
                    {'publish_robot_description_semantic': True},
                    kinematics_yaml,
                    ompl_planning_pipeline_config,
                    trajectory_execution,
                    planning_scene_monitor_parameters,
                    moveit_controllers,
                    robot_description_semantic,
                    {"wait_for_initial_state_timeout": 10.0},
                    {"move_group/trajectory_execution/allowed_start_tolerance": 0.01},
                    {"move_group/trajectory_execution/wait_for_trajectory_completion": True},
                    {"move_group/robot_state_monitor/robot_state_update_interval": 0.1},
                    {"move_group/robot_state_monitor/robot_state_update_timeout": 5.0}, 

                    ],
                    
        arguments=["--ros-args", "--log-level", "info"],
    )


    # RViz
    rviz_config = os.path.join(
        get_package_share_directory("articubot_moveit"),
            "config",
            "moveit1.rviz",
    )
    rviz_node = Node(package='rviz2',
                     executable='rviz2',
                     name='rviz2',
                     output='log',
                     arguments=['-d', rviz_config],
                     parameters=[robot_description,
                                 robot_description_semantic,
                                 ompl_planning_pipeline_config,
                                 kinematics_yaml])

    static_tf = Node(package='tf2_ros',
                     executable='static_transform_publisher',
                     name='static_transform_publisher',
                     output='log',
                     arguments=['0.0', '0.0', '0.0', '0.0', '0.0', '0.0', 'base_link', 'arm_base_link'])

    remote_interface = IncludeLaunchDescription(
            os.path.join(
                get_package_share_directory("articubot_remote"),
                "launch",
                "remote_interface.launch.py"
            ),
        )
    arm_pnp_as = Node(
            name='articubot_node',
            package='articubot_remote',
            executable='test_server_node',
            output='screen',
            parameters=[
                    robot_description,
                    {'use_sim_time': True},
                    {'publish_robot_description_semantic': True},
                    kinematics_yaml,
                    ompl_planning_pipeline_config,
                    trajectory_execution,
                    planning_scene_monitor_parameters,
                    moveit_controllers,
                    robot_description_semantic,
                    {"wait_for_initial_state_timeout": 10.0},
            ],
            # condition=IfCondition(use_pnp)
        )

    return LaunchDescription(
        [   declare_loaded_description,
            is_sim_arg,

            move_group_node, 
            # rviz_node,
            # remote_interface,
            static_tf,
            arm_pnp_as,

            
        ]
    )









# import os

# from ament_index_python.packages import get_package_share_directory
# from launch import LaunchDescription
# from launch.actions import DeclareLaunchArgument
# from launch.substitutions import LaunchConfiguration
# from launch_ros.actions import Node
# import yaml
# from launch.actions import IncludeLaunchDescription

# def load_file(package_name, file_path):
#     package_path = get_package_share_directory(package_name)
#     absolute_file_path = os.path.join(package_path, file_path)

#     try:
#         with open(absolute_file_path, 'r') as file:
#             return file.read()
#     except EnvironmentError:  # parent of IOError, OSError *and* WindowsError where available
#         return None


# def load_yaml(package_name, file_path):
#     package_path = get_package_share_directory(package_name)
#     absolute_file_path = os.path.join(package_path, file_path)

#     try:
#         with open(absolute_file_path, 'r') as file:
#             return yaml.safe_load(file)
#     except EnvironmentError:  # parent of IOError, OSError *and* WindowsError where available
#         return None


# def generate_launch_description():
#     declare_loaded_description = DeclareLaunchArgument(
#         'loaded_description',
#         default_value='',
#         description='Set robot_description text.  \
#                      It is recommended to use RobotDescriptionLoader() in crane_plus_description.'
#     )

#     declare_rviz_config_file = DeclareLaunchArgument(
#         'rviz_config_file',
#         default_value=get_package_share_directory(
#             'articubot_moveit') + '/config/moveit1.rviz',
#         description='Set the path to rviz configuration file.'
#     )

#     robot_description = {'robot_description': LaunchConfiguration('loaded_description')}

#     robot_description_semantic_config = load_file(
#         'articubot_moveit', 'config/articubot.srdf')
#     robot_description_semantic = {
#         'robot_description_semantic': robot_description_semantic_config}

#     kinematics_yaml = load_yaml('articubot_moveit', 'config/kinematics.yaml')
#     # Planning Functionality
#     ompl_planning_pipeline_config = {'move_group': {
#         'planning_plugin': 'ompl_interface/OMPLPlanner',
#         'request_adapters': 'default_planner_request_adapters/AddTimeOptimalParameterization \
#                                default_planner_request_adapters/FixWorkspaceBounds \
#                                default_planner_request_adapters/FixStartStateBounds \
#                                default_planner_request_adapters/FixStartStateCollision \
#                                default_planner_request_adapters/FixStartStatePathConstraints',
#         'start_state_max_bounds_error': 0.1}}
#     ompl_planning_yaml = load_yaml('articubot_moveit', 'config/ompl_planning.yaml')
#     ompl_planning_pipeline_config['move_group'].update(ompl_planning_yaml)

#     # Trajectory Execution Functionality
#     controllers_yaml = load_yaml('articubot_moveit', 'config/moveit_controllers.yaml')
#     moveit_controllers = {
#         'moveit_simple_controller_manager': controllers_yaml,
#         'moveit_controller_manager':
#             'moveit_simple_controller_manager/MoveItSimpleControllerManager'}

#     trajectory_execution = {'moveit_manage_controllers': True,
#                             'trajectory_execution.allowed_execution_duration_scaling': 1.2,
#                             'trajectory_execution.allowed_goal_duration_margin': 0.5,
#                             'trajectory_execution.allowed_start_tolerance': 0.1}

#     planning_scene_monitor_parameters = {'publish_planning_scene': True,
#                                          'publish_geometry_updates': True,
#                                          'publish_state_updates': True,
#                                          'publish_transforms_updates': True}

#     # Start the actual move_group node/action server
#     run_move_group_node = Node(package='moveit_ros_move_group',
#                                executable='move_group',
#                                output='screen',
#                                parameters=[robot_description,
#                                            robot_description_semantic,
#                                            kinematics_yaml,
#                                            ompl_planning_pipeline_config,
#                                            trajectory_execution,
#                                            moveit_controllers,
#                                            planning_scene_monitor_parameters,
#                                            {'publish_robot_description_semantic': True}])

#     # RViz
#     rviz_config_file = LaunchConfiguration('rviz_config_file')
#     rviz_node = Node(package='rviz2',
#                      executable='rviz2',
#                      name='rviz2',
#                      output='log',
#                      arguments=['-d', rviz_config_file],
#                      parameters=[robot_description,
#                                  robot_description_semantic,
#                                  ompl_planning_pipeline_config,
#                                  kinematics_yaml])

#     # Static TF
#     static_tf = Node(package='tf2_ros',
#                      executable='static_transform_publisher',
#                      name='static_transform_publisher',
#                      output='log',
#                      arguments=['0.0', '0.0', '0.0', '0.0', '0.0', '0.0', 'base_link', 'arm_base_link'])

#     # Publish TF
#     robot_state_publisher = Node(package='robot_state_publisher',
#                                  executable='robot_state_publisher',
#                                  name='robot_state_publisher',
#                                  output='both',
#                                  parameters=[robot_description])
#     remote_interface = IncludeLaunchDescription(
#             os.path.join(
#                 get_package_share_directory("articubot_remote"),
#                 "launch",
#                 "remote_interface.launch.py"
#             ),
#         )
#     return LaunchDescription([declare_loaded_description,
#                               declare_rviz_config_file,
#                               run_move_group_node,
#                               rviz_node,
#                               static_tf,
#                               robot_state_publisher,
#                             #   remote_interface,

#                               ])
