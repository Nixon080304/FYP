from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node


def generate_launch_description():
    world = PathJoinSubstitution([
        FindPackageShare('simulation_bringup'),
        'worlds',
        'semantic_rect_world.world'
    ])

    gazebo_launch = PathJoinSubstitution([
        FindPackageShare('gazebo_ros'),
        'launch',
        'gazebo.launch.py'
    ])

    robot_state_publisher_launch = PathJoinSubstitution([
        FindPackageShare('turtlebot3_gazebo'),
        'launch',
        'robot_state_publisher.launch.py'
    ])

    camera_link_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=[
            '--x', '-0.025',
            '--y', '0.0',
            '--z', '1.0',
            '--roll', '0.0',
            '--pitch', '0.35',
            '--yaw', '0.0',
            '--frame-id', 'base_link',
            '--child-frame-id', 'camera_link'
        ],
        output='screen'
    )

    camera_optical_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=[
            '--x', '0.0',
            '--y', '0.0',
            '--z', '0.0',
            '--roll', '-1.5708',
            '--pitch', '0.0',
            '--yaw', '-1.5708',
            '--frame-id', 'camera_link',
            '--child-frame-id', 'camera_optical_frame'
        ],
        output='screen'
    )

    return LaunchDescription([
        SetEnvironmentVariable(
            name='TURTLEBOT3_MODEL',
            value='burger'
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gazebo_launch),
            launch_arguments={
                'world': world
            }.items()
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(robot_state_publisher_launch),
            launch_arguments={
                'use_sim_time': 'true'
            }.items()
        ),

        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
                '-entity', 'burger',
                '-file', '/opt/ros/humble/share/turtlebot3_gazebo/models/turtlebot3_burger/model.sdf',
                '-x', '0.0',
                '-y', '0.0',
                '-z', '0.022'
            ],
            output='screen'
        ),

        camera_link_tf,
        camera_optical_tf,
    ])