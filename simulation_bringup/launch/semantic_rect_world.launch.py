from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node


def generate_launch_description():
    robot_model = LaunchConfiguration('robot_model')
    x_pose = LaunchConfiguration('x_pose')
    y_pose = LaunchConfiguration('y_pose')
    z_pose = LaunchConfiguration('z_pose')

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

    robot_model_file = PathJoinSubstitution([
        FindPackageShare('turtlebot3_gazebo'),
        'models',
        PythonExpression(["'turtlebot3_' + '", robot_model, "'"]),
        'model.sdf'
    ])

    camera_link_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=[
            '--x', '-0.025',
            '--y', '0.0',
            '--z', '1.0',
            '--roll', '0.0',
            '--pitch', '0.0',
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
        DeclareLaunchArgument(
            'robot_model',
            default_value='burger',
            description='TurtleBot3 model to spawn: burger, waffle, or waffle_pi'
        ),
        DeclareLaunchArgument(
            'x_pose',
            default_value='0.0',
            description='Initial robot x position in Gazebo'
        ),
        DeclareLaunchArgument(
            'y_pose',
            default_value='0.0',
            description='Initial robot y position in Gazebo'
        ),
        DeclareLaunchArgument(
            'z_pose',
            default_value='0.022',
            description='Initial robot z position in Gazebo'
        ),
        SetEnvironmentVariable(
            name='TURTLEBOT3_MODEL',
            value=robot_model
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
                '-entity', robot_model,
                '-file', robot_model_file,
                '-x', x_pose,
                '-y', y_pose,
                '-z', z_pose
            ],
            output='screen'
        ),

        camera_link_tf,
        camera_optical_tf,
    ])
