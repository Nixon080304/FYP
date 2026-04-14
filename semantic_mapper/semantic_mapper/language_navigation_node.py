#!/usr/bin/env python3

import math
from typing import List, Optional

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav2_msgs.action import NavigateToPose
from tf2_ros import Buffer, TransformListener, TransformException

from semantic_interfaces.msg import SemanticMapObjectArray


class LanguageNavigationNode(Node):
    def __init__(self):
        super().__init__('language_navigation_node')

        self.declare_parameter('semantic_objects_topic', '/semantic_map/objects')
        self.declare_parameter('command_topic', '/language_command')
        self.declare_parameter('goal_offset_distance', 0.8)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('robot_frame', 'base_link')

        self.semantic_objects_topic = self.get_parameter('semantic_objects_topic').value
        self.command_topic = self.get_parameter('command_topic').value
        self.goal_offset_distance = float(self.get_parameter('goal_offset_distance').value)
        self.map_frame = self.get_parameter('map_frame').value
        self.robot_frame = self.get_parameter('robot_frame').value

        self.semantic_objects: List = []

        self.semantic_sub = self.create_subscription(
            SemanticMapObjectArray,
            self.semantic_objects_topic,
            self.semantic_objects_callback,
            10
        )

        self.command_sub = self.create_subscription(
            String,
            self.command_topic,
            self.command_callback,
            10
        )

        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.supported_aliases = {
            'person': 'person',
            'bicycle': 'bicycle',
            'bike': 'bicycle',
            'car': 'car',
            'bed': 'bed',
            'chair': 'chair',
            'table': 'dining table',
            'dining table': 'dining table',
        }

        self.get_logger().info('Language navigation node started.')
        self.get_logger().info(f'Semantic objects topic: {self.semantic_objects_topic}')
        self.get_logger().info(f'Command topic: {self.command_topic}')

    def semantic_objects_callback(self, msg: SemanticMapObjectArray):
        self.semantic_objects = list(msg.objects)

    def command_callback(self, msg: String):
        command = msg.data.strip().lower()

        if not command:
            self.get_logger().warn('Received empty command.')
            return

        self.get_logger().info(f'Received command: "{command}"')

        target_class = self.extract_target_class(command)
        if target_class is None:
            self.get_logger().warn('Could not find target object class in command.')
            return

        target_object = self.find_target_object(target_class)
        if target_object is None:
            self.get_logger().warn(f'No confirmed semantic object found for class: {target_class}')
            return

        robot_pose = self.get_robot_pose()
        if robot_pose is None:
            self.get_logger().warn('Could not get robot pose from TF.')
            return

        robot_x, robot_y = robot_pose
        obj_x = target_object.pose.position.x
        obj_y = target_object.pose.position.y

        goal_pose = self.make_goal_near_object(robot_x, robot_y, obj_x, obj_y)

        self.get_logger().info(
            f'Target object "{target_class}" at ({obj_x:.2f}, {obj_y:.2f}) | '
            f'Robot at ({robot_x:.2f}, {robot_y:.2f})'
        )

        self.send_nav_goal(goal_pose, target_class)

    def extract_target_class(self, command: str) -> Optional[str]:
        for phrase, class_name in self.supported_aliases.items():
            if phrase in command:
                return class_name
        return None

    def find_target_object(self, target_class: str):
        candidates = [
            obj for obj in self.semantic_objects
            if obj.confirmed and obj.class_name == target_class
        ]

        if not candidates:
            return None

        # Choose highest-confidence confirmed object
        candidates.sort(key=lambda obj: obj.confidence, reverse=True)
        return candidates[0]

    def get_robot_pose(self) -> Optional[tuple]:
        try:
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.robot_frame,
                rclpy.time.Time()
            )
        except TransformException:
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.map_frame,
                    'base_footprint',
                    rclpy.time.Time()
                )
            except TransformException as e:
                self.get_logger().warn(f'Failed to get robot pose transform: {e}')
                return None

        x = transform.transform.translation.x
        y = transform.transform.translation.y
        return x, y

    def make_goal_near_object(
        self,
        robot_x: float,
        robot_y: float,
        obj_x: float,
        obj_y: float
    ) -> PoseStamped:
        goal = PoseStamped()
        goal.header.frame_id = self.map_frame
        goal.header.stamp = self.get_clock().now().to_msg()

        dx = robot_x - obj_x
        dy = robot_y - obj_y
        dist = math.hypot(dx, dy)

        if dist < 1e-6:
            # fallback if robot is basically at the object already
            unit_x = 1.0
            unit_y = 0.0
        else:
            unit_x = dx / dist
            unit_y = dy / dist

        goal_x = obj_x + self.goal_offset_distance * unit_x
        goal_y = obj_y + self.goal_offset_distance * unit_y

        yaw = math.atan2(obj_y - goal_y, obj_x - goal_x)

        goal.pose.position.x = goal_x
        goal.pose.position.y = goal_y
        goal.pose.position.z = 0.0
        goal.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.orientation.w = math.cos(yaw / 2.0)

        self.get_logger().info(
            f'Computed goal near object: ({goal_x:.2f}, {goal_y:.2f}), yaw={yaw:.2f} rad'
        )

        return goal

    def send_nav_goal(self, goal_pose: PoseStamped, target_class: str):
        if not self.nav_to_pose_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error('NavigateToPose action server not available.')
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal_pose

        self.get_logger().info(
            f'Sending Nav2 goal near {target_class}: '
            f'({goal_pose.pose.position.x:.2f}, {goal_pose.pose.position.y:.2f})'
        )

        send_future = self.nav_to_pose_client.send_goal_async(goal_msg)
        send_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        try:
            goal_handle = future.result()
        except Exception as e:
            self.get_logger().error(f'Failed to send goal: {e}')
            return

        if not goal_handle.accepted:
            self.get_logger().warn('Language navigation goal was rejected by Nav2.')
            return

        self.get_logger().info('Language navigation goal accepted by Nav2.')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_result_callback)

    def goal_result_callback(self, future):
        try:
            result_msg = future.result()
        except Exception as e:
            self.get_logger().error(f'Failed to get navigation result: {e}')
            return

        status = result_msg.status
        result = result_msg.result

        error_code = getattr(result, 'error_code', None)
        error_msg = getattr(result, 'error_msg', '')

        if error_code is not None:
            self.get_logger().info(
                f'Language navigation finished | status={status} | '
                f'error_code={error_code} | error_msg="{error_msg}"'
            )
        else:
            self.get_logger().info(
                f'Language navigation finished | status={status}'
            )


def main(args=None):
    rclpy.init(args=args)
    node = LanguageNavigationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()