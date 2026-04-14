#!/usr/bin/env python3

import math
from dataclasses import dataclass
from typing import List, Optional

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Pose, PoseArray
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from semantic_interfaces.msg import ProjectedSemanticObjectArray, SemanticMapObject, SemanticMapObjectArray


@dataclass
class SemanticTrack:
    track_id: int
    class_name: str
    x: float
    y: float
    z: float
    confidence: float
    first_seen: Time
    last_seen: Time
    observation_count: int = 1
    confirmed: bool = False


class SemanticMapNode(Node):
    def __init__(self) -> None:
        super().__init__('semantic_map_node')

        self.declare_parameter('input_topic', '/semantic_objects/projected')
        self.declare_parameter('marker_topic', '/semantic_map/markers')
        self.declare_parameter('pose_topic', '/semantic_map/poses')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('objects_topic', '/semantic_map/objects')

        self.declare_parameter('min_confidence', 0.45)
        self.declare_parameter('merge_distance', 0.60)
        self.declare_parameter('min_observations', 2)
        self.declare_parameter('track_timeout', 8.0)
        self.declare_parameter('position_alpha', 0.35)
        self.declare_parameter('publish_rate', 2.0)

        self.declare_parameter('marker_scale', 0.22)
        self.declare_parameter('text_scale', 0.20)

        self.input_topic = self.get_parameter('input_topic').value
        self.marker_topic = self.get_parameter('marker_topic').value
        self.pose_topic = self.get_parameter('pose_topic').value
        self.map_frame = self.get_parameter('map_frame').value
        self.objects_topic = self.get_parameter('objects_topic').value

        self.min_confidence = float(self.get_parameter('min_confidence').value)
        self.merge_distance = float(self.get_parameter('merge_distance').value)
        self.min_observations = int(self.get_parameter('min_observations').value)
        self.track_timeout = float(self.get_parameter('track_timeout').value)
        self.position_alpha = float(self.get_parameter('position_alpha').value)
        self.publish_rate = float(self.get_parameter('publish_rate').value)
        self.marker_scale = float(self.get_parameter('marker_scale').value)
        self.text_scale = float(self.get_parameter('text_scale').value)

        self.tracks: List[SemanticTrack] = []
        self.next_track_id = 0

        self.subscription = self.create_subscription(
            ProjectedSemanticObjectArray,
            self.input_topic,
            self.projected_callback,
            10
        )

        self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, 10)
        self.pose_pub = self.create_publisher(PoseArray, self.pose_topic, 10)
        self.objects_pub = self.create_publisher(SemanticMapObjectArray, self.objects_topic, 10)

        self.timer = self.create_timer(1.0 / self.publish_rate, self.publish_outputs)

        self.get_logger().info('Semantic map node started.')
        self.get_logger().info(f'Input topic: {self.input_topic}')
        self.get_logger().info(f'Marker topic: {self.marker_topic}')
        self.get_logger().info(f'Pose topic: {self.pose_topic}')
        self.get_logger().info(f'Objects topic: {self.objects_topic}')

    def projected_callback(self, msg: ProjectedSemanticObjectArray) -> None:
        now = self.get_clock().now()

        for obj in msg.objects:
            class_name = obj.class_name.strip()
            confidence = float(obj.confidence)

            if not class_name:
                continue

            if confidence < self.min_confidence:
                continue

            x = obj.pose.position.x
            y = obj.pose.position.y
            z = obj.pose.position.z

            matched_track = self.find_matching_track(class_name, x, y, z)

            if matched_track is None:
                self.create_track(class_name, x, y, z, confidence, now)
            else:
                self.update_track(matched_track, x, y, z, confidence, now)

        self.remove_stale_tracks(now)

    def find_matching_track(self, class_name: str, x: float, y: float, z: float) -> Optional[SemanticTrack]:
        best_track: Optional[SemanticTrack] = None
        best_distance = float('inf')

        for track in self.tracks:
            if track.class_name != class_name:
                continue

            distance = self.euclidean_distance(track.x, track.y, track.z, x, y, z)

            if distance < self.merge_distance and distance < best_distance:
                best_distance = distance
                best_track = track

        return best_track

    def create_track(self, class_name: str, x: float, y: float, z: float, confidence: float, now: Time) -> None:
        track = SemanticTrack(
            track_id=self.next_track_id,
            class_name=class_name,
            x=x,
            y=y,
            z=z,
            confidence=confidence,
            first_seen=now,
            last_seen=now,
            observation_count=1,
            confirmed=False
        )

        self.tracks.append(track)
        self.next_track_id += 1

        self.get_logger().info(
            f'New tentative object | id={track.track_id} class={track.class_name} '
            f'pos=({track.x:.2f}, {track.y:.2f}, {track.z:.2f}) conf={track.confidence:.2f}'
        )

    def update_track(self, track: SemanticTrack, x: float, y: float, z: float, confidence: float, now: Time) -> None:
        alpha = self.position_alpha

        track.x = (1.0 - alpha) * track.x + alpha * x
        track.y = (1.0 - alpha) * track.y + alpha * y
        track.z = (1.0 - alpha) * track.z + alpha * z
        track.confidence = max(track.confidence, confidence)
        track.last_seen = now
        track.observation_count += 1

        if not track.confirmed and track.observation_count >= self.min_observations:
            track.confirmed = True
            self.get_logger().info(
                f'Confirmed object | id={track.track_id} class={track.class_name} '
                f'obs={track.observation_count} pos=({track.x:.2f}, {track.y:.2f}, {track.z:.2f})'
            )

    def remove_stale_tracks(self, now: Time) -> None:
        alive_tracks: List[SemanticTrack] = []
        removed_tracks: List[SemanticTrack] = []

        for track in self.tracks:
            age_sec = (now - track.last_seen).nanoseconds / 1e9

            # Keep confirmed objects as persistent memory
            if track.confirmed:
                alive_tracks.append(track)
                continue

            # Only remove stale tentative objects
            if age_sec <= self.track_timeout:
                alive_tracks.append(track)
            else:
                removed_tracks.append(track)

        self.tracks = alive_tracks

        for track in removed_tracks:
            self.get_logger().info(
                f'Removed stale tentative object | id={track.track_id} class={track.class_name}'
            )

    def publish_outputs(self) -> None:
        now_msg = self.get_clock().now().to_msg()
        confirmed_tracks = [track for track in self.tracks if track.confirmed]

        pose_array = PoseArray()
        pose_array.header.frame_id = self.map_frame
        pose_array.header.stamp = now_msg

        marker_array = MarkerArray()

        semantic_array = SemanticMapObjectArray()
        semantic_array.header.frame_id = self.map_frame
        semantic_array.header.stamp = now_msg

        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)

        for track in confirmed_tracks:
            pose = self.make_pose(track.x, track.y, track.z)
            pose_array.poses.append(pose)

            semantic_obj = SemanticMapObject()
            semantic_obj.class_name = track.class_name
            semantic_obj.confidence = float(track.confidence)
            semantic_obj.pose = pose
            semantic_obj.observation_count = int(track.observation_count)
            semantic_obj.confirmed = bool(track.confirmed)
            semantic_array.objects.append(semantic_obj)

            marker_array.markers.append(self.make_sphere_marker(track, now_msg))
            marker_array.markers.append(self.make_text_marker(track, now_msg))

        self.pose_pub.publish(pose_array)
        self.marker_pub.publish(marker_array)
        self.objects_pub.publish(semantic_array)

    def make_pose(self, x: float, y: float, z: float) -> Pose:
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation.w = 1.0
        return pose

    def make_sphere_marker(self, track: SemanticTrack, stamp) -> Marker:
        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = stamp
        marker.ns = 'semantic_objects'
        marker.id = track.track_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = track.x
        marker.pose.position.y = track.y
        marker.pose.position.z = max(track.z, 0.05)
        marker.pose.orientation.w = 1.0

        marker.scale.x = self.marker_scale
        marker.scale.y = self.marker_scale
        marker.scale.z = self.marker_scale

        marker.color = self.class_color(track.class_name)
        marker.lifetime = Duration(sec=0, nanosec=0)

        return marker

    def make_text_marker(self, track: SemanticTrack, stamp) -> Marker:
        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = stamp
        marker.ns = 'semantic_labels'
        marker.id = 10000 + track.track_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD

        marker.pose.position.x = track.x
        marker.pose.position.y = track.y
        marker.pose.position.z = max(track.z, 0.05) + 0.35
        marker.pose.orientation.w = 1.0

        marker.scale.z = self.text_scale
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        marker.text = f'{track.class_name} [{track.observation_count}]'
        marker.lifetime = Duration(sec=0, nanosec=0)

        return marker

    def class_color(self, class_name: str) -> ColorRGBA:
        value = abs(hash(class_name)) % 6

        color = ColorRGBA()
        color.a = 0.9

        if value == 0:
            color.r, color.g, color.b = 1.0, 0.2, 0.2
        elif value == 1:
            color.r, color.g, color.b = 0.2, 1.0, 0.2
        elif value == 2:
            color.r, color.g, color.b = 0.2, 0.4, 1.0
        elif value == 3:
            color.r, color.g, color.b = 1.0, 1.0, 0.2
        elif value == 4:
            color.r, color.g, color.b = 1.0, 0.3, 1.0
        else:
            color.r, color.g, color.b = 0.2, 1.0, 1.0

        return color

    @staticmethod
    def euclidean_distance(x1: float, y1: float, z1: float, x2: float, y2: float, z2: float) -> float:
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SemanticMapNode()

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
