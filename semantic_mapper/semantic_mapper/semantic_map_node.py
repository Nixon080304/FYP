#!/usr/bin/env python3

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Pose, PoseArray
from nav_msgs.msg import OccupancyGrid
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
    class_scores: Dict[str, float] = field(default_factory=dict)
    class_counts: Dict[str, int] = field(default_factory=dict)


class SemanticMapNode(Node):
    def __init__(self) -> None:
        super().__init__('semantic_map_node')

        self.declare_parameter('input_topic', '/semantic_objects/projected')
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('marker_topic', '/semantic_map/markers')
        self.declare_parameter('pose_topic', '/semantic_map/poses')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('objects_topic', '/semantic_map/objects')

        self.declare_parameter('min_confidence', 0.45)
        self.declare_parameter('merge_distance', 0.60)
        self.declare_parameter('confirmed_merge_distance', 1.80)
        self.declare_parameter('cross_class_match_distance', 0.75)
        self.declare_parameter('confirmed_cross_class_match_distance', 1.00)
        self.declare_parameter('min_observations', 2)
        self.declare_parameter('track_timeout', 8.0)
        self.declare_parameter('position_alpha', 0.35)
        self.declare_parameter('publish_rate', 2.0)
        self.declare_parameter('max_confirmed_tracks_per_class', 1)
        self.declare_parameter('occupied_threshold', 50)
        self.declare_parameter('map_boundary_margin_cells', 2)
        self.declare_parameter('support_radius_cells', 2)
        self.declare_parameter('min_supporting_occupied_cells', 2)

        self.declare_parameter('marker_scale', 0.22)
        self.declare_parameter('text_scale', 0.20)

        self.input_topic = self.get_parameter('input_topic').value
        self.map_topic = self.get_parameter('map_topic').value
        self.marker_topic = self.get_parameter('marker_topic').value
        self.pose_topic = self.get_parameter('pose_topic').value
        self.map_frame = self.get_parameter('map_frame').value
        self.objects_topic = self.get_parameter('objects_topic').value

        self.min_confidence = float(self.get_parameter('min_confidence').value)
        self.merge_distance = float(self.get_parameter('merge_distance').value)
        self.confirmed_merge_distance = float(self.get_parameter('confirmed_merge_distance').value)
        self.cross_class_match_distance = float(self.get_parameter('cross_class_match_distance').value)
        self.confirmed_cross_class_match_distance = float(self.get_parameter('confirmed_cross_class_match_distance').value)
        self.min_observations = int(self.get_parameter('min_observations').value)
        self.track_timeout = float(self.get_parameter('track_timeout').value)
        self.position_alpha = float(self.get_parameter('position_alpha').value)
        self.publish_rate = float(self.get_parameter('publish_rate').value)
        self.max_confirmed_tracks_per_class = int(
            self.get_parameter('max_confirmed_tracks_per_class').value
        )
        self.occupied_threshold = int(self.get_parameter('occupied_threshold').value)
        self.map_boundary_margin_cells = int(self.get_parameter('map_boundary_margin_cells').value)
        self.support_radius_cells = int(self.get_parameter('support_radius_cells').value)
        self.min_supporting_occupied_cells = int(self.get_parameter('min_supporting_occupied_cells').value)
        self.marker_scale = float(self.get_parameter('marker_scale').value)
        self.text_scale = float(self.get_parameter('text_scale').value)

        self.tracks: List[SemanticTrack] = []
        self.next_track_id = 0
        self.latest_map: Optional[OccupancyGrid] = None

        self.subscription = self.create_subscription(
            ProjectedSemanticObjectArray,
            self.input_topic,
            self.projected_callback,
            10
        )
        self.map_subscription = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_callback,
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

    def map_callback(self, msg: OccupancyGrid) -> None:
        self.latest_map = msg

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

            if not self.is_map_consistent(x, y):
                self.get_logger().info(
                    f'Rejected projected object outside plausible map area | '
                    f'class={class_name} pos=({x:.2f}, {y:.2f}, {z:.2f})',
                    throttle_duration_sec=1.0
                )
                continue

            matched_track = self.find_matching_track(class_name, x, y, z)

            if matched_track is None:
                self.create_track(class_name, x, y, z, confidence, now)
            else:
                self.update_track(matched_track, class_name, x, y, z, confidence, now)

        self.consolidate_tracks()
        self.prune_invalid_tracks()
        self.limit_confirmed_tracks_per_class()
        self.remove_stale_tracks(now)

    def find_matching_track(self, class_name: str, x: float, y: float, z: float) -> Optional[SemanticTrack]:
        same_class_track = self.find_nearest_track(
            x,
            y,
            z,
            class_name=class_name
        )
        if same_class_track is not None:
            return same_class_track

        return self.find_nearest_track(
            x,
            y,
            z,
            class_name=None
        )

    def find_nearest_track(
        self,
        x: float,
        y: float,
        z: float,
        class_name: Optional[str]
    ) -> Optional[SemanticTrack]:
        best_track: Optional[SemanticTrack] = None
        best_distance = float('inf')

        for track in self.tracks:
            if class_name is not None and track.class_name != class_name:
                continue

            distance = self.euclidean_distance(track.x, track.y, track.z, x, y, z)
            if class_name is None:
                allowed_distance = (
                    self.confirmed_cross_class_match_distance
                    if track.confirmed else
                    self.cross_class_match_distance
                )
            else:
                allowed_distance = (
                    self.confirmed_merge_distance
                    if track.confirmed else
                    self.merge_distance
                )

            if distance < allowed_distance and distance < best_distance:
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
            confirmed=False,
            class_scores={class_name: confidence},
            class_counts={class_name: 1}
        )

        self.tracks.append(track)
        self.next_track_id += 1

        self.get_logger().info(
            f'New tentative object | id={track.track_id} class={track.class_name} '
            f'pos=({track.x:.2f}, {track.y:.2f}, {track.z:.2f}) conf={track.confidence:.2f}'
        )

    def update_track(
        self,
        track: SemanticTrack,
        detected_class_name: str,
        x: float,
        y: float,
        z: float,
        confidence: float,
        now: Time
    ) -> None:
        alpha = self.position_alpha
        previous_class = track.class_name

        track.x = (1.0 - alpha) * track.x + alpha * x
        track.y = (1.0 - alpha) * track.y + alpha * y
        track.z = (1.0 - alpha) * track.z + alpha * z
        track.confidence = max(track.confidence, confidence)
        track.last_seen = now
        track.observation_count += 1
        track.class_scores[detected_class_name] = (
            track.class_scores.get(detected_class_name, 0.0) + confidence
        )
        track.class_counts[detected_class_name] = (
            track.class_counts.get(detected_class_name, 0) + 1
        )

        self.refresh_track_class(track)

        if previous_class != track.class_name:
            self.get_logger().info(
                f'Reclassified object | id={track.track_id} '
                f'{previous_class} -> {track.class_name}'
            )

        if not track.confirmed and track.observation_count >= self.min_observations:
            track.confirmed = True
            self.get_logger().info(
                f'Confirmed object | id={track.track_id} class={track.class_name} '
                f'obs={track.observation_count} pos=({track.x:.2f}, {track.y:.2f}, {track.z:.2f})'
            )

    def refresh_track_class(self, track: SemanticTrack) -> None:
        best_class = track.class_name
        best_score = float('-inf')
        best_count = -1

        for class_name, score in track.class_scores.items():
            count = track.class_counts.get(class_name, 0)
            if score > best_score or (score == best_score and count > best_count):
                best_class = class_name
                best_score = score
                best_count = count

        track.class_name = best_class

    def consolidate_tracks(self) -> None:
        merged_any = True

        while merged_any:
            merged_any = False

            for i, primary_candidate in enumerate(self.tracks):
                for j in range(i + 1, len(self.tracks)):
                    secondary_candidate = self.tracks[j]

                    if primary_candidate.class_name != secondary_candidate.class_name:
                        continue

                    merge_distance = (
                        self.confirmed_merge_distance
                        if primary_candidate.confirmed or secondary_candidate.confirmed else
                        self.merge_distance
                    )

                    distance = self.euclidean_distance(
                        primary_candidate.x,
                        primary_candidate.y,
                        primary_candidate.z,
                        secondary_candidate.x,
                        secondary_candidate.y,
                        secondary_candidate.z
                    )

                    if distance > merge_distance:
                        continue

                    primary_track, secondary_track = self.pick_primary_track(
                        primary_candidate,
                        secondary_candidate
                    )
                    self.absorb_track(primary_track, secondary_track)
                    self.tracks.remove(secondary_track)

                    self.get_logger().info(
                        f'Merged duplicate semantic object | kept={primary_track.track_id} '
                        f'removed={secondary_track.track_id} class={primary_track.class_name} '
                        f'distance={distance:.2f} obs={primary_track.observation_count}'
                    )

                    merged_any = True
                    break

                if merged_any:
                    break

    def pick_primary_track(
        self,
        track_a: SemanticTrack,
        track_b: SemanticTrack
    ) -> tuple[SemanticTrack, SemanticTrack]:
        if track_a.confirmed != track_b.confirmed:
            return (track_a, track_b) if track_a.confirmed else (track_b, track_a)

        if track_a.observation_count != track_b.observation_count:
            return (
                (track_a, track_b)
                if track_a.observation_count > track_b.observation_count else
                (track_b, track_a)
            )

        if track_a.confidence != track_b.confidence:
            return (
                (track_a, track_b)
                if track_a.confidence >= track_b.confidence else
                (track_b, track_a)
            )

        return (
            (track_a, track_b)
            if track_a.track_id <= track_b.track_id else
            (track_b, track_a)
        )

    def absorb_track(
        self,
        primary_track: SemanticTrack,
        secondary_track: SemanticTrack
    ) -> None:
        total_observations = primary_track.observation_count + secondary_track.observation_count

        primary_weight = primary_track.observation_count / total_observations
        secondary_weight = secondary_track.observation_count / total_observations

        primary_track.x = (
            primary_weight * primary_track.x +
            secondary_weight * secondary_track.x
        )
        primary_track.y = (
            primary_weight * primary_track.y +
            secondary_weight * secondary_track.y
        )
        primary_track.z = (
            primary_weight * primary_track.z +
            secondary_weight * secondary_track.z
        )
        primary_track.confidence = max(primary_track.confidence, secondary_track.confidence)
        for class_name, score in secondary_track.class_scores.items():
            primary_track.class_scores[class_name] = (
                primary_track.class_scores.get(class_name, 0.0) + score
            )
        for class_name, count in secondary_track.class_counts.items():
            primary_track.class_counts[class_name] = (
                primary_track.class_counts.get(class_name, 0) + count
            )
        primary_track.first_seen = (
            primary_track.first_seen
            if primary_track.first_seen <= secondary_track.first_seen else
            secondary_track.first_seen
        )
        primary_track.last_seen = (
            primary_track.last_seen
            if primary_track.last_seen >= secondary_track.last_seen else
            secondary_track.last_seen
        )
        primary_track.observation_count = total_observations
        primary_track.confirmed = (
            primary_track.confirmed or
            secondary_track.confirmed or
            total_observations >= self.min_observations
        )
        self.refresh_track_class(primary_track)

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

    def prune_invalid_tracks(self) -> None:
        if self.latest_map is None:
            return

        kept_tracks: List[SemanticTrack] = []

        for track in self.tracks:
            if self.is_map_consistent(track.x, track.y):
                kept_tracks.append(track)
                continue

            self.get_logger().info(
                f'Removed invalid semantic object | id={track.track_id} '
                f'class={track.class_name} pos=({track.x:.2f}, {track.y:.2f}, {track.z:.2f})'
            )

        self.tracks = kept_tracks

    def limit_confirmed_tracks_per_class(self) -> None:
        if self.max_confirmed_tracks_per_class <= 0:
            return

        grouped_tracks: Dict[str, List[SemanticTrack]] = {}
        unconfirmed_tracks: List[SemanticTrack] = []

        for track in self.tracks:
            if not track.confirmed:
                unconfirmed_tracks.append(track)
                continue

            grouped_tracks.setdefault(track.class_name, []).append(track)

        kept_tracks: List[SemanticTrack] = list(unconfirmed_tracks)

        for class_name, class_tracks in grouped_tracks.items():
            sorted_tracks = sorted(
                class_tracks,
                key=self.track_priority,
                reverse=True
            )
            kept_tracks.extend(sorted_tracks[:self.max_confirmed_tracks_per_class])

            for removed_track in sorted_tracks[self.max_confirmed_tracks_per_class:]:
                self.get_logger().info(
                    f'Removed weaker duplicate class track | id={removed_track.track_id} '
                    f'class={class_name} obs={removed_track.observation_count} '
                    f'conf={removed_track.confidence:.2f}'
                )

        kept_track_ids = {track.track_id for track in kept_tracks}
        self.tracks = [track for track in self.tracks if track.track_id in kept_track_ids]

    def is_map_consistent(self, x: float, y: float) -> bool:
        if self.latest_map is None:
            return True

        map_cell = self.world_to_map(self.latest_map, x, y)
        if map_cell is None:
            return False

        mx, my = map_cell
        width = self.latest_map.info.width
        height = self.latest_map.info.height

        if (
            mx < self.map_boundary_margin_cells or
            my < self.map_boundary_margin_cells or
            mx >= width - self.map_boundary_margin_cells or
            my >= height - self.map_boundary_margin_cells
        ):
            return False

        value = self.latest_map.data[self.to_index(mx, my, width)]
        if value < 0:
            return False

        supporting_cells = self.count_supporting_occupied_cells(mx, my)
        return supporting_cells >= self.min_supporting_occupied_cells

    def count_supporting_occupied_cells(self, mx: int, my: int) -> int:
        if self.latest_map is None:
            return 0

        width = self.latest_map.info.width
        height = self.latest_map.info.height
        data = self.latest_map.data

        count = 0

        for ny in range(my - self.support_radius_cells, my + self.support_radius_cells + 1):
            for nx in range(mx - self.support_radius_cells, mx + self.support_radius_cells + 1):
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue

                if data[self.to_index(nx, ny, width)] >= self.occupied_threshold:
                    count += 1

        return count

    def world_to_map(
        self,
        occ_grid: OccupancyGrid,
        x: float,
        y: float
    ) -> Optional[tuple[int, int]]:
        resolution = occ_grid.info.resolution
        origin = occ_grid.info.origin

        mx = int((x - origin.position.x) / resolution)
        my = int((y - origin.position.y) / resolution)

        if mx < 0 or my < 0 or mx >= occ_grid.info.width or my >= occ_grid.info.height:
            return None

        return mx, my

    @staticmethod
    def to_index(x: int, y: int, width: int) -> int:
        return y * width + x

    def track_priority(self, track: SemanticTrack) -> tuple[float, int, int]:
        class_score = track.class_scores.get(track.class_name, 0.0)
        last_seen_ns = track.last_seen.nanoseconds
        return (
            class_score,
            track.observation_count,
            last_seen_ns
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
