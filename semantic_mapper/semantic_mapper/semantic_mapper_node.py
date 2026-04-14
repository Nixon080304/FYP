#!/usr/bin/env python3

from collections import deque
from typing import List, Tuple, Set, Optional

import rclpy
from rclpy.node import Node

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point, TransformStamped, PointStamped
from visualization_msgs.msg import Marker
from tf2_ros import Buffer, TransformListener, TransformException


GridCell = Tuple[int, int]
Cluster = List[GridCell]


class SemanticMapperNode(Node):

    def __init__(self) -> None:
        super().__init__('semantic_mapper')

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        self.frontier_pub = self.create_publisher(
            Marker,
            '/semantic_frontiers',
            10
        )

        self.label_pub = self.create_publisher(
            Marker,
            '/semantic_labels',
            10
        )

        self.best_pub = self.create_publisher(
            Marker,
            '/semantic_best_frontier',
            10
        )
        self.best_point_pub = self.create_publisher(
            PointStamped,
            '/semantic_best_frontier_point',
            10
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.occupied_threshold = 50
        self.min_cluster_size = 5

        self.unknown_radius = 5
        self.free_radius = 5
        self.obstacle_radius = 4
        self.boundary_margin = 6

        self.get_logger().info('Semantic mapper node started.')

    def map_callback(self, msg: OccupancyGrid) -> None:
        frontier_cells = self.detect_frontiers(msg)
        clusters = self.cluster_frontiers(frontier_cells)
        clusters = [c for c in clusters if len(c) >= self.min_cluster_size]

        if not clusters:
            self.publish_frontier_markers(msg, [])
            self.publish_label_markers(msg, [])
            self.publish_best_marker(msg, None)
            return

        scored_frontiers = []

        for cluster in clusters:
            centroid = self.compute_centroid(cluster)
            cx = int(round(centroid[0]))
            cy = int(round(centroid[1]))

            unknown_gain = self.count_unknown_cells(msg, cx, cy, self.unknown_radius)
            free_openness = self.count_free_cells(msg, cx, cy, self.free_radius)
            obstacle_density = self.count_occupied_cells(msg, cx, cy, self.obstacle_radius)
            boundary_penalty = 1.0 if self.is_near_boundary(msg, cx, cy) else 0.0

            semantic_score = (
                1.2 * unknown_gain +
                0.8 * free_openness -
                1.0 * obstacle_density -
                15.0 * boundary_penalty
            )

            label = self.assign_label(
                unknown_gain,
                free_openness,
                obstacle_density,
                boundary_penalty
            )

            scored_frontiers.append({
                'cluster': cluster,
                'centroid': centroid,
                'label': label,
                'score': semantic_score,
                'unknown_gain': unknown_gain,
                'free_openness': free_openness,
                'obstacle_density': obstacle_density,
                'boundary_penalty': boundary_penalty
            })

        scored_frontiers.sort(key=lambda item: item['score'], reverse=True)

        self.publish_frontier_markers(msg, scored_frontiers)
        self.publish_label_markers(msg, scored_frontiers)
        self.publish_best_marker(msg, scored_frontiers[0]['centroid'])
        self.publish_best_frontier_point(msg, scored_frontiers[0]['centroid'])

        best = scored_frontiers[0]
        self.get_logger().info(
            f"Best semantic frontier | "
            f"label={best['label']} | "
            f"score={best['score']:.2f} | "
            f"unknown={best['unknown_gain']} | "
            f"free={best['free_openness']} | "
            f"occupied={best['obstacle_density']} | "
            f"centroid=({best['centroid'][0]:.1f}, {best['centroid'][1]:.1f})",
            throttle_duration_sec=2.0
        )

    def detect_frontiers(self, occ_grid: OccupancyGrid) -> List[GridCell]:
        width = occ_grid.info.width
        height = occ_grid.info.height
        data = list(occ_grid.data)

        frontiers: List[GridCell] = []

        for y in range(height):
            for x in range(width):
                idx = self.to_index(x, y, width)

                if data[idx] < 0 or data[idx] >= self.occupied_threshold:
                    continue

                if self.has_unknown_neighbor(x, y, width, height, data):
                    frontiers.append((x, y))

        return frontiers

    def has_unknown_neighbor(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        data: List[int]
    ) -> bool:
        for ny in range(y - 1, y + 2):
            for nx in range(x - 1, x + 2):
                if nx == x and ny == y:
                    continue
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue
                if data[self.to_index(nx, ny, width)] == -1:
                    return True
        return False

    def cluster_frontiers(self, frontier_cells: List[GridCell]) -> List[Cluster]:
        frontier_set: Set[GridCell] = set(frontier_cells)
        visited: Set[GridCell] = set()
        clusters: List[Cluster] = []

        for cell in frontier_cells:
            if cell in visited:
                continue

            cluster: Cluster = []
            queue = deque([cell])
            visited.add(cell)

            while queue:
                current = queue.popleft()
                cluster.append(current)

                cx, cy = current
                for ny in range(cy - 1, cy + 2):
                    for nx in range(cx - 1, cx + 2):
                        neighbor = (nx, ny)
                        if neighbor == current:
                            continue
                        if neighbor in frontier_set and neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)

            clusters.append(cluster)

        return clusters

    def compute_centroid(self, cluster: Cluster) -> Tuple[float, float]:
        mean_x = sum(cell[0] for cell in cluster) / len(cluster)
        mean_y = sum(cell[1] for cell in cluster) / len(cluster)
        return mean_x, mean_y

    def count_unknown_cells(
        self,
        occ_grid: OccupancyGrid,
        x: int,
        y: int,
        radius: int
    ) -> int:
        return self.count_cells_with_value(occ_grid, x, y, radius, target='unknown')

    def count_free_cells(
        self,
        occ_grid: OccupancyGrid,
        x: int,
        y: int,
        radius: int
    ) -> int:
        return self.count_cells_with_value(occ_grid, x, y, radius, target='free')

    def count_occupied_cells(
        self,
        occ_grid: OccupancyGrid,
        x: int,
        y: int,
        radius: int
    ) -> int:
        return self.count_cells_with_value(occ_grid, x, y, radius, target='occupied')

    def count_cells_with_value(
        self,
        occ_grid: OccupancyGrid,
        x: int,
        y: int,
        radius: int,
        target: str
    ) -> int:
        width = occ_grid.info.width
        height = occ_grid.info.height
        data = list(occ_grid.data)

        count = 0

        for ny in range(y - radius, y + radius + 1):
            for nx in range(x - radius, x + radius + 1):
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue

                value = data[self.to_index(nx, ny, width)]

                if target == 'unknown' and value == -1:
                    count += 1
                elif target == 'free' and 0 <= value < self.occupied_threshold:
                    count += 1
                elif target == 'occupied' and value >= self.occupied_threshold:
                    count += 1

        return count

    def is_near_boundary(
        self,
        occ_grid: OccupancyGrid,
        x: int,
        y: int
    ) -> bool:
        width = occ_grid.info.width
        height = occ_grid.info.height

        return (
            x < self.boundary_margin or
            y < self.boundary_margin or
            x >= width - self.boundary_margin or
            y >= height - self.boundary_margin
        )

    def assign_label(
        self,
        unknown_gain: int,
        free_openness: int,
        obstacle_density: int,
        boundary_penalty: float
    ) -> str:
        if boundary_penalty > 0.0:
            return 'boundary_frontier'

        if unknown_gain > 40 and free_openness > 45 and obstacle_density < 10:
            return 'open_area_entry'

        if unknown_gain > 20 and obstacle_density > 20:
            return 'narrow_passage'

        if obstacle_density > 30:
            return 'obstacle_dense'

        if unknown_gain < 10:
            return 'low_value_frontier'

        return 'mixed_frontier'

    def to_index(self, x: int, y: int, width: int) -> int:
        return y * width + x

    def map_to_world(
        self,
        occ_grid: OccupancyGrid,
        mx: float,
        my: float
    ) -> Tuple[float, float]:
        resolution = occ_grid.info.resolution
        origin = occ_grid.info.origin

        wx = origin.position.x + (mx + 0.5) * resolution
        wy = origin.position.y + (my + 0.5) * resolution

        return wx, wy

    def publish_frontier_markers(self, occ_grid: OccupancyGrid, scored_frontiers: List[dict]) -> None:
        marker = Marker()
        marker.header.frame_id = occ_grid.header.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'semantic_frontiers'
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = 0.12
        marker.scale.y = 0.12
        marker.color.r = 1.0
        marker.color.g = 0.5
        marker.color.b = 0.0
        marker.color.a = 1.0

        for item in scored_frontiers:
            mx, my = item['centroid']
            wx, wy = self.map_to_world(occ_grid, mx, my)

            p = Point()
            p.x = wx
            p.y = wy
            p.z = 0.0
            marker.points.append(p)

        self.frontier_pub.publish(marker)

    def publish_label_markers(self, occ_grid: OccupancyGrid, scored_frontiers: List[dict]) -> None:
        marker_id = 0

        for item in scored_frontiers:
            marker = Marker()
            marker.header.frame_id = occ_grid.header.frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'semantic_labels'
            marker.id = marker_id
            marker.type = Marker.TEXT_VIEW_FACING
            marker.action = Marker.ADD
            marker.scale.z = 0.18
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            marker.color.a = 1.0

            mx, my = item['centroid']
            wx, wy = self.map_to_world(occ_grid, mx, my)

            marker.pose.position.x = wx
            marker.pose.position.y = wy
            marker.pose.position.z = 0.2
            marker.pose.orientation.w = 1.0
            marker.text = f"{item['label']} ({item['score']:.1f})"

            self.label_pub.publish(marker)
            marker_id += 1

    def publish_best_marker(
        self,
        occ_grid: OccupancyGrid,
        best_centroid: Optional[Tuple[float, float]]
    ) -> None:
        marker = Marker()
        marker.header.frame_id = occ_grid.header.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'semantic_best_frontier'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = 0.25
        marker.scale.y = 0.25
        marker.scale.z = 0.25
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        if best_centroid is None:
            marker.action = Marker.DELETE
        else:
            wx, wy = self.map_to_world(occ_grid, best_centroid[0], best_centroid[1])
            marker.pose.position.x = wx
            marker.pose.position.y = wy
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0

        self.best_pub.publish(marker)

    def publish_best_frontier_point(
        self,
        occ_grid: OccupancyGrid,
        best_centroid: Optional[Tuple[float, float]]
    ) -> None:
        if best_centroid is None:
            return

        msg = PointStamped()
        msg.header.frame_id = occ_grid.header.frame_id
        msg.header.stamp = self.get_clock().now().to_msg()

        wx, wy = self.map_to_world(occ_grid, best_centroid[0], best_centroid[1])
        msg.point.x = wx
        msg.point.y = wy
        msg.point.z = 0.0

        self.best_point_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SemanticMapperNode()

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