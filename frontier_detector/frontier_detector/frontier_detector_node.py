#!/usr/bin/env python3

from typing import List, Sequence, Tuple

import rclpy
from rclpy.node import Node

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker


class FrontierDetectorNode(Node):

    def __init__(self) -> None:
        super().__init__('frontier_detector')

        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('frontier_topic', '/frontier_markers')
        self.declare_parameter('occupied_threshold', 50)
        self.declare_parameter('min_free_neighbors', 1)

        self.map_topic = str(self.get_parameter('map_topic').value)
        self.frontier_topic = str(self.get_parameter('frontier_topic').value)
        self.occupied_threshold = int(self.get_parameter('occupied_threshold').value)
        self.min_free_neighbors = int(self.get_parameter('min_free_neighbors').value)

        # Subscribe to map
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_callback,
            10
        )

        # Publish frontier visualization
        self.frontier_pub = self.create_publisher(
            Marker,
            self.frontier_topic,
            10
        )

        self.get_logger().info(
            'Frontier detector node started. '
            f'map_topic={self.map_topic}, frontier_topic={self.frontier_topic}, '
            f'occupied_threshold={self.occupied_threshold}, '
            f'min_free_neighbors={self.min_free_neighbors}'
        )

    def map_callback(self, msg: OccupancyGrid) -> None:

        unknown_count, free_count, occupied_count = self.compute_map_stats(msg.data)
        frontier_cells = self.detect_frontiers(msg)

        self.get_logger().info(
            f'Map stats | unknown={unknown_count} free={free_count} '
            f'occupied={occupied_count} frontier_cells={len(frontier_cells)}',
            throttle_duration_sec=2.0
        )

        self.publish_frontier_markers(msg, frontier_cells)

    def detect_frontiers(self, occ_grid: OccupancyGrid) -> List[Tuple[int, int]]:

        width = occ_grid.info.width
        height = occ_grid.info.height
        data = list(occ_grid.data)

        frontiers: List[Tuple[int, int]] = []

        for y in range(height):
            for x in range(width):

                idx = self.to_index(x, y, width)

                if data[idx] < 0 or data[idx] >= self.occupied_threshold:
                    continue

                if not self.has_unknown_neighbor(x, y, width, height, data):
                    continue

                if self.count_free_neighbors(x, y, width, height, data) < self.min_free_neighbors:
                    continue

                if self.is_boundary_cell(x, y, width, height):
                    continue

                frontiers.append((x, y))

        return frontiers

    def has_unknown_neighbor(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        data: Sequence[int]
    ) -> bool:

        for ny in range(y - 1, y + 2):
            for nx in range(x - 1, x + 2):

                if nx == x and ny == y:
                    continue

                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue

                n_idx = self.to_index(nx, ny, width)

                if data[n_idx] == -1:
                    return True

        return False

    def count_free_neighbors(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        data: Sequence[int]
    ) -> int:

        free_neighbors = 0

        for ny in range(y - 1, y + 2):
            for nx in range(x - 1, x + 2):

                if nx == x and ny == y:
                    continue

                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue

                n_idx = self.to_index(nx, ny, width)
                value = data[n_idx]

                if 0 <= value < self.occupied_threshold:
                    free_neighbors += 1

        return free_neighbors

    def is_boundary_cell(self, x: int, y: int, width: int, height: int) -> bool:
        return x == 0 or y == 0 or x == width - 1 or y == height - 1

    def compute_map_stats(self, data: Sequence[int]) -> Tuple[int, int, int]:
        unknown_count = sum(1 for value in data if value == -1)
        free_count = sum(1 for value in data if 0 <= value < self.occupied_threshold)
        occupied_count = sum(1 for value in data if value >= self.occupied_threshold)
        return unknown_count, free_count, occupied_count

    def to_index(self, x: int, y: int, width: int) -> int:
        return y * width + x

    def publish_frontier_markers(
        self,
        occ_grid: OccupancyGrid,
        frontier_cells: List[Tuple[int, int]]
    ) -> None:

        marker = Marker()

        marker.header.frame_id = occ_grid.header.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = "frontiers"
        marker.id = 0

        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.05
        marker.scale.y = 0.05

        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        resolution = occ_grid.info.resolution
        origin = occ_grid.info.origin

        for mx, my in frontier_cells:

            wx = origin.position.x + (mx + 0.5) * resolution
            wy = origin.position.y + (my + 0.5) * resolution

            p = Point()
            p.x = wx
            p.y = wy
            p.z = 0.0

            marker.points.append(p)

        self.frontier_pub.publish(marker)


def main(args=None) -> None:

    rclpy.init(args=args)

    node = FrontierDetectorNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
