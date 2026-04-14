#!/usr/bin/env python3

from collections import deque
from math import hypot
from typing import List, Tuple, Set, Optional

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from visualization_msgs.msg import Marker
from nav2_msgs.action import NavigateToPose
from tf2_ros import Buffer, TransformListener
from tf2_ros import TransformException
from std_msgs.msg import Bool, String


GridCell = Tuple[int, int]
Cluster = List[GridCell]


class ExplorationPlannerNode(Node):

    def __init__(self) -> None:
        super().__init__('exploration_planner')

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        self.centroid_pub = self.create_publisher(
            Marker,
            '/frontier_centroids',
            10
        )

        self.selected_pub = self.create_publisher(
            Marker,
            '/selected_frontier',
            10
        )

        self.reachable_pub = self.create_publisher(
            Marker,
            '/debug_reachable_cells',
            10
        )

        self.leftover_frontier_pub = self.create_publisher(
            Marker,
            '/debug_leftover_frontiers',
            10
        )

        self.rejected_pub = self.create_publisher(
            Marker,
            '/debug_rejected_goals',
            10
        )

        self.recovery_pub = self.create_publisher(
            Marker,
            '/debug_recovery_goal',
            10
        )

        self.exploration_finished_pub = self.create_publisher(
            Bool,
            '/exploration_finished',
            10
        )

        self.exploration_status_pub = self.create_publisher(
            String,
            '/exploration_status',
            10
        )

        self.nav_to_pose_client = ActionClient(
            self,
            NavigateToPose,
            'navigate_to_pose'
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.occupied_threshold = 50

        self.base_min_cluster_size = 5
        self.min_cluster_size = self.base_min_cluster_size
        self.late_stage_min_cluster_size = 6

        # Safer frontier goal settings
        self.clearance_radius_cells = 6
        self.max_allowed_occupied_cells_near_goal = 1

        self.info_gain_radius_cells = 4

        self.goal_in_progress = False
        self.current_goal: Optional[Tuple[float, float]] = None
        self.goal_start_time = None
        self.max_goal_duration_sec = 60.0

        self.last_sent_goal: Optional[Tuple[float, float]] = None
        self.min_goal_separation = 0.75

        self.failed_goals: List[Tuple[float, float]] = []
        self.failed_goal_radius = 0.60
        self.max_failed_goals = 50

        self.no_frontier_counter = 0
        self.no_frontier_limit = 30

        self.exploration_complete = False
        self.completion_reason: Optional[str] = None

        self.coverage_target = 95.0
        self.coverage_history: List[Tuple[float, float]] = []
        self.stagnation_window_sec = 45.0
        self.stagnation_min_gain = 0.1

        self.recovery_enabled = True
        self.recovery_coverage_threshold = 90.0
        self.recovery_no_goal_trigger = 2
        self.recovery_search_radius_cells = 12
        self.recovery_min_goal_distance = 0.8
        self.recovery_max_goal_distance = 3.0
        self.recovery_failure_count = 0
        self.recovery_failure_trigger = 2

        self.goal_repeat_count = 0
        self.goal_repeat_trigger = 4
        self.previous_selected_goal: Optional[Tuple[float, float]] = None

        self.min_progress_distance = 8
        self.goal_start_robot_pose = None

        self.debug_rejected_cells: List[GridCell] = []

        # New cooldown after failures/aborts
        self.replan_cooldown_sec = 5.0
        self.last_failure_time = None

        self.get_logger().info('Exploration planner node started.')
        self.publish_exploration_state()

    def publish_exploration_state(self) -> None:
        finished_msg = Bool()
        finished_msg.data = self.exploration_complete
        self.exploration_finished_pub.publish(finished_msg)

        status_msg = String()
        if self.exploration_complete:
            reason = self.completion_reason if self.completion_reason is not None else 'UNKNOWN'
            status_msg.data = f'FINISHED:{reason}'
        else:
            status_msg.data = 'EXPLORING'
        self.exploration_status_pub.publish(status_msg)

    def is_in_replan_cooldown(self) -> bool:
        if self.last_failure_time is None:
            return False

        elapsed = (self.get_clock().now() - self.last_failure_time).nanoseconds / 1e9
        return elapsed < self.replan_cooldown_sec

    def map_callback(self, msg: OccupancyGrid) -> None:
        if self.exploration_complete:
            self.publish_exploration_state()
            return

        self.handle_goal_timeout()

        if self.is_in_replan_cooldown():
            self.publish_exploration_state()
            self.get_logger().info(
                'Planner in cooldown after failed/aborted goal. Waiting before replanning.',
                throttle_duration_sec=2.0
            )
            return

        # Prevent replanning too soon after a goal was just completed/sent
        if self.goal_start_robot_pose is not None and not self.goal_in_progress:
            robot_map = self.get_robot_map_position(msg)
            if robot_map is not None:
                dist_from_last_goal_start = hypot(
                    robot_map[0] - self.goal_start_robot_pose[0],
                    robot_map[1] - self.goal_start_robot_pose[1]
                )
                if dist_from_last_goal_start < self.min_progress_distance:
                    self.publish_exploration_state()
                    return

        unknown, free, occupied, raw_coverage = self.compute_map_stats(msg)
        effective_coverage, interior_unknown, interior_free, interior_occupied = self.compute_effective_coverage(msg)

        self.update_dynamic_parameters(effective_coverage)

        frontier_cells = self.detect_frontiers(msg)
        clusters = self.cluster_frontiers(frontier_cells)
        clusters = [c for c in clusters if len(c) >= self.min_cluster_size]

        reachable_cells = self.compute_reachable_cells(msg)

        centroids = [self.compute_centroid(cluster) for cluster in clusters]
        self.publish_centroid_markers(msg, centroids)

        self.publish_point_marker(
            msg,
            list(reachable_cells),
            self.reachable_pub,
            'reachable_cells',
            0.0, 0.7, 1.0,
            scale=0.03
        )

        self.publish_point_marker(
            msg,
            frontier_cells,
            self.leftover_frontier_pub,
            'leftover_frontiers',
            1.0, 0.0, 1.0,
            scale=0.06
        )

        selected_goal_map = self.select_best_goal(msg, clusters, reachable_cells)

        self.publish_point_marker(
            msg,
            self.debug_rejected_cells,
            self.rejected_pub,
            'rejected_goals',
            1.0, 0.0, 0.0,
            scale=0.08
        )

        if selected_goal_map is not None and self.previous_selected_goal is not None:
            repeat_dist = hypot(
                selected_goal_map[0] - self.previous_selected_goal[0],
                selected_goal_map[1] - self.previous_selected_goal[1]
            )
            if repeat_dist < 3.0:
                self.goal_repeat_count += 1
            else:
                self.goal_repeat_count = 0
        else:
            self.goal_repeat_count = 0

        self.previous_selected_goal = selected_goal_map

        should_try_recovery = (
            self.recovery_enabled and
            effective_coverage < self.recovery_coverage_threshold and
            not self.goal_in_progress and
            (
                (selected_goal_map is None and self.no_frontier_counter >= self.recovery_no_goal_trigger) or
                (self.recovery_failure_count >= self.recovery_failure_trigger) or
                (self.goal_repeat_count >= self.goal_repeat_trigger)
            )
        )

        recovery_goal_map = None

        if should_try_recovery:
            recovery_goal_map = self.select_open_space_goal(msg, reachable_cells)
            self.publish_recovery_marker(msg, recovery_goal_map)

            if recovery_goal_map is not None:
                self.get_logger().info(
                    f'Recovery triggered | '
                    f'failure_count={self.recovery_failure_count} | '
                    f'repeat_count={self.goal_repeat_count} | '
                    f'recovery_goal={recovery_goal_map}'
                )
                selected_goal_map = recovery_goal_map
                self.recovery_failure_count = 0
                self.goal_repeat_count = 0
        else:
            self.publish_recovery_marker(msg, None)

        self.get_logger().info(
            f'Raw coverage: {raw_coverage:.2f}% | '
            f'Effective coverage: {effective_coverage:.2f}% | '
            f'Unknown(raw): {unknown} | Interior Unknown: {interior_unknown} | '
            f'Free(raw): {free} | Occupied(raw): {occupied} | '
            f'Frontiers: {len(frontier_cells)} | Clusters: {len(clusters)} | '
            f'Selected: {selected_goal_map} | Goal in progress: {self.goal_in_progress} | '
            f'Recovery failures: {self.recovery_failure_count} | '
            f'Repeat count: {self.goal_repeat_count}'
        )

        self.publish_selected_marker(msg, selected_goal_map)
        self.update_completion_state(
            selected_goal_map,
            effective_coverage,
            len(frontier_cells),
            interior_free + interior_occupied
        )
        self.publish_exploration_state()

        if self.exploration_complete:
            return

        if selected_goal_map is None:
            return

        if self.goal_in_progress:
            return

        goal_wx, goal_wy = self.map_to_world(
            msg,
            selected_goal_map[0],
            selected_goal_map[1]
        )

        if self.is_near_failed_goal(goal_wx, goal_wy):
            return

        if self.last_sent_goal is not None:
            dist_to_last = hypot(
                goal_wx - self.last_sent_goal[0],
                goal_wy - self.last_sent_goal[1]
            )
            if dist_to_last < self.min_goal_separation:
                return

        robot_map = self.get_robot_map_position(msg)
        if robot_map is not None:
            self.goal_start_robot_pose = robot_map

        self.send_goal(goal_wx, goal_wy, msg.header.frame_id)

    def update_dynamic_parameters(self, coverage: float) -> None:
        if coverage < 40.0:
            self.min_cluster_size = self.base_min_cluster_size
        elif coverage < 60.0:
            self.min_cluster_size = 7
        else:
            self.min_cluster_size = self.late_stage_min_cluster_size

    def compute_map_stats(self, occ_grid: OccupancyGrid) -> Tuple[int, int, int, float]:
        data = list(occ_grid.data)

        unknown = sum(1 for v in data if v == -1)
        free = sum(1 for v in data if 0 <= v < self.occupied_threshold)
        occupied = sum(1 for v in data if v >= self.occupied_threshold)

        total = len(data)
        explored = free + occupied
        coverage = (explored / total * 100.0) if total > 0 else 0.0

        return unknown, free, occupied, coverage

    def compute_exterior_mask(self, occ_grid: OccupancyGrid) -> Set[GridCell]:
        width = occ_grid.info.width
        height = occ_grid.info.height
        data = list(occ_grid.data)

        exterior: Set[GridCell] = set()
        queue = deque()

        def try_add(x: int, y: int) -> None:
            if x < 0 or y < 0 or x >= width or y >= height:
                return
            if (x, y) in exterior:
                return

            idx = self.to_index(x, y, width)

            if data[idx] >= self.occupied_threshold:
                return

            exterior.add((x, y))
            queue.append((x, y))

        for x in range(width):
            try_add(x, 0)
            try_add(x, height - 1)

        for y in range(height):
            try_add(0, y)
            try_add(width - 1, y)

        while queue:
            cx, cy = queue.popleft()

            for nx, ny in [
                (cx + 1, cy),
                (cx - 1, cy),
                (cx, cy + 1),
                (cx, cy - 1)
            ]:
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue
                if (nx, ny) in exterior:
                    continue

                idx = self.to_index(nx, ny, width)

                if data[idx] >= self.occupied_threshold:
                    continue

                exterior.add((nx, ny))
                queue.append((nx, ny))

        return exterior

    def compute_effective_coverage(
        self,
        occ_grid: OccupancyGrid
    ) -> Tuple[float, int, int, int]:
        width = occ_grid.info.width
        height = occ_grid.info.height
        data = list(occ_grid.data)

        exterior = self.compute_exterior_mask(occ_grid)

        considered_total = 0
        explored = 0
        unknown_inside = 0
        free_inside = 0
        occupied_inside = 0

        for y in range(height):
            for x in range(width):
                idx = self.to_index(x, y, width)
                value = data[idx]

                if (x, y) in exterior:
                    continue

                considered_total += 1

                if value == -1:
                    unknown_inside += 1
                elif value >= self.occupied_threshold:
                    occupied_inside += 1
                    explored += 1
                else:
                    free_inside += 1
                    explored += 1

        total_cells = width * height
        if considered_total < 0.1 * total_cells:
            unknown, free, occupied, raw_coverage = self.compute_map_stats(occ_grid)
            return raw_coverage, unknown, free, occupied

        effective_coverage = (
            explored / considered_total * 100.0 if considered_total > 0 else 0.0
        )

        return effective_coverage, unknown_inside, free_inside, occupied_inside

    def update_completion_state(
        self,
        selected_goal_map: Optional[Tuple[float, float]],
        coverage: float,
        frontier_count: int,
        explored_cells: int
    ) -> None:
        now_sec = self.get_clock().now().nanoseconds / 1e9
        self.coverage_history.append((now_sec, coverage))

        self.coverage_history = [
            item for item in self.coverage_history
            if now_sec - item[0] <= self.stagnation_window_sec
        ]

        if selected_goal_map is None and not self.goal_in_progress:
            self.no_frontier_counter += 1
        else:
            self.no_frontier_counter = 0

        if explored_cells < 500:
            return

        if not self.exploration_complete and coverage >= self.coverage_target:
            self.exploration_complete = True
            self.completion_reason = 'TARGET_COVERAGE_REACHED'
            self.get_logger().info(
                f'Exploration completed: {self.completion_reason}. '
                f'Final effective coverage: {coverage:.2f}%'
            )
            return

        coverage_gain = 0.0
        if len(self.coverage_history) >= 2:
            coverage_gain = self.coverage_history[-1][1] - self.coverage_history[0][1]

        if (
            not self.exploration_complete and
            not self.goal_in_progress and
            self.no_frontier_counter >= self.no_frontier_limit and
            frontier_count < 8 and
            self.recovery_failure_count >= self.recovery_failure_trigger
        ):
            self.exploration_complete = True
            self.completion_reason = 'NO_VALID_FRONTIERS'
            self.get_logger().info(
                f'Exploration completed: {self.completion_reason}. '
                f'Coverage gain over last {self.stagnation_window_sec:.1f}s: '
                f'{coverage_gain:.2f}% | Final effective coverage: {coverage:.2f}%'
            )
            return

        if (
            not self.exploration_complete and
            len(self.coverage_history) >= 2 and
            not self.goal_in_progress and
            frontier_count < 8 and
            self.no_frontier_counter >= 5 and
            self.recovery_failure_count >= self.recovery_failure_trigger and
            0.0 <= coverage_gain < self.stagnation_min_gain
        ):
            self.exploration_complete = True
            self.completion_reason = 'COVERAGE_STAGNATION'
            self.get_logger().info(
                f'Exploration completed: {self.completion_reason}. '
                f'Coverage gain over last {self.stagnation_window_sec:.1f}s: '
                f'{coverage_gain:.2f}% | Final effective coverage: {coverage:.2f}%'
            )

    def handle_goal_timeout(self) -> None:
        if not self.goal_in_progress or self.goal_start_time is None:
            return

        elapsed = (self.get_clock().now() - self.goal_start_time).nanoseconds / 1e9
        if elapsed > self.max_goal_duration_sec:
            self.get_logger().warn('Goal timeout reached. Releasing current goal for replanning.')

            if self.current_goal is not None:
                self.add_failed_goal(self.current_goal)

            self.recovery_failure_count += 1
            self.last_failure_time = self.get_clock().now()

            self.goal_in_progress = False
            self.current_goal = None
            self.goal_start_time = None
            self.last_sent_goal = None

    def add_failed_goal(self, goal: Tuple[float, float]) -> None:
        self.failed_goals.append(goal)
        if len(self.failed_goals) > self.max_failed_goals:
            self.failed_goals.pop(0)

    def is_near_failed_goal(self, wx: float, wy: float) -> bool:
        for fx, fy in self.failed_goals:
            if hypot(wx - fx, wy - fy) < self.failed_goal_radius:
                return True
        return False

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

                n_idx = self.to_index(nx, ny, width)
                if data[n_idx] == -1:
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

    def select_best_goal(
        self,
        occ_grid: OccupancyGrid,
        clusters: List[Cluster],
        reachable_cells: Set[GridCell]
    ) -> Optional[Tuple[float, float]]:
        self.debug_rejected_cells = []

        if not clusters:
            return None

        robot_map = self.get_robot_map_position(occ_grid)
        if robot_map is None:
            self.get_logger().warn('Could not get robot pose from TF.')
            return None

        rx, ry = robot_map
        strict_candidates = []

        for cluster in clusters:
            if len(cluster) < self.min_cluster_size:
                centroid = self.compute_centroid(cluster)
                representative = self.closest_frontier_cell_to_centroid(cluster, centroid)
                self.debug_rejected_cells.append(representative)
                continue

            centroid = self.compute_centroid(cluster)
            representative = self.closest_frontier_cell_to_centroid(cluster, centroid)
            safe_goal = self.shift_goal_inward(occ_grid, representative)

            if safe_goal is None:
                self.debug_rejected_cells.append(representative)
                continue

            gx = int(round(safe_goal[0]))
            gy = int(round(safe_goal[1]))

            if not self.is_goal_reachable(reachable_cells, gx, gy):
                self.debug_rejected_cells.append((gx, gy))
                continue

            if not self.is_goal_safe(occ_grid, gx, gy):
                self.debug_rejected_cells.append((gx, gy))
                continue

            wx, wy = self.map_to_world(occ_grid, gx, gy)

            if self.is_near_failed_goal(wx, wy):
                self.debug_rejected_cells.append((gx, gy))
                continue

            distance = hypot(gx - rx, gy - ry)
            info_gain = self.compute_information_gain(occ_grid, gx, gy)
            clearance = self.compute_clearance_score(occ_grid, gx, gy)
            cluster_size = len(cluster)

            score = (
                1.5 * info_gain +
                1.0 * cluster_size +
                1.0 * clearance -
                1.2 * distance
            )

            strict_candidates.append((score, (float(gx), float(gy)), cluster))

        if strict_candidates:
            strict_candidates.sort(key=lambda item: item[0], reverse=True)
            return strict_candidates[0][1]

        relaxed_candidates = []

        for cluster in clusters:
            if len(cluster) < 3:
                centroid = self.compute_centroid(cluster)
                representative = self.closest_frontier_cell_to_centroid(cluster, centroid)
                self.debug_rejected_cells.append(representative)
                continue

            centroid = self.compute_centroid(cluster)
            representative = self.closest_frontier_cell_to_centroid(cluster, centroid)
            safe_goal = self.shift_goal_inward(occ_grid, representative)

            if safe_goal is None:
                self.debug_rejected_cells.append(representative)
                continue

            gx = int(round(safe_goal[0]))
            gy = int(round(safe_goal[1]))

            if not self.is_goal_reachable(reachable_cells, gx, gy):
                self.debug_rejected_cells.append((gx, gy))
                continue

            if not self.is_goal_safe_relaxed(occ_grid, gx, gy):
                self.debug_rejected_cells.append((gx, gy))
                continue

            wx, wy = self.map_to_world(occ_grid, gx, gy)
            if self.is_near_failed_goal(wx, wy):
                self.debug_rejected_cells.append((gx, gy))
                continue

            distance = hypot(gx - rx, gy - ry)
            info_gain = self.compute_information_gain(occ_grid, gx, gy)
            cluster_size = len(cluster)

            score = (
                1.2 * info_gain +
                0.8 * cluster_size -
                1.0 * distance
            )

            relaxed_candidates.append((score, (float(gx), float(gy)), cluster))

        if relaxed_candidates:
            relaxed_candidates.sort(key=lambda item: item[0], reverse=True)
            return relaxed_candidates[0][1]

        return None

    def select_open_space_goal(
        self,
        occ_grid: OccupancyGrid,
        reachable_cells: Set[GridCell]
    ) -> Optional[Tuple[float, float]]:
        robot_map = self.get_robot_map_position(occ_grid)
        if robot_map is None:
            return None

        rx, ry = robot_map

        width = occ_grid.info.width
        height = occ_grid.info.height
        data = list(occ_grid.data)
        resolution = occ_grid.info.resolution

        best_goal = None
        best_score = float('-inf')

        for dy in range(-self.recovery_search_radius_cells, self.recovery_search_radius_cells + 1):
            for dx in range(-self.recovery_search_radius_cells, self.recovery_search_radius_cells + 1):
                gx = int(round(rx + dx))
                gy = int(round(ry + dy))

                if gx < 0 or gy < 0 or gx >= width or gy >= height:
                    continue

                if (gx, gy) not in reachable_cells:
                    continue

                idx = self.to_index(gx, gy, width)

                if data[idx] < 0 or data[idx] >= self.occupied_threshold:
                    continue

                wx, wy = self.map_to_world(occ_grid, gx, gy)

                if self.is_near_failed_goal(wx, wy):
                    continue

                dist_m = hypot((gx - rx) * resolution, (gy - ry) * resolution)

                if dist_m < self.recovery_min_goal_distance:
                    continue

                if dist_m > self.recovery_max_goal_distance:
                    continue

                clearance = self.compute_clearance_score(occ_grid, gx, gy)
                unknown_gain = self.compute_information_gain(occ_grid, gx, gy)

                score = (
                    1.5 * clearance +
                    0.8 * unknown_gain -
                    1.0 * dist_m
                )

                if score > best_score:
                    best_score = score
                    best_goal = (float(gx), float(gy))

        if best_goal is not None:
            self.get_logger().info(
                f'Recovery mode selected open-space goal: {best_goal}'
            )

        return best_goal

    def closest_frontier_cell_to_centroid(
        self,
        cluster: Cluster,
        centroid: Tuple[float, float]
    ) -> GridCell:
        cx, cy = centroid
        return min(
            cluster,
            key=lambda cell: (cell[0] - cx) ** 2 + (cell[1] - cy) ** 2
        )

    def shift_goal_inward(
        self,
        occ_grid: OccupancyGrid,
        frontier_cell: GridCell
    ) -> Optional[Tuple[float, float]]:
        width = occ_grid.info.width
        height = occ_grid.info.height
        data = list(occ_grid.data)

        fx, fy = frontier_cell
        search_radius = 14

        best_cell = None
        best_score = float('inf')

        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                nx = fx + dx
                ny = fy + dy

                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue

                idx = self.to_index(nx, ny, width)

                if data[idx] < 0 or data[idx] >= self.occupied_threshold:
                    continue

                unknown_neighbors = self.count_unknown_neighbors(nx, ny, width, height, data)
                occupied_neighbors = self.count_occupied_neighbors(nx, ny, width, height, data)

                distance_penalty = hypot(dx, dy)
                score = distance_penalty + 8.0 * occupied_neighbors + 2.0 * unknown_neighbors

                if score < best_score:
                    best_score = score
                    best_cell = (float(nx), float(ny))

        return best_cell

    def compute_clearance_score(
        self,
        occ_grid: OccupancyGrid,
        x: int,
        y: int
    ) -> int:
        width = occ_grid.info.width
        height = occ_grid.info.height
        data = list(occ_grid.data)

        safe_count = 0

        for ny in range(y - self.clearance_radius_cells, y + self.clearance_radius_cells + 1):
            for nx in range(x - self.clearance_radius_cells, x + self.clearance_radius_cells + 1):
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue

                idx = self.to_index(nx, ny, width)
                if 0 <= data[idx] < self.occupied_threshold:
                    safe_count += 1

        return safe_count

    def is_goal_safe(
        self,
        occ_grid: OccupancyGrid,
        x: int,
        y: int
    ) -> bool:
        width = occ_grid.info.width
        height = occ_grid.info.height
        data = list(occ_grid.data)

        occupied_count = 0

        for ny in range(y - self.clearance_radius_cells, y + self.clearance_radius_cells + 1):
            for nx in range(x - self.clearance_radius_cells, x + self.clearance_radius_cells + 1):
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue

                idx = self.to_index(nx, ny, width)
                if data[idx] >= self.occupied_threshold:
                    occupied_count += 1

        return occupied_count <= self.max_allowed_occupied_cells_near_goal

    def is_goal_safe_relaxed(
        self,
        occ_grid: OccupancyGrid,
        x: int,
        y: int
    ) -> bool:
        width = occ_grid.info.width
        height = occ_grid.info.height
        data = list(occ_grid.data)

        occupied_count = 0

        for ny in range(y - self.clearance_radius_cells, y + self.clearance_radius_cells + 1):
            for nx in range(x - self.clearance_radius_cells, x + self.clearance_radius_cells + 1):
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue

                idx = self.to_index(nx, ny, width)
                if data[idx] >= self.occupied_threshold:
                    occupied_count += 1

        return occupied_count <= 3

    def compute_information_gain(
        self,
        occ_grid: OccupancyGrid,
        x: int,
        y: int
    ) -> int:
        width = occ_grid.info.width
        height = occ_grid.info.height
        data = list(occ_grid.data)

        unknown_count = 0

        for ny in range(y - self.info_gain_radius_cells, y + self.info_gain_radius_cells + 1):
            for nx in range(x - self.info_gain_radius_cells, x + self.info_gain_radius_cells + 1):
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue

                idx = self.to_index(nx, ny, width)
                if data[idx] == -1:
                    unknown_count += 1

        return unknown_count

    def count_unknown_neighbors(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        data: List[int]
    ) -> int:
        count = 0
        for ny in range(y - 1, y + 2):
            for nx in range(x - 1, x + 2):
                if nx == x and ny == y:
                    continue
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue
                if data[self.to_index(nx, ny, width)] == -1:
                    count += 1
        return count

    def count_occupied_neighbors(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        data: List[int]
    ) -> int:
        count = 0
        for ny in range(y - 1, y + 2):
            for nx in range(x - 1, x + 2):
                if nx == x and ny == y:
                    continue
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue
                if data[self.to_index(nx, ny, width)] >= self.occupied_threshold:
                    count += 1
        return count

    def get_robot_map_position(self, occ_grid: OccupancyGrid) -> Optional[Tuple[float, float]]:
        try:
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                occ_grid.header.frame_id,
                'base_footprint',
                rclpy.time.Time()
            )
        except TransformException:
            try:
                transform = self.tf_buffer.lookup_transform(
                    occ_grid.header.frame_id,
                    'base_link',
                    rclpy.time.Time()
                )
            except TransformException:
                return None

        wx = transform.transform.translation.x
        wy = transform.transform.translation.y

        resolution = occ_grid.info.resolution
        origin = occ_grid.info.origin

        mx = (wx - origin.position.x) / resolution - 0.5
        my = (wy - origin.position.y) / resolution - 0.5

        return mx, my

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

    def publish_point_marker(
        self,
        occ_grid: OccupancyGrid,
        cells: List[GridCell],
        publisher,
        ns: str,
        r: float,
        g: float,
        b: float,
        scale: float = 0.05
    ) -> None:
        marker = Marker()
        marker.header.frame_id = occ_grid.header.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = ns
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = scale
        marker.scale.y = scale
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = 1.0

        for mx, my in cells:
            wx, wy = self.map_to_world(occ_grid, mx, my)
            p = Point()
            p.x = wx
            p.y = wy
            p.z = 0.0
            marker.points.append(p)

        publisher.publish(marker)

    def publish_centroid_markers(
        self,
        occ_grid: OccupancyGrid,
        centroids: List[Tuple[float, float]]
    ) -> None:
        marker = Marker()
        marker.header.frame_id = occ_grid.header.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'frontier_centroids'
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = 0.12
        marker.scale.y = 0.12
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        for mx, my in centroids:
            wx, wy = self.map_to_world(occ_grid, mx, my)
            p = Point()
            p.x = wx
            p.y = wy
            p.z = 0.0
            marker.points.append(p)

        self.centroid_pub.publish(marker)

    def publish_selected_marker(
        self,
        occ_grid: OccupancyGrid,
        selected_goal: Optional[Tuple[float, float]]
    ) -> None:
        marker = Marker()
        marker.header.frame_id = occ_grid.header.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'selected_frontier'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = 0.25
        marker.scale.y = 0.25
        marker.scale.z = 0.25
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        if selected_goal is not None:
            wx, wy = self.map_to_world(occ_grid, selected_goal[0], selected_goal[1])
            marker.pose.position.x = wx
            marker.pose.position.y = wy
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0
        else:
            marker.action = Marker.DELETE

        self.selected_pub.publish(marker)

    def publish_recovery_marker(
        self,
        occ_grid: OccupancyGrid,
        recovery_goal: Optional[Tuple[float, float]]
    ) -> None:
        marker = Marker()
        marker.header.frame_id = occ_grid.header.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'recovery_goal'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = 0.22
        marker.scale.y = 0.22
        marker.scale.z = 0.22
        marker.color.r = 0.0
        marker.color.g = 0.5
        marker.color.b = 1.0
        marker.color.a = 1.0

        if recovery_goal is not None:
            wx, wy = self.map_to_world(occ_grid, recovery_goal[0], recovery_goal[1])
            marker.pose.position.x = wx
            marker.pose.position.y = wy
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0
        else:
            marker.action = Marker.DELETE

        self.recovery_pub.publish(marker)

    def send_goal(self, x: float, y: float, frame_id: str) -> None:
        if not self.nav_to_pose_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn('NavigateToPose action server not available.')
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = frame_id
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.position.z = 0.0
        goal_msg.pose.pose.orientation.w = 1.0

        self.goal_in_progress = True
        self.current_goal = (x, y)
        self.goal_start_time = self.get_clock().now()
        self.last_sent_goal = (x, y)

        self.get_logger().info(f'Sending goal to frontier: x={x:.2f}, y={y:.2f}')

        send_goal_future = self.nav_to_pose_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future) -> None:
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().warn('Goal was rejected by Nav2.')

            if self.current_goal is not None:
                self.add_failed_goal(self.current_goal)

            self.recovery_failure_count += 1
            self.last_failure_time = self.get_clock().now()

            self.goal_in_progress = False
            self.current_goal = None
            self.goal_start_time = None
            self.last_sent_goal = None
            return

        self.get_logger().info('Goal accepted by Nav2.')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_result_callback)

    def goal_result_callback(self, future) -> None:
        result_msg = future.result()
        status = result_msg.status
        result = result_msg.result

        error_code = getattr(result, 'error_code', None)
        error_msg = getattr(result, 'error_msg', '')

        if error_code is not None:
            self.get_logger().info(
                f'Goal finished with status code: {status}, '
                f'error_code: {error_code}, '
                f'error_msg: {error_msg}'
            )
        else:
            self.get_logger().info(
                f'Goal finished with status code: {status}'
            )

        if status == 4:
            self.recovery_failure_count = 0
        else:
            if self.current_goal is not None:
                self.add_failed_goal(self.current_goal)
            self.recovery_failure_count += 1
            self.last_failure_time = self.get_clock().now()

        self.goal_in_progress = False
        self.current_goal = None
        self.goal_start_time = None
        self.last_sent_goal = None

    def feedback_callback(self, feedback_msg) -> None:
        pass

    def is_free_cell(
        self,
        occ_grid: OccupancyGrid,
        x: int,
        y: int
    ) -> bool:
        width = occ_grid.info.width
        height = occ_grid.info.height

        if x < 0 or y < 0 or x >= width or y >= height:
            return False

        data = list(occ_grid.data)
        idx = self.to_index(x, y, width)
        return 0 <= data[idx] < self.occupied_threshold

    def compute_reachable_cells(
        self,
        occ_grid: OccupancyGrid
    ) -> Set[GridCell]:
        robot_map = self.get_robot_map_position(occ_grid)
        if robot_map is None:
            return set()

        start_x = int(round(robot_map[0]))
        start_y = int(round(robot_map[1]))

        if not self.is_free_cell(occ_grid, start_x, start_y):
            return set()

        width = occ_grid.info.width
        height = occ_grid.info.height
        data = list(occ_grid.data)

        reachable: Set[GridCell] = set()
        queue = deque()
        queue.append((start_x, start_y))
        reachable.add((start_x, start_y))

        while queue:
            cx, cy = queue.popleft()

            for nx, ny in [
                (cx + 1, cy),
                (cx - 1, cy),
                (cx, cy + 1),
                (cx, cy - 1)
            ]:
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue

                if (nx, ny) in reachable:
                    continue

                idx = self.to_index(nx, ny, width)

                if not (0 <= data[idx] < self.occupied_threshold):
                    continue

                reachable.add((nx, ny))
                queue.append((nx, ny))

        return reachable

    def is_goal_reachable(
        self,
        reachable_cells: Set[GridCell],
        gx: int,
        gy: int
    ) -> bool:
        return (gx, gy) in reachable_cells


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ExplorationPlannerNode()

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