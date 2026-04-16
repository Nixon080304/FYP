#!/usr/bin/env python3

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

from sensor_msgs.msg import CameraInfo
from vision_msgs.msg import Detection2DArray
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseArray, Pose

import tf2_ros

from semantic_interfaces.msg import ProjectedSemanticObject, ProjectedSemanticObjectArray


class SemanticProjectionNode(Node):
    def __init__(self):
        super().__init__('semantic_projection_node')

        # Parameters
        self.declare_parameter('detections_topic', '/yolo/detections')
        self.declare_parameter('camera_info_topic', '/camera/camera/camera_info')
        self.declare_parameter('output_frame', 'map')
        self.declare_parameter('ground_z', 0.0)
        self.declare_parameter('min_confidence', 0.55)
        self.declare_parameter('table_min_confidence', 0.20)
        self.declare_parameter('bbox_bottom_fraction', 0.95)
        self.declare_parameter('marker_lifetime', 0.5)
        self.declare_parameter('debug', True)

        self.detections_topic = self.get_parameter('detections_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.output_frame = self.get_parameter('output_frame').value
        self.ground_z = float(self.get_parameter('ground_z').value)
        self.min_confidence = float(self.get_parameter('min_confidence').value)
        self.table_min_confidence = float(self.get_parameter('table_min_confidence').value)
        self.bbox_bottom_fraction = float(self.get_parameter('bbox_bottom_fraction').value)
        self.marker_lifetime = float(self.get_parameter('marker_lifetime').value)
        self.debug = bool(self.get_parameter('debug').value)

        self.allowed_classes = {
            'person',
            'bicycle',
            'car',
            'bed',
            'chair',
            'dining table'
        }
        self.class_aliases = {
            'table': 'dining table',
            'dining_table': 'dining table',
        }

        # Camera intrinsics
        self.camera_ready = False
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.camera_frame = None

        # TF
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subs / pubs
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            10
        )

        self.detections_sub = self.create_subscription(
            Detection2DArray,
            self.detections_topic,
            self.detections_callback,
            10
        )

        self.marker_pub = self.create_publisher(MarkerArray, '/semantic_objects/markers', 10)
        self.pose_pub = self.create_publisher(PoseArray, '/semantic_objects/poses', 10)
        self.projected_pub = self.create_publisher(
            ProjectedSemanticObjectArray,
            '/semantic_objects/projected',
            10
        )

        self.get_logger().info('Semantic projection node started.')
        self.get_logger().info(f'Detections topic: {self.detections_topic}')
        self.get_logger().info(f'CameraInfo topic: {self.camera_info_topic}')
        self.get_logger().info(f'Output frame: {self.output_frame}')
        self.get_logger().info(f'Allowed classes: {sorted(self.allowed_classes)}')
        self.get_logger().info(f'Min confidence: {self.min_confidence:.2f}')
        self.get_logger().info(f'Table min confidence: {self.table_min_confidence:.2f}')

    def camera_info_callback(self, msg: CameraInfo):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        self.camera_frame = msg.header.frame_id
        self.camera_ready = True

        if self.debug:
            self.get_logger().info(
                f'Camera intrinsics loaded | frame={self.camera_frame} '
                f'fx={self.fx:.3f} fy={self.fy:.3f} cx={self.cx:.3f} cy={self.cy:.3f}',
                throttle_duration_sec=5.0
            )

    def detections_callback(self, msg: Detection2DArray):
        if not self.camera_ready:
            self.get_logger().warn('Waiting for CameraInfo...', throttle_duration_sec=2.0)
            return

        if not self.camera_frame:
            self.get_logger().warn('Camera frame_id is empty.', throttle_duration_sec=2.0)
            return

        detection_stamp = msg.header.stamp
        detection_time = Time.from_msg(detection_stamp)

        marker_array = MarkerArray()
        pose_array = PoseArray()
        projected_array = ProjectedSemanticObjectArray()

        pose_array.header.stamp = detection_stamp
        pose_array.header.frame_id = self.output_frame

        projected_array.header.stamp = detection_stamp
        projected_array.header.frame_id = self.output_frame

        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)

        if self.debug:
            self.get_logger().info(
                f'Received Detection2DArray with {len(msg.detections)} detections',
                throttle_duration_sec=0.5
            )

        marker_id = 0

        for det in msg.detections:
            try:
                class_id, score = self.extract_best_class(det)
            except Exception as e:
                self.get_logger().warn(
                    f'Failed to parse detection class/score: {e}',
                    throttle_duration_sec=1.0
                )
                continue

            if class_id is None or score is None:
                self.get_logger().warn(
                    'Skipping detection: could not parse class_id or score',
                    throttle_duration_sec=1.0
                )
                continue

            class_name = self.normalize_class_name(class_id)

            if class_name not in self.allowed_classes:
                if self.debug:
                    self.get_logger().info(
                        f'Skipping detection: class {class_id} not in allowed classes',
                        throttle_duration_sec=1.0
                    )
                continue

            class_min_confidence = self.class_min_confidence(class_name)

            if score < class_min_confidence:
                if self.debug:
                    self.get_logger().warn(
                        f'Skipping detection: score {score:.2f} below threshold {class_min_confidence:.2f}',
                        throttle_duration_sec=1.0
                    )
                continue

            bbox = det.bbox

            try:
                u_center, v_center = self.extract_bbox_center(bbox)
            except Exception as e:
                self.get_logger().warn(
                    f'Failed to parse bbox center: {e}',
                    throttle_duration_sec=1.0
                )
                continue

            size_x = float(bbox.size_x)
            size_y = float(bbox.size_y)

            # Bottom-center pixel of bbox
            u = u_center
            v = v_center + 0.5 * size_y * self.bbox_bottom_fraction

            if self.debug:
                self.get_logger().info(
                    f'Detection: class={class_name} score={score:.2f} '
                    f'center=({u_center:.1f},{v_center:.1f}) size=({size_x:.1f},{size_y:.1f}) '
                    f'bottom_pixel=({u:.1f},{v:.1f})',
                    throttle_duration_sec=0.5
                )

            point_world = self.project_pixel_to_ground(
                u=u,
                v=v,
                source_frame=self.camera_frame,
                source_time=detection_time
            )

            if point_world is None:
                self.get_logger().warn(
                    f'Skipping detection {class_name}: projection returned None',
                    throttle_duration_sec=1.0
                )
                continue

            x_world, y_world, z_world = point_world

            pose = Pose()
            pose.position.x = float(x_world)
            pose.position.y = float(y_world)
            pose.position.z = float(z_world)
            pose.orientation.w = 1.0
            pose_array.poses.append(pose)

            projected_obj = ProjectedSemanticObject()
            projected_obj.class_name = class_name
            projected_obj.confidence = float(score)
            projected_obj.pose = pose
            projected_obj.stamp = detection_stamp
            projected_array.objects.append(projected_obj)

            sphere = Marker()
            sphere.header.frame_id = self.output_frame
            sphere.header.stamp = detection_stamp
            sphere.ns = 'semantic_objects'
            sphere.id = marker_id
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = float(x_world)
            sphere.pose.position.y = float(y_world)
            sphere.pose.position.z = 0.15
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = 0.25
            sphere.scale.y = 0.25
            sphere.scale.z = 0.25
            r, g, b = self.class_color(class_name)
            sphere.color.r = r
            sphere.color.g = g
            sphere.color.b = b
            sphere.color.a = 0.9
            sphere.lifetime = Duration(seconds=self.marker_lifetime).to_msg()
            marker_array.markers.append(sphere)
            marker_id += 1

            text = Marker()
            text.header.frame_id = self.output_frame
            text.header.stamp = detection_stamp
            text.ns = 'semantic_labels'
            text.id = marker_id
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = float(x_world)
            text.pose.position.y = float(y_world)
            text.pose.position.z = 0.45
            text.pose.orientation.w = 1.0
            text.scale.z = 0.20
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 1.0
            text.text = f'{class_name} ({score:.2f})'
            text.lifetime = Duration(seconds=self.marker_lifetime).to_msg()
            marker_array.markers.append(text)
            marker_id += 1

            self.get_logger().info(
                f'Projected {class_name} -> {self.output_frame} ({x_world:.2f}, {y_world:.2f}, {z_world:.2f})',
                throttle_duration_sec=0.5
            )

        self.pose_pub.publish(pose_array)
        self.marker_pub.publish(marker_array)
        self.projected_pub.publish(projected_array)

    def project_pixel_to_ground(
        self,
        u: float,
        v: float,
        source_frame: str,
        source_time: Time
    ):
        # Pixel -> ray in camera frame
        x_cam = (u - self.cx) / self.fx
        y_cam = (v - self.cy) / self.fy
        ray_cam = np.array([x_cam, y_cam, 1.0], dtype=np.float64)
        ray_cam /= np.linalg.norm(ray_cam)

        try:
            tf = self.tf_buffer.lookup_transform(
                self.output_frame,
                source_frame,
                source_time,
                timeout=Duration(seconds=1.0)
            )
        except Exception as e:
            self.get_logger().warn(
                f'Could not transform {source_frame} -> {self.output_frame}: {e}',
                throttle_duration_sec=1.0
            )
            return None

        tx = tf.transform.translation.x
        ty = tf.transform.translation.y
        tz = tf.transform.translation.z
        origin_world = np.array([tx, ty, tz], dtype=np.float64)

        qx = tf.transform.rotation.x
        qy = tf.transform.rotation.y
        qz = tf.transform.rotation.z
        qw = tf.transform.rotation.w
        rot_world_from_cam = self.quaternion_to_rotation_matrix(qx, qy, qz, qw)

        ray_world = rot_world_from_cam @ ray_cam

        if self.debug:
            self.get_logger().info(
                f'Ray in {self.output_frame}: origin=({origin_world[0]:.2f},{origin_world[1]:.2f},{origin_world[2]:.2f}) '
                f'dir=({ray_world[0]:.3f},{ray_world[1]:.3f},{ray_world[2]:.3f})',
                throttle_duration_sec=0.5
            )

        denom = ray_world[2]
        if abs(denom) < 1e-6:
            return None

        t = (self.ground_z - origin_world[2]) / denom
        if t <= 0.0:
            return None

        point_world = origin_world + t * ray_world
        return point_world[0], point_world[1], point_world[2]

    def extract_bbox_center(self, bbox):
        center = bbox.center

        if hasattr(center, 'x') and hasattr(center, 'y'):
            return float(center.x), float(center.y)

        if hasattr(center, 'position'):
            return float(center.position.x), float(center.position.y)

        raise AttributeError('Unsupported BoundingBox2D center format.')

    def extract_best_class(self, det):
        if len(det.results) == 0:
            return None, None

        best_class = None
        best_score = -1.0

        for r in det.results:
            class_id = None
            score = None

            if hasattr(r, 'hypothesis'):
                if hasattr(r.hypothesis, 'class_id'):
                    class_id = str(r.hypothesis.class_id)
                if hasattr(r.hypothesis, 'score'):
                    score = float(r.hypothesis.score)

            if class_id is None and hasattr(r, 'id'):
                class_id = str(r.id)
            if score is None and hasattr(r, 'score'):
                score = float(r.score)

            if class_id is not None and score is not None and score > best_score:
                best_class = class_id
                best_score = score

        if best_class is None:
            return None, None

        return best_class, best_score

    def normalize_class_name(self, class_id: str) -> str:
        normalized = class_id.lower().strip()
        return self.class_aliases.get(normalized, normalized)

    def class_min_confidence(self, class_name: str) -> float:
        if class_name == 'dining table':
            return self.table_min_confidence
        return self.min_confidence

    def quaternion_to_rotation_matrix(self, x, y, z, w):
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z

        return np.array([
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
            [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)]
        ], dtype=np.float64)

    def class_color(self, class_id: str):
        name = class_id.lower()
        if name == 'person':
            return (1.0, 0.2, 0.2)
        elif name == 'car':
            return (0.2, 0.8, 1.0)
        elif name == 'bicycle':
            return (0.2, 1.0, 0.2)
        elif name == 'bed':
            return (1.0, 0.6, 0.2)
        elif name == 'chair':
            return (0.8, 0.2, 1.0)
        elif 'table' in name:
            return (1.0, 1.0, 0.2)
        else:
            return (0.7, 0.7, 0.7)


def main(args=None):
    rclpy.init(args=args)
    node = SemanticProjectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
