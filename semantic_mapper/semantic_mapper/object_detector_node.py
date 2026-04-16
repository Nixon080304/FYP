#!/usr/bin/env python3

from typing import List

import cv2
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, BoundingBox2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
from ultralytics import YOLO


class ObjectDetectorNode(Node):
    def __init__(self) -> None:
        super().__init__('object_detector')

        self.declare_parameter('model_path', 'yolov8n.pt')
        self.declare_parameter('image_topic', '/camera/camera/image_raw')
        self.declare_parameter('detection_topic', '/yolo/detections')
        self.declare_parameter('annotated_image_topic', '/yolo/annotated_image')
        self.declare_parameter('conf_threshold', 0.40)
        self.declare_parameter('table_conf_threshold', 0.15)
        self.declare_parameter('image_size', 960)

        self.bridge = CvBridge()
        self.model_path = str(self.get_parameter('model_path').value)
        self.image_topic = str(self.get_parameter('image_topic').value)
        self.detection_topic = str(self.get_parameter('detection_topic').value)
        self.annotated_image_topic = str(self.get_parameter('annotated_image_topic').value)
        self.conf_threshold = float(self.get_parameter('conf_threshold').value)
        self.table_conf_threshold = float(self.get_parameter('table_conf_threshold').value)
        self.image_size = int(self.get_parameter('image_size').value)

        self.model = YOLO(self.model_path)

        self.class_aliases = {
            'table': 'dining table',
            'dining_table': 'dining table',
        }

        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10
        )

        self.detection_pub = self.create_publisher(
            Detection2DArray,
            self.detection_topic,
            10
        )

        self.annotated_image_pub = self.create_publisher(
            Image,
            self.annotated_image_topic,
            10
        )

        self.last_labels: List[str] = []
        self.get_logger().info('Object detector node started.')
        self.get_logger().info(f'Model: {self.model_path}')
        self.get_logger().info(f'Confidence threshold: {self.conf_threshold:.2f}')
        self.get_logger().info(f'Table confidence threshold: {self.table_conf_threshold:.2f}')
        self.get_logger().info(f'Inference image size: {self.image_size}')
        self.get_logger().info(f'Subscribed to: {self.image_topic}')
        self.get_logger().info(f'Publishing detections to: {self.detection_topic}')
        self.get_logger().info(f'Publishing annotated image to: {self.annotated_image_topic}')

    def image_callback(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge conversion failed: {e}')
            return

        try:
            results = self.model(
                frame,
                verbose=False,
                imgsz=self.image_size,
                conf=min(self.conf_threshold, self.table_conf_threshold)
            )
        except Exception as e:
            self.get_logger().error(f'YOLO inference failed: {e}')
            return

        if not results:
            return

        result = results[0]
        names = result.names
        boxes = result.boxes

        detection_array = Detection2DArray()
        detection_array.header = msg.header

        annotated_frame = frame.copy()
        detected_labels = []

        if boxes is None:
            self.detection_pub.publish(detection_array)
            return

        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())

            raw_label = str(names[cls_id])
            label = self.normalize_label(raw_label)

            if conf < self.class_conf_threshold(label):
                continue

            xyxy = box.xyxy[0].tolist()
            x_min, y_min, x_max, y_max = xyxy

            width = x_max - x_min
            height = y_max - y_min
            center_x = x_min + width / 2.0
            center_y = y_min + height / 2.0

            detection_msg = Detection2D()
            detection_msg.header = msg.header

            bbox = BoundingBox2D()
            bbox.center.position.x = float(center_x)
            bbox.center.position.y = float(center_y)
            bbox.size_x = float(width)
            bbox.size_y = float(height)
            detection_msg.bbox = bbox

            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = label
            hypothesis.hypothesis.score = conf

            detection_msg.results.append(hypothesis)
            detection_array.detections.append(detection_msg)

            detected_labels.append(f'{label} ({conf:.2f})')

            cv2.rectangle(
                annotated_frame,
                (int(x_min), int(y_min)),
                (int(x_max), int(y_max)),
                (0, 255, 0),
                2
            )
            cv2.putText(
                annotated_frame,
                f'{label} {conf:.2f}',
                (int(x_min), max(0, int(y_min) - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        self.detection_pub.publish(detection_array)

        try:
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
            annotated_msg.header = msg.header
            self.annotated_image_pub.publish(annotated_msg)
        except Exception as e:
            self.get_logger().error(f'Failed to publish annotated image: {e}')

        detected_labels = sorted(set(detected_labels))
        if detected_labels != self.last_labels and detected_labels:
            self.last_labels = detected_labels
            self.get_logger().info(f'Detected: {", ".join(detected_labels)}')

    def normalize_label(self, label: str) -> str:
        normalized = label.strip().lower()
        return self.class_aliases.get(normalized, normalized)

    def class_conf_threshold(self, label: str) -> float:
        if label == 'dining table':
            return self.table_conf_threshold
        return self.conf_threshold


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ObjectDetectorNode()

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
