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

        self.bridge = CvBridge()
        self.model = YOLO('yolov8n.pt')
        self.conf_threshold = 0.5

        self.image_sub = self.create_subscription(
            Image,
            '/camera/camera/image_raw',
            self.image_callback,
            10
        )

        self.detection_pub = self.create_publisher(
            Detection2DArray,
            '/yolo/detections',
            10
        )

        self.annotated_image_pub = self.create_publisher(
            Image,
            '/yolo/annotated_image',
            10
        )

        self.last_labels: List[str] = []
        self.get_logger().info('Object detector node started.')
        self.get_logger().info('Subscribed to: /camera/image_raw')
        self.get_logger().info('Publishing detections to: /yolo/detections')
        self.get_logger().info('Publishing annotated image to: /yolo/annotated_image')

    def image_callback(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge conversion failed: {e}')
            return

        try:
            results = self.model(frame, verbose=False)
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

            if conf < self.conf_threshold:
                continue

            label = names[cls_id]

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