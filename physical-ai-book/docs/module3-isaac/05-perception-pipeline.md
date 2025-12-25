---
sidebar_position: 5
---

# Perception Pipeline with Isaac

## Chapter Objectives

By the end of this chapter, you will be able to:
- Design and implement GPU-accelerated perception pipelines
- Integrate Isaac ROS perception packages for Physical AI
- Configure object detection and tracking systems
- Build semantic segmentation and scene understanding systems
- Evaluate perception performance and accuracy

## Perception Pipeline Architecture

### Overview of Isaac Perception Stack

The Isaac perception pipeline leverages GPU acceleration for real-time processing of sensor data to enable Physical AI systems to understand their environment:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Sensors       │    │ Isaac ROS       │    │   Perception    │
│   (Cameras,     │───▶│   Perception    │───▶│   Results       │
│   LiDAR, etc.)  │    │   (GPU)         │    │   (Objects,     │
└─────────────────┘    └─────────────────┘    │   Segmentation, │
                                              │   Tracking, etc.)│
                                              └─────────────────┘
```

### Key Perception Components

1. **Image Processing**: GPU-accelerated image rectification, filtering, and enhancement
2. **Object Detection**: Real-time detection of objects using deep learning
3. **Semantic Segmentation**: Pixel-level scene understanding
4. **Instance Segmentation**: Individual object segmentation
5. **Object Tracking**: Multi-object tracking across frames
6. **Pose Estimation**: 6-DOF pose estimation for objects
7. **Scene Understanding**: 3D scene reconstruction and understanding

## Isaac ROS Perception Packages

### Core Perception Packages

1. **isaac_ros_image_pipeline**: GPU-accelerated image processing
2. **isaac_ros_detectnet**: Object detection with NVIDIA DetectNet
3. **isaac_ros_segmentation**: Semantic segmentation
4. **isaac_ros_pose_estimation**: 6-DOF pose estimation
5. **isaac_ros_tracking**: Multi-object tracking
6. **isaac_ros_pointcloud_utils**: Point cloud processing

### Image Pipeline Configuration

```yaml
# File: config/image_pipeline_params.yaml
isaac_ros_image_pipeline:
  ros__parameters:
    # Input settings
    input_width: 640
    input_height: 480
    input_format: "rgb8"

    # Processing settings
    enable_rectification: true
    enable_resize: false
    enable_format_conversion: true

    # GPU settings
    cuda_device: 0
    enable_profiler: false

    # Output settings
    output_qos: 10
    publish_processed_images: true
```

### Launch File for Image Pipeline

```python
# File: launch/image_pipeline.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    config_file = os.path.join(
        get_package_share_directory('your_package'),
        'config',
        'image_pipeline_params.yaml'
    )

    # Image rectification node
    rectify_node = Node(
        package='isaac_ros_image_pipeline',
        executable='image_rectification_node',
        parameters=[config_file],
        remappings=[
            ('image_raw', '/camera/image_raw'),
            ('camera_info', '/camera/camera_info'),
            ('image_rect', '/camera/image_rect_color')
        ]
    )

    # Image resize node
    resize_node = Node(
        package='isaac_ros_image_pipeline',
        executable='image_resize_node',
        parameters=[config_file],
        remappings=[
            ('image', '/camera/image_rect_color'),
            ('resized_image', '/camera/image_resized')
        ]
    )

    return LaunchDescription([
        rectify_node,
        resize_node
    ])
```

## Object Detection with Isaac

### Isaac DetectNet Configuration

```yaml
# File: config/detectnet_params.yaml
isaac_ros_detectnet:
  ros__parameters:
    # Model settings
    model_name: "ssd_mobilenet_v2_coco"
    confidence_threshold: 0.7
    enable_profiler: false

    # Input settings
    input_width: 640
    input_height: 480
    input_format: "rgb8"

    # GPU settings
    cuda_device: 0
    input_tensor: "input_tensor"
    output_coverage: "output_cov"
    output_bbox: "output_bbox"

    # Output settings
    publish_overlay: true
    publish_mask: false
    mask_pool_size: 16

    # Topic settings
    image_input_topic: "/camera/image_rect_color"
    camera_info_input_topic: "/camera/camera_info"
    detections_output_topic: "/detectnet/detections"
    overlay_image_output_topic: "/detectnet/overlay"
```

### Object Detection Node Implementation

```python
# File: perception/object_detection_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np

class IsaacObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('isaac_object_detection_node')

        # Create CV bridge
        self.bridge = CvBridge()

        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_rect_color',
            self.image_callback,
            10
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera_info',
            self.camera_info_callback,
            10
        )

        # Publishers
        self.detections_pub = self.create_publisher(
            Detection2DArray,
            '/object_detections',
            10
        )

        self.overlay_pub = self.create_publisher(
            Image,
            '/detection_overlay',
            10
        )

        # State
        self.camera_info = None
        self.detection_model = None  # Would be Isaac DetectNet model

        # Detection parameters
        self.confidence_threshold = 0.7

        self.get_logger().info('Isaac Object Detection Node Started')

    def camera_info_callback(self, msg):
        """Process camera calibration information"""
        self.camera_info = msg

    def image_callback(self, msg):
        """Process incoming images for object detection"""
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Perform object detection (Isaac GPU-accelerated)
            detections = self.perform_detection(cv_image)

            # Create detection messages
            detection_msg = self.create_detection_message(detections, msg.header)

            # Create overlay image
            overlay_image = self.create_detection_overlay(cv_image, detections)

            # Publish results
            self.detections_pub.publish(detection_msg)

            overlay_msg = self.bridge.cv2_to_imgmsg(overlay_image, "bgr8")
            overlay_msg.header = msg.header
            self.overlay_pub.publish(overlay_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def perform_detection(self, image):
        """Perform GPU-accelerated object detection"""
        # In real implementation, this would use Isaac DetectNet
        # For demonstration, using a placeholder
        detections = []

        # Simulate detection results
        if np.random.random() > 0.5:  # Random detection for demo
            detections.append({
                'bbox': [100, 100, 200, 150],  # [x, y, width, height]
                'class_id': 1,
                'class_name': 'person',
                'confidence': 0.85
            })

        return detections

    def create_detection_message(self, detections, header):
        """Create vision_msgs Detection2DArray message"""
        detection_array = Detection2DArray()
        detection_array.header = header

        for detection in detections:
            detection_msg = Detection2D()
            detection_msg.header = header

            # Set bounding box
            bbox = detection['bbox']
            detection_msg.bbox.center.x = bbox[0] + bbox[2] / 2  # center x
            detection_msg.bbox.center.y = bbox[1] + bbox[3] / 2  # center y
            detection_msg.bbox.size_x = bbox[2]  # width
            detection_msg.bbox.size_y = bbox[3]  # height

            # Set hypothesis
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = str(detection['class_name'])
            hypothesis.hypothesis.score = detection['confidence']
            detection_msg.results.append(hypothesis)

            detection_array.detections.append(detection_msg)

        return detection_array

    def create_detection_overlay(self, image, detections):
        """Create image overlay with detection bounding boxes"""
        overlay = image.copy()

        for detection in detections:
            bbox = detection['bbox']
            x, y, w, h = bbox

            # Draw bounding box
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw label
            label = f"{detection['class_name']}: {detection['confidence']:.2f}"
            cv2.putText(overlay, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return overlay

def main(args=None):
    rclpy.init(args=args)
    node = IsaacObjectDetectionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Semantic Segmentation

### Isaac Segmentation Configuration

```yaml
# File: config/segmentation_params.yaml
isaac_ros_segmentation:
  ros__parameters:
    # Model settings
    model_name: "unet_coco"
    confidence_threshold: 0.5
    colormap: "coco_colormap"

    # Input settings
    input_width: 640
    input_height: 480
    input_format: "rgb8"

    # GPU settings
    cuda_device: 0
    enable_profiler: false

    # Output settings
    publish_segmentation_map: true
    publish_overlay: true
    publish_mask: true

    # Topic settings
    image_input_topic: "/camera/image_rect_color"
    segmentation_output_topic: "/segmentation/segmentation_map"
    overlay_output_topic: "/segmentation/overlay"
    mask_output_topic: "/segmentation/mask"
```

### Segmentation Node Implementation

```python
# File: perception/segmentation_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np

class IsaacSegmentationNode(Node):
    def __init__(self):
        super().__init__('isaac_segmentation_node')

        # Create CV bridge
        self.bridge = CvBridge()

        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_rect_color',
            self.image_callback,
            10
        )

        # Publishers
        self.segmentation_pub = self.create_publisher(
            Image,
            '/segmentation/segmentation_map',
            10
        )

        self.overlay_pub = self.create_publisher(
            Image,
            '/segmentation/overlay',
            10
        )

        # State
        self.segmentation_model = None  # Would be Isaac segmentation model

        self.get_logger().info('Isaac Segmentation Node Started')

    def image_callback(self, msg):
        """Process incoming images for semantic segmentation"""
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Perform segmentation (Isaac GPU-accelerated)
            segmentation_map = self.perform_segmentation(cv_image)

            # Create overlay
            overlay = self.create_segmentation_overlay(cv_image, segmentation_map)

            # Publish results
            seg_msg = self.bridge.cv2_to_imgmsg(segmentation_map.astype(np.uint8), "mono8")
            seg_msg.header = msg.header
            self.segmentation_pub.publish(seg_msg)

            overlay_msg = self.bridge.cv2_to_imgmsg(overlay, "bgr8")
            overlay_msg.header = msg.header
            self.overlay_pub.publish(overlay_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def perform_segmentation(self, image):
        """Perform GPU-accelerated semantic segmentation"""
        # In real implementation, this would use Isaac segmentation
        # For demonstration, creating a simple segmentation map
        height, width = image.shape[:2]

        # Create a simple segmentation map (class IDs)
        segmentation_map = np.zeros((height, width), dtype=np.uint8)

        # Simulate some segmentation results
        # In reality, this would come from a trained model
        for i in range(5):  # Simulate 5 different objects
            center_x = np.random.randint(50, width-50)
            center_y = np.random.randint(50, height-50)
            radius = np.random.randint(20, 50)

            y, x = np.ogrid[:height, :width]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            segmentation_map[mask] = i + 1  # Class ID

        return segmentation_map

    def create_segmentation_overlay(self, image, segmentation_map):
        """Create overlay with colored segmentation"""
        # Generate color map for different classes
        colored_mask = self.apply_colormap(segmentation_map)

        # Blend original image with segmentation mask
        alpha = 0.6  # Transparency factor
        overlay = cv2.addWeighted(image, alpha, colored_mask, 1 - alpha, 0)

        return overlay

    def apply_colormap(self, segmentation_map):
        """Apply color map to segmentation results"""
        # Create a color map with different colors for each class
        height, width = segmentation_map.shape
        colored_map = np.zeros((height, width, 3), dtype=np.uint8)

        # Define colors for different classes
        colors = [
            [0, 0, 0],      # Class 0: Background (black)
            [255, 0, 0],    # Class 1: Red
            [0, 255, 0],    # Class 2: Green
            [0, 0, 255],    # Class 3: Blue
            [255, 255, 0],  # Class 4: Yellow
            [255, 0, 255],  # Class 5: Magenta
        ]

        for class_id in range(len(colors)):
            mask = segmentation_map == class_id
            colored_map[mask] = colors[class_id]

        return colored_map

def main(args=None):
    rclpy.init(args=args)
    node = IsaacSegmentationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Multi-Object Tracking

### Isaac Tracking Configuration

```yaml
# File: config/tracking_params.yaml
isaac_ros_tracking:
  ros__parameters:
    # Tracking settings
    max_objects: 50
    max_track_age: 100
    min_track_age: 5
    matching_threshold: 0.3

    # Detection settings
    detection_topic: "/detectnet/detections"
    confidence_threshold: 0.7

    # GPU settings
    cuda_device: 0
    enable_profiler: false

    # Output settings
    publish_tracked_objects: true
    publish_trajectories: true

    # Topic settings
    image_input_topic: "/camera/image_rect_color"
    tracked_objects_output_topic: "/tracking/tracked_objects"
    trajectory_output_topic: "/tracking/trajectories"
```

### Tracking Node Implementation

```python
# File: perception/tracking_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import Header
from geometry_msgs.msg import Point
import numpy as np
from collections import deque
import cv2

class TrackedObject:
    def __init__(self, detection, track_id):
        self.id = track_id
        self.detections = deque(maxlen=50)  # Store last 50 detections
        self.kalman_filter = self.initialize_kalman_filter(detection)
        self.age = 0
        self.hits = 1
        self.hit_streak = 1
        self.time_since_update = 0

    def initialize_kalman_filter(self, detection):
        """Initialize Kalman filter for tracking"""
        # Simple Kalman filter for position and velocity
        # In practice, would use more sophisticated tracking
        bbox = detection.bbox
        x = bbox.center.x
        y = bbox.center.y

        # State: [x, y, vx, vy]
        kf = {
            'state': np.array([x, y, 0, 0], dtype=np.float32),
            'covariance': np.eye(4, dtype=np.float32) * 100,
            'process_noise': np.eye(4, dtype=np.float32) * 0.1,
            'measurement_noise': np.eye(2, dtype=np.float32) * 10
        }
        return kf

    def predict(self):
        """Predict next state"""
        # Simple constant velocity model
        dt = 1.0  # Time step
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        self.kalman_filter['state'] = F @ self.kalman_filter['state']
        P = self.kalman_filter['covariance']
        Q = self.kalman_filter['process_noise']
        self.kalman_filter['covariance'] = F @ P @ F.T + Q

        return self.kalman_filter['state'][:2]  # Return position [x, y]

    def update(self, measurement):
        """Update with new measurement"""
        # Extract position from measurement
        z = np.array([measurement.bbox.center.x, measurement.bbox.center.y], dtype=np.float32)

        # Measurement matrix
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        # Predicted measurement
        x = self.kalman_filter['state']
        P = self.kalman_filter['covariance']
        R = self.kalman_filter['measurement_noise']

        y = z - H @ x  # Innovation
        S = H @ P @ H.T + R  # Innovation covariance
        K = P @ H.T @ np.linalg.inv(S)  # Kalman gain

        # Update state and covariance
        self.kalman_filter['state'] = x + K @ y
        I = np.eye(len(x))
        self.kalman_filter['covariance'] = (I - K @ H) @ P

        # Store detection
        self.detections.append(measurement)
        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0

class IsaacTrackingNode(Node):
    def __init__(self):
        super().__init__('isaac_tracking_node')

        # Subscribers
        self.detections_sub = self.create_subscription(
            Detection2DArray,
            '/object_detections',
            self.detections_callback,
            10
        )

        # Publishers
        self.tracked_objects_pub = self.create_publisher(
            Detection2DArray,
            '/tracking/tracked_objects',
            10
        )

        # State
        self.tracks = []
        self.next_track_id = 0
        self.max_disappeared = 10  # Frames before deleting track

        self.get_logger().info('Isaac Tracking Node Started')

    def detections_callback(self, msg):
        """Process new detections and update tracks"""
        # Convert detections to format suitable for tracking
        detections = []
        for detection in msg.detections:
            detections.append(detection)

        # Update existing tracks
        self.update_tracks(detections)

        # Create message with tracked objects
        tracked_msg = Detection2DArray()
        tracked_msg.header = msg.header

        for track in self.tracks:
            if track.time_since_update == 0:  # Only include active tracks
                # Create detection message with track ID
                detection = Detection2D()
                detection.header = msg.header
                detection.bbox = track.detections[-1].bbox  # Use latest detection

                # Add track ID as class name
                hypothesis = track.detections[-1].results[0] if track.detections else None
                if hypothesis:
                    hypothesis.hypothesis.class_id = f"track_{track.id}"
                    detection.results.append(hypothesis)

                tracked_msg.detections.append(detection)

        self.tracked_objects_pub.publish(tracked_msg)

    def update_tracks(self, detections):
        """Update tracks with new detections"""
        # Predict new positions for all tracks
        for track in self.tracks:
            track.predict()

        # If no tracks exist, create new ones for all detections
        if len(self.tracks) == 0:
            for detection in detections:
                new_track = TrackedObject(detection, self.next_track_id)
                self.tracks.append(new_track)
                self.next_track_id += 1
            return

        # Calculate distance matrix between tracks and detections
        if len(detections) > 0:
            # Simple distance-based association
            track_predictions = [track.predict() for track in self.tracks]
            detection_positions = [
                [detection.bbox.center.x, detection.bbox.center.y]
                for detection in detections
            ]

            # Associate tracks with detections (simple nearest neighbor)
            used_detections = set()
            for i, track_pos in enumerate(track_predictions):
                min_dist = float('inf')
                best_det_idx = -1

                for j, det_pos in enumerate(detection_positions):
                    if j in used_detections:
                        continue

                    dist = np.sqrt(
                        (track_pos[0] - det_pos[0])**2 +
                        (track_pos[1] - det_pos[1])**2
                    )

                    if dist < min_dist and dist < 50:  # Threshold for association
                        min_dist = dist
                        best_det_idx = j

                if best_det_idx != -1:
                    # Update track with associated detection
                    self.tracks[i].update(detections[best_det_idx])
                    used_detections.add(best_det_idx)
                else:
                    # No detection associated, increment time since update
                    self.tracks[i].time_since_update += 1

            # Create new tracks for unassociated detections
            for i, detection in enumerate(detections):
                if i not in used_detections:
                    new_track = TrackedObject(detection, self.next_track_id)
                    self.tracks.append(new_track)
                    self.next_track_id += 1

        # Remove old tracks
        self.tracks = [
            track for track in self.tracks
            if track.time_since_update <= self.max_disappeared
        ]

def main(args=None):
    rclpy.init(args=args)
    node = IsaacTrackingNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 6-DOF Pose Estimation

### Isaac Pose Estimation Configuration

```yaml
# File: config/pose_estimation_params.yaml
isaac_ros_pose_estimation:
  ros__parameters:
    # Model settings
    model_name: "6dof_pose_model"
    confidence_threshold: 0.8
    enable_refinement: true

    # Input settings
    input_width: 640
    input_height: 480
    input_format: "rgb8"

    # GPU settings
    cuda_device: 0
    enable_profiler: false

    # Object settings
    object_models_path: "/path/to/object/models"
    enable_multi_object: true

    # Output settings
    publish_poses: true
    publish_visualization: true

    # Topic settings
    image_input_topic: "/camera/image_rect_color"
    camera_info_input_topic: "/camera/camera_info"
    poses_output_topic: "/pose_estimation/poses"
    visualization_output_topic: "/pose_estimation/visualization"
```

### Pose Estimation Node

```python
# File: perception/pose_estimation_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, PoseArray
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np
from tf_transformations import quaternion_from_euler

class IsaacPoseEstimationNode(Node):
    def __init__(self):
        super().__init__('isaac_pose_estimation_node')

        # Create CV bridge
        self.bridge = CvBridge()

        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_rect_color',
            self.image_callback,
            10
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera_info',
            self.camera_info_callback,
            10
        )

        # Publishers
        self.poses_pub = self.create_publisher(
            PoseArray,
            '/pose_estimation/poses',
            10
        )

        self.visualization_pub = self.create_publisher(
            Image,
            '/pose_estimation/visualization',
            10
        )

        # State
        self.camera_matrix = None
        self.distortion_coeffs = None
        self.pose_model = None  # Would be Isaac pose estimation model

        self.get_logger().info('Isaac Pose Estimation Node Started')

    def camera_info_callback(self, msg):
        """Process camera calibration information"""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.distortion_coeffs = np.array(msg.d)

    def image_callback(self, msg):
        """Process incoming images for pose estimation"""
        if self.camera_matrix is None:
            return

        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Perform pose estimation
            poses = self.estimate_poses(cv_image)

            # Create PoseArray message
            pose_array = PoseArray()
            pose_array.header = msg.header
            pose_array.poses = poses

            # Create visualization
            viz_image = self.create_pose_visualization(cv_image, poses)

            # Publish results
            self.poses_pub.publish(pose_array)

            viz_msg = self.bridge.cv2_to_imgmsg(viz_image, "bgr8")
            viz_msg.header = msg.header
            self.visualization_pub.publish(viz_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def estimate_poses(self, image):
        """Estimate 6-DOF poses of objects in image"""
        # In real implementation, this would use Isaac pose estimation
        # For demonstration, returning placeholder poses
        poses = []

        # Detect objects first (simplified)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter small contours
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)

                # Estimate 3D pose (simplified)
                center_x = x + w / 2
                center_y = y + h / 2

                # Convert pixel coordinates to 3D (simplified)
                z = 1.0  # Assume fixed depth for demo
                x_3d = (center_x - self.camera_matrix[0, 2]) * z / self.camera_matrix[0, 0]
                y_3d = (center_y - self.camera_matrix[1, 2]) * z / self.camera_matrix[1, 1]

                # Create pose
                pose = PoseStamped()
                pose.pose.position.x = x_3d
                pose.pose.position.y = y_3d
                pose.pose.position.z = z

                # Set orientation (simplified)
                quat = quaternion_from_euler(0, 0, 0)
                pose.pose.orientation.x = quat[0]
                pose.pose.orientation.y = quat[1]
                pose.pose.orientation.z = quat[2]
                pose.pose.orientation.w = quat[3]

                poses.append(pose.pose)

        return poses

    def create_pose_visualization(self, image, poses):
        """Create visualization with pose information"""
        viz_image = image.copy()

        for i, pose in enumerate(poses):
            # Convert 3D position back to 2D for visualization
            point_3d = np.array([pose.position.x, pose.position.y, pose.position.z])
            rvec = np.array([0, 0, 0])  # Rotation vector (simplified)
            tvec = np.array([0, 0, 0])  # Translation vector (simplified)

            # Project 3D point to 2D
            point_2d, _ = cv2.projectPoints(
                point_3d.reshape(1, 1, 3),
                rvec, tvec,
                self.camera_matrix,
                self.distortion_coeffs if self.distortion_coeffs is not None else np.array([])
            )

            # Draw pose indicator
            center = (int(point_2d[0][0][0]), int(point_2d[0][0][1]))
            cv2.circle(viz_image, center, 10, (0, 255, 0), 2)
            cv2.putText(viz_image, f'Object {i}', center,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return viz_image

def main(args=None):
    rclpy.init(args=args)
    node = IsaacPoseEstimationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Perception Pipeline Integration

### Complete Perception Pipeline Launch

```python
# File: launch/perception_pipeline.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    config_dir = LaunchConfiguration('config_dir', default='config')

    # Get package path
    pkg_perception = FindPackageShare('isaac_perception_demos')

    # Image pipeline
    image_pipeline = Node(
        package='isaac_ros_image_pipeline',
        executable='image_pipeline_node',
        parameters=[
            PathJoinSubstitution([pkg_perception, config_dir, 'image_pipeline_params.yaml'])
        ],
        remappings=[
            ('image_raw', '/camera/image_raw'),
            ('camera_info', '/camera/camera_info'),
            ('image_rect', '/camera/image_rect_color')
        ]
    )

    # Object detection
    object_detection = Node(
        package='isaac_ros_detectnet',
        executable='detectnet_node',
        parameters=[
            PathJoinSubstitution([pkg_perception, config_dir, 'detectnet_params.yaml'])
        ],
        remappings=[
            ('image', '/camera/image_rect_color'),
            ('camera_info', '/camera/camera_info'),
            ('detections', '/object_detections')
        ]
    )

    # Semantic segmentation
    segmentation = Node(
        package='isaac_ros_segmentation',
        executable='segmentation_node',
        parameters=[
            PathJoinSubstitution([pkg_perception, config_dir, 'segmentation_params.yaml'])
        ],
        remappings=[
            ('image', '/camera/image_rect_color'),
            ('segmentation_map', '/segmentation/map')
        ]
    )

    # Object tracking
    tracking = Node(
        package='isaac_perception_demos',
        executable='tracking_node',
        parameters=[
            PathJoinSubstitution([pkg_perception, config_dir, 'tracking_params.yaml'])
        ],
        remappings=[
            ('detections', '/object_detections'),
            ('tracked_objects', '/tracked_objects')
        ]
    )

    # Pose estimation
    pose_estimation = Node(
        package='isaac_perception_demos',
        executable='pose_estimation_node',
        parameters=[
            PathJoinSubstitution([pkg_perception, config_dir, 'pose_estimation_params.yaml'])
        ],
        remappings=[
            ('image', '/camera/image_rect_color'),
            ('camera_info', '/camera/camera_info'),
            ('poses', '/object_poses')
        ]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation time'
        ),
        DeclareLaunchArgument(
            'config_dir',
            default_value='config',
            description='Configuration directory'
        ),
        image_pipeline,
        object_detection,
        segmentation,
        tracking,
        pose_estimation
    ])
```

## Performance Evaluation

### Perception Quality Metrics

```python
# File: perception/perception_evaluator.py
import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Int32
from cv_bridge import CvBridge
import numpy as np

class PerceptionEvaluator(Node):
    def __init__(self):
        super().__init__('perception_evaluator')

        # Create CV bridge
        self.bridge = CvBridge()

        # Subscribers
        self.detections_sub = self.create_subscription(
            Detection2DArray,
            '/object_detections',
            self.detections_callback,
            10
        )

        # Publishers
        self.detection_rate_pub = self.create_publisher(Float32, '/perception/detection_rate', 10)
        self.accuracy_pub = self.create_publisher(Float32, '/perception/accuracy', 10)
        self.object_count_pub = self.create_publisher(Int32, '/perception/object_count', 10)

        # State
        self.detection_history = []
        self.frame_count = 0
        self.detection_count = 0
        self.last_time = self.get_clock().now()

        # Timer for evaluation
        self.eval_timer = self.create_timer(1.0, self.evaluate_performance)

        self.get_logger().info('Perception Evaluator Node Started')

    def detections_callback(self, msg):
        """Process detection results"""
        self.detection_count += len(msg.detections)
        self.frame_count += 1

        # Store detection confidence for accuracy calculation
        for detection in msg.detections:
            if detection.results:
                confidence = detection.results[0].hypothesis.score
                self.detection_history.append(confidence)

    def evaluate_performance(self):
        """Evaluate perception performance metrics"""
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9

        if dt > 0 and self.frame_count > 0:
            # Detection rate (detections per second)
            detection_rate = self.detection_count / dt
            rate_msg = Float32()
            rate_msg.data = detection_rate
            self.detection_rate_pub.publish(rate_msg)

            # Average confidence (proxy for accuracy)
            avg_confidence = np.mean(self.detection_history) if self.detection_history else 0.0
            accuracy_msg = Float32()
            accuracy_msg.data = avg_confidence
            self.accuracy_pub.publish(accuracy_msg)

            # Object count
            obj_count_msg = Int32()
            obj_count_msg.data = len(self.detection_history)
            self.object_count_pub.publish(obj_count_msg)

            # Log metrics
            self.get_logger().info(
                f'Perception Metrics - Rate: {detection_rate:.2f} det/s, '
                f'Accuracy: {avg_confidence:.3f}, Objects: {len(self.detection_history)}'
            )

        # Reset counters
        self.last_time = current_time
        self.frame_count = 0
        self.detection_count = 0
        self.detection_history = []  # Keep only recent history

def main(args=None):
    rclpy.init(args=args)
    evaluator = PerceptionEvaluator()

    try:
        rclpy.spin(evaluator)
    except KeyboardInterrupt:
        pass
    finally:
        evaluator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Best Practices for Isaac Perception

### Configuration Best Practices
- Use appropriate models for your specific use case
- Configure input resolution based on GPU capabilities
- Set confidence thresholds to balance precision and recall
- Use proper camera calibration for accurate measurements

### Performance Best Practices
- Optimize pipeline for your specific GPU
- Use appropriate batch sizes for deep learning models
- Implement multi-threading for CPU-bound operations
- Monitor GPU utilization and memory usage

### Integration Best Practices
- Validate perception results against ground truth when possible
- Implement fallback strategies for perception failures
- Use multiple sensors for redundancy
- Plan for graceful degradation when perception fails

## Chapter Summary

The Isaac perception pipeline provides GPU-accelerated processing for real-time Physical AI applications. By leveraging Isaac ROS packages for object detection, segmentation, tracking, and pose estimation, robots can achieve sophisticated scene understanding capabilities. Proper configuration and integration of these components enable robust perception in complex environments.

## Exercises

1. Implement a complete perception pipeline with object detection and tracking.
2. Evaluate perception performance in different lighting conditions.
3. Integrate perception with navigation for obstacle avoidance.

## Next Steps

In the next chapter, we'll work on a project that integrates all the Isaac concepts learned in Module 3, creating a complete intelligent robot system with perception, navigation, and interaction capabilities.