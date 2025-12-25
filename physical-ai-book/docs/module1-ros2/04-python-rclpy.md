---
sidebar_position: 4
---

# Python Integration with rclpy

## Chapter Objectives

By the end of this chapter, you will be able to:
- Use rclpy to create sophisticated ROS 2 nodes in Python
- Implement advanced communication patterns (actions, parameters)
- Handle asynchronous operations and callbacks
- Design robust Physical AI applications in Python
- Integrate with Python's rich ecosystem of AI libraries

## Understanding rclpy

rclpy is the Python client library for ROS 2, providing Python bindings for the ROS 2 client library (rcl). It allows Python developers to leverage ROS 2's capabilities while using Python's extensive libraries for AI, machine learning, and robotics.

### Core rclpy Concepts

```python
import rclpy
from rclpy.node import Node

class AdvancedNode(Node):
    def __init__(self):
        super().__init__('advanced_node')

        # Node initialization with parameters
        self.declare_parameter('sensor_frequency', 10.0)
        self.frequency = self.get_parameter('sensor_frequency').value

        # Logging
        self.get_logger().info(f'Node initialized with frequency: {self.frequency}Hz')
```

## Advanced Node Features

### Parameters

Parameters allow runtime configuration of nodes:

```python
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import ParameterDescriptor

class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters with descriptions
        self.declare_parameter(
            'max_velocity',
            1.0,
            ParameterDescriptor(description='Maximum robot velocity in m/s')
        )

        self.declare_parameter('robot_name', 'my_robot')

        # Set parameter callback
        self.add_on_set_parameters_callback(self.parameter_callback)

        # Create timer that uses parameters
        self.timer = self.create_timer(0.1, self.timer_callback)

    def parameter_callback(self, params):
        for param in params:
            if param.name == 'max_velocity' and param.value > 5.0:
                return SetParametersResult(successful=False, reason='Velocity too high')
        return SetParametersResult(successful=True)

    def timer_callback(self):
        max_vel = self.get_parameter('max_velocity').value
        robot_name = self.get_parameter('robot_name').value
        self.get_logger().info(f'{robot_name} max velocity: {max_vel} m/s')
```

### Actions

Actions provide goal-oriented communication for long-running tasks:

```python
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class FibonacciActionServer(Node):
    def __init__(self):
        super().__init__('fibonacci_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback)

    def goal_callback(self, goal_request):
        self.get_logger().info('Received goal request')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')

        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()

            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1])

            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().info(f'Feedback: {feedback_msg.sequence}')

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.sequence
        self.get_logger().info(f'Result: {result.sequence}')

        return result
```

## Asynchronous Programming with rclpy

### Async Nodes

```python
import rclpy
from rclpy.node import Node
import asyncio

class AsyncNode(Node):
    def __init__(self):
        super().__init__('async_node')

        # Create async timer
        self.async_timer = self.create_timer(
            1.0,
            self.async_timer_callback
        )

    def async_timer_callback(self):
        # Schedule async task
        self.executor.create_task(self.background_task())

    async def background_task(self):
        # Simulate async work
        await asyncio.sleep(0.5)
        self.get_logger().info('Async task completed')
```

### Async Service Implementation

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts
import asyncio

class AsyncServiceNode(Node):
    def __init__(self):
        super().__init__('async_service')
        self.srv = self.create_service(
            AddTwoInts,
            'async_add_two_ints',
            self.handle_async_request
        )

    async def handle_async_request(self, request, response):
        # Simulate async processing
        await asyncio.sleep(0.1)
        response.sum = request.a + request.b
        return response
```

## Physical AI Integration with Python Libraries

### Computer Vision Integration

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        self.subscription = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            10)

        self.publisher = self.create_publisher(Image, 'camera/image_processed', 10)
        self.bridge = CvBridge()

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Apply computer vision processing
            processed_image = self.process_image(cv_image)

            # Convert back to ROS Image message
            processed_msg = self.bridge.cv2_to_imgmsg(processed_image, "bgr8")
            processed_msg.header = msg.header

            self.publisher.publish(processed_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def process_image(self, image):
        # Apply edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Combine with original
        result = image.copy()
        result[edges != 0] = [0, 255, 0]  # Highlight edges in green

        return result
```

### Machine Learning Integration

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from sklearn.cluster import DBSCAN
import numpy as np

class MLProcessingNode(Node):
    def __init__(self):
        super().__init__('ml_processing_node')
        self.subscription = self.create_subscription(
            LaserScan,
            'laser_scan',
            self.scan_callback,
            10)

        self.publisher = self.create_publisher(String, 'object_detection', 10)

    def scan_callback(self, msg):
        # Convert laser scan to points
        points = self.laser_scan_to_points(msg)

        if len(points) > 0:
            # Apply clustering to detect objects
            clusters = self.detect_objects(points)

            # Publish results
            result_msg = String()
            result_msg.data = f'Detected {len(clusters)} objects'
            self.publisher.publish(result_msg)

    def laser_scan_to_points(self, scan):
        points = []
        angle = scan.angle_min

        for range_val in scan.ranges:
            if scan.range_min <= range_val <= scan.range_max:
                x = range_val * np.cos(angle)
                y = range_val * np.sin(angle)
                points.append([x, y])
            angle += scan.angle_increment

        return np.array(points)

    def detect_objects(self, points):
        if len(points) < 2:
            return []

        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=0.5, min_samples=3).fit(points)
        labels = clustering.labels_

        # Group points by cluster
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(points[i])

        # Filter out noise (label -1)
        return {k: v for k, v in clusters.items() if k != -1}
```

## Best Practices for Physical AI Applications

### Error Handling and Safety

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
import traceback

class SafeNode(Node):
    def __init__(self):
        super().__init__('safe_node')

        # Use appropriate QoS for safety-critical topics
        qos_profile = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE
        )

        self.subscription = self.create_subscription(
            String, 'critical_topic', self.safe_callback, qos_profile
        )

    def safe_callback(self, msg):
        try:
            # Process message
            result = self.process_message(msg)
            # Validate result
            if self.validate_result(result):
                self.publish_result(result)
            else:
                self.get_logger().error('Invalid result detected')
        except Exception as e:
            self.get_logger().error(f'Error in callback: {e}')
            traceback.print_exc()
            # Implement safety measures
            self.emergency_stop()

    def validate_result(self, result):
        # Add validation logic
        return True

    def emergency_stop(self):
        # Implement emergency procedures
        self.get_logger().warn('Emergency stop activated')
```

### Performance Optimization

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import threading
from queue import Queue

class OptimizedNode(Node):
    def __init__(self):
        super().__init__('optimized_node')

        # Use threading for CPU-intensive tasks
        self.processing_queue = Queue(maxsize=5)
        self.result_queue = Queue()

        self.subscription = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, 10
        )

        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_images)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def image_callback(self, msg):
        # Add to processing queue if not full
        if not self.processing_queue.full():
            self.processing_queue.put(msg)

    def process_images(self):
        while rclpy.ok():
            try:
                msg = self.processing_queue.get(timeout=0.1)
                # Process image in separate thread
                result = self.heavy_computation(msg)
                self.result_queue.put(result)
            except:
                continue  # Timeout is normal

    def heavy_computation(self, msg):
        # Perform heavy computation
        pass
```

## Chapter Summary

Python integration with rclpy enables powerful Physical AI applications by combining ROS 2's distributed computing capabilities with Python's rich ecosystem of AI and machine learning libraries. Advanced features like parameters, actions, and asynchronous programming provide the flexibility needed for complex robotic systems.

## Exercises

1. Create a node that uses TensorFlow or PyTorch for real-time object detection from camera feeds.
2. Implement an action server that performs path planning with feedback.
3. Design a parameter server that allows runtime configuration of robot behaviors.

## Next Steps

In the next chapter, we'll explore URDF (Unified Robot Description Format) for describing humanoid robots in Physical AI systems.