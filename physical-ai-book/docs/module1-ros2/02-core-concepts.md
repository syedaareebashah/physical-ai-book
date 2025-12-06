---
sidebar_position: 2
---

# Core Concepts: Nodes, Topics, Services

## Chapter Objectives

By the end of this chapter, you will be able to:
- Create and run ROS 2 nodes in both Python and C++
- Implement topic-based communication between nodes
- Design service-based request/reply interactions
- Understand the publish/subscribe and client/server patterns
- Apply these concepts to Physical AI scenarios

## Understanding Nodes

A node is an executable that uses ROS 2 to communicate with other nodes. Nodes are the fundamental building blocks of ROS 2 applications.

### Node Lifecycle

Nodes in ROS 2 follow a specific lifecycle:
1. **Unconfigured**: Initial state after creation
2. **Inactive**: Configured but not active
3. **Active**: Running and processing callbacks
4. **Finalized**: Cleaned up and ready for destruction

### Creating a Node in Python

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class TalkerNode(Node):
    def __init__(self):
        super().__init__('talker')
        self.publisher = self.create_publisher(String, 'chatter', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    talker = TalkerNode()
    rclpy.spin(talker)
    talker.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Topics and Publish/Subscribe Pattern

Topics enable asynchronous, decoupled communication between nodes using a publish/subscribe pattern.

### Topic Communication Characteristics
- **Many-to-many**: Multiple publishers and subscribers can use the same topic
- **Asynchronous**: Publishers and subscribers don't need to run simultaneously
- **Typed**: Messages follow specific data types defined in `.msg` files
- **Buffered**: Messages are queued when publishers outpace subscribers

### Quality of Service (QoS) Settings

QoS settings allow fine-tuning communication behavior:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

# For real-time sensor data
sensor_qos = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE
)

# For critical control commands
control_qos = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL
)
```

## Services and Client/Server Pattern

Services provide synchronous request/reply communication for actions requiring responses.

### Service Implementation

```python
# Service definition (in srv/AddTwoInts.srv)
# int64 a
# int64 b
# ---
# int64 sum

# Service server
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class AddTwoIntsService(Node):
    def __init__(self):
        super().__init__('add_two_ints_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Returning {request.a} + {request.b} = {response.sum}')
        return response

# Service client
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class AddTwoIntsClient(Node):
    def __init__(self):
        super().__init__('add_two_ints_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()
```

## Physical AI Applications

### Sensor Data Pipeline
In Physical AI systems, nodes often form a pipeline:
1. **Sensor Nodes**: Publish raw sensor data (camera images, LiDAR scans)
2. **Processing Nodes**: Subscribe to sensor data, perform perception tasks
3. **Decision Nodes**: Use processed information for planning and control

### Example: Camera Data Pipeline
```python
# Camera driver node (publisher)
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        self.publisher = self.create_publisher(Image, 'camera/image_raw', 10)
        self.timer = self.create_timer(0.1, self.capture_image)  # 10Hz
        self.cap = cv2.VideoCapture(0)

    def capture_image(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert OpenCV image to ROS Image message
            # (simplified - actual implementation requires cv2_to_imgmsg)
            pass

# Object detection node (subscriber)
class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection')
        self.subscription = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            10)
        self.subscription  # prevent unused variable warning

    def image_callback(self, msg):
        # Process image for object detection
        # Publish results to another topic
        pass
```

## Best Practices for Physical AI

### Node Design Principles
1. **Single Responsibility**: Each node should perform one primary function
2. **Modularity**: Design nodes to be reusable across different robots
3. **Robustness**: Handle errors gracefully and provide meaningful logging
4. **Performance**: Optimize for real-time constraints in Physical AI systems

### Communication Patterns
1. **Use appropriate QoS settings** for different types of data
2. **Minimize message size** for bandwidth-constrained scenarios
3. **Implement proper error handling** for network interruptions
4. **Consider message frequency** to avoid overwhelming the system

## Chapter Summary

Nodes, topics, and services form the core communication infrastructure of ROS 2. Understanding these concepts is essential for building distributed Physical AI systems. The publish/subscribe pattern enables decoupled sensor processing pipelines, while services provide synchronous interfaces for critical operations.

## Exercises

1. Create a simple ROS 2 node that publishes random sensor data and another that subscribes to it.
2. Design a service that calculates the distance between two 3D points for robot navigation.
3. Explain when you would use topics vs services in a Physical AI application.

## Next Steps

In the next chapter, we'll create our first complete ROS 2 node and run it in a practical example.