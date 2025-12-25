---
sidebar_position: 3
---

# Your First ROS 2 Node

## Chapter Objectives

By the end of this chapter, you will be able to:
- Create a ROS 2 package using colcon
- Write a simple publisher and subscriber node
- Build and run ROS 2 nodes
- Use ROS 2 command-line tools for debugging
- Apply basic Physical AI concepts in practice

## Creating a ROS 2 Package

Before creating nodes, we need to create a package. A package is a container for ROS 2 code and resources.

### Package Structure
```
my_robot_package/
├── CMakeLists.txt
├── package.xml
├── src/
│   └── my_node.cpp
├── scripts/
├── launch/
├── config/
└── test/
```

### Creating the Package

```bash
# Create workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src

# Create package (Python)
ros2 pkg create --build-type ament_python my_robot_package

# Or for C++
ros2 pkg create --build-type ament_cmake my_robot_package
```

## Simple Publisher Node

Let's create a simple publisher that simulates sensor data from a Physical AI system:

```python
# File: my_robot_package/my_robot_package/sensor_publisher.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
import random

class SensorPublisher(Node):
    def __init__(self):
        super().__init__('sensor_publisher')

        # Create publishers for different sensor types
        self.laser_pub = self.create_publisher(LaserScan, 'laser_scan', 10)
        self.temperature_pub = self.create_publisher(Float32, 'temperature', 10)

        # Create a timer to publish data at 10Hz
        self.timer = self.create_timer(0.1, self.publish_sensor_data)

        self.get_logger().info('Sensor Publisher Node Started')

    def publish_sensor_data(self):
        # Publish simulated laser scan data
        laser_msg = LaserScan()
        laser_msg.header.stamp = self.get_clock().now().to_msg()
        laser_msg.header.frame_id = 'laser_frame'
        laser_msg.angle_min = -1.57  # -90 degrees
        laser_msg.angle_max = 1.57   # 90 degrees
        laser_msg.angle_increment = 0.0174  # 1 degree
        laser_msg.range_min = 0.1
        laser_msg.range_max = 10.0
        laser_msg.ranges = [random.uniform(0.5, 5.0) for _ in range(181)]

        self.laser_pub.publish(laser_msg)

        # Publish simulated temperature data
        temp_msg = Float32()
        temp_msg.data = random.uniform(20.0, 30.0)
        self.temperature_pub.publish(temp_msg)

        self.get_logger().info(f'Published sensor data - Laser ranges: {len(laser_msg.ranges)}, Temperature: {temp_msg.data:.2f}°C')

def main(args=None):
    rclpy.init(args=args)
    node = SensorPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Simple Subscriber Node

Now let's create a subscriber that processes the sensor data:

```python
# File: my_robot_package/my_robot_package/sensor_subscriber.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
import math

class SensorSubscriber(Node):
    def __init__(self):
        super().__init__('sensor_subscriber')

        # Create subscribers
        self.laser_sub = self.create_subscription(
            LaserScan,
            'laser_scan',
            self.laser_callback,
            10)

        self.temperature_sub = self.create_subscription(
            Float32,
            'temperature',
            self.temperature_callback,
            10)

        self.get_logger().info('Sensor Subscriber Node Started')

    def laser_callback(self, msg):
        # Process laser scan data for obstacle detection
        min_distance = min(msg.ranges) if msg.ranges else float('inf')

        if min_distance < 1.0:  # Obstacle within 1 meter
            self.get_logger().warn(f'OBSTACLE DETECTED! Distance: {min_distance:.2f}m')
        else:
            self.get_logger().info(f'Clear path ahead. Distance: {min_distance:.2f}m')

    def temperature_callback(self, msg):
        # Process temperature data
        if msg.data > 28.0:
            self.get_logger().warn(f'High temperature detected: {msg.data:.2f}°C')
        else:
            self.get_logger().info(f'Normal temperature: {msg.data:.2f}°C')

def main(args=None):
    rclpy.init(args=args)
    node = SensorSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Setting up package.xml

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>my_robot_package</name>
  <version>0.0.0</version>
  <description>Package for Physical AI sensor processing</description>
  <maintainer email="user@example.com">user</maintainer>
  <license>Apache-2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>sensor_msgs</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

## Setting up setup.py

```python
from setuptools import setup
import os
from glob import glob

package_name = 'my_robot_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='Package for Physical AI sensor processing',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sensor_publisher = my_robot_package.sensor_publisher:main',
            'sensor_subscriber = my_robot_package.sensor_subscriber:main',
        ],
    },
)
```

## Building and Running the Package

### Build the Package
```bash
cd ~/ros2_ws
colcon build --packages-select my_robot_package
source install/setup.bash
```

### Run the Nodes

Terminal 1 - Publisher:
```bash
ros2 run my_robot_package sensor_publisher
```

Terminal 2 - Subscriber:
```bash
ros2 run my_robot_package sensor_subscriber
```

### Alternative: Launch Both Nodes Together
Create a launch file `launch/sensor_nodes.launch.py`:

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_robot_package',
            executable='sensor_publisher',
            name='sensor_publisher',
        ),
        Node(
            package='my_robot_package',
            executable='sensor_subscriber',
            name='sensor_subscriber',
        ),
    ])
```

Run with:
```bash
ros2 launch my_robot_package sensor_nodes.launch.py
```

## ROS 2 Command Line Tools

### Essential Debugging Commands

```bash
# List all topics
ros2 topic list

# Echo messages on a topic
ros2 topic echo /laser_scan

# List all nodes
ros2 node list

# Get information about a node
ros2 node info /sensor_publisher

# List all services
ros2 service list

# Check network connectivity
ros2 doctor
```

## Physical AI Context

### Real-world Application
This simple example demonstrates the foundation of Physical AI systems:
- **Sensor Integration**: Multiple sensor types feeding data
- **Real-time Processing**: Continuous data processing pipeline
- **Decision Making**: Obstacle detection and response
- **Modular Architecture**: Separate publisher and subscriber nodes

### Extending to Physical AI
To make this more realistic for Physical AI:
1. Replace random data with actual sensor drivers
2. Add more sophisticated processing algorithms
3. Include actuator control nodes
4. Implement safety and error handling

## Chapter Summary

You've created your first complete ROS 2 package with publisher and subscriber nodes. This demonstrates the fundamental communication patterns used in Physical AI systems. The modular approach allows for scalable and maintainable robot software.

## Exercises

1. Modify the sensor publisher to simulate a moving obstacle.
2. Add a third node that calculates and publishes the robot's velocity based on sensor data.
3. Implement a service that allows external nodes to request sensor calibration.

## Next Steps

In the next chapter, we'll dive deeper into Python integration with rclpy and explore more advanced ROS 2 features.