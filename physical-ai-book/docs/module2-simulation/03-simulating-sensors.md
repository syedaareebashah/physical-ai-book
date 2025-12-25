---
sidebar_position: 3
---

# Simulating Sensors (LiDAR, Cameras, IMU)

## Chapter Objectives

By the end of this chapter, you will be able to:
- Configure and simulate various sensor types in Gazebo
- Understand the characteristics and limitations of simulated sensors
- Integrate simulated sensors with ROS 2 for Physical AI applications
- Validate sensor data accuracy and performance
- Create realistic sensor noise models for Physical AI systems

## Sensor Simulation Overview

In Physical AI, accurate sensor simulation is crucial for developing robust perception and control systems. Simulated sensors must replicate real-world characteristics including:
- Physical limitations and constraints
- Noise and uncertainty models
- Temporal characteristics
- Environmental dependencies

## Camera Simulation

### Basic Camera Configuration

```xml
<sensor name="camera" type="camera">
  <always_on>true</always_on>
  <update_rate>30</update_rate>
  <camera name="head_camera">
    <horizontal_fov>1.047</horizontal_fov> <!-- 60 degrees -->
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>10.0</far>
    </clip>
    <noise>
      <type>gaussian</type>
      <mean>0.0</mean>
      <stddev>0.007</stddev>
    </noise>
  </camera>
  <plugin filename="libgazebo_ros_camera.so" name="camera_controller">
    <frame_name>camera_frame</frame_name>
    <topic_name>/camera/image_raw</topic_name>
    <camera_info_topic_name>/camera/camera_info</camera_info_topic_name>
  </plugin>
</sensor>
```

### Depth Camera Configuration

```xml
<sensor name="depth_camera" type="depth">
  <always_on>true</always_on>
  <update_rate>30</update_rate>
  <camera name="depth_head">
    <horizontal_fov>1.047</horizontal_fov>
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>10.0</far>
    </clip>
  </camera>
  <plugin filename="libgazebo_ros_openni_kinect.so" name="depth_camera_controller">
    <baseline>0.2</baseline>
    <distortion_k1>0.0</distortion_k1>
    <distortion_k2>0.0</distortion_k2>
    <distortion_k3>0.0</distortion_k3>
    <distortion_t1>0.0</distortion_t1>
    <distortion_t2>0.0</distortion_t2>
    <point_cloud_cutoff>0.1</point_cloud_cutoff>
    <frame_name>depth_camera_frame</frame_name>
    <point_cloud_topic_name>/camera/depth/points</point_cloud_topic_name>
    <depth_image_topic_name>/camera/depth/image_raw</depth_image_topic_name>
    <depth_image_camera_info_topic_name>/camera/depth/camera_info</depth_image_camera_info_topic_name>
    <camera_name>depth_camera</camera_name>
  </plugin>
</sensor>
```

### Camera Processing Node

```python
# File: sensor_processing/camera_processor.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np

class CameraProcessor(Node):
    def __init__(self):
        super().__init__('camera_processor')

        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)

        self.info_subscription = self.create_subscription(
            CameraInfo, '/camera/camera_info', self.info_callback, 10)

        self.object_pub = self.create_publisher(Image, '/camera/processed', 10)

        self.bridge = CvBridge()
        self.camera_matrix = None
        self.distortion_coeffs = None

    def info_callback(self, msg):
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.distortion_coeffs = np.array(msg.d)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Process image (example: object detection)
            processed_image = self.detect_objects(cv_image)

            # Publish processed image
            processed_msg = self.bridge.cv2_to_imgmsg(processed_image, "bgr8")
            processed_msg.header = msg.header
            self.object_pub.publish(processed_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def detect_objects(self, image):
        # Simple color-based object detection for demonstration
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define range for red color
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)

        # Upper red range
        lower_red = np.array([170, 50, 50])
        upper_red = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red, upper_red)

        mask = mask1 + mask2

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding boxes
        result = image.copy()
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Filter small contours
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)

        return result

def main(args=None):
    rclpy.init(args=args)
    node = CameraProcessor()

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

## LiDAR Simulation

### 2D LiDAR Configuration

```xml
<sensor name="lidar_2d" type="ray">
  <always_on>true</always_on>
  <update_rate>10</update_rate>
  <ray>
    <scan>
      <horizontal>
        <samples>360</samples>
        <resolution>1.0</resolution>
        <min_angle>-3.14159</min_angle> <!-- -π -->
        <max_angle>3.14159</max_angle>   <!-- π -->
      </horizontal>
    </scan>
    <range>
      <min>0.1</min>
      <max>10.0</max>
      <resolution>0.01</resolution>
    </range>
  </ray>
  <plugin filename="libgazebo_ros_ray_sensor.so" name="lidar_2d_controller">
    <ros_topic>/scan</ros_topic>
    <frame_name>lidar_frame</frame_name>
    <update_rate>10</update_rate>
  </plugin>
</sensor>
```

### 3D LiDAR Configuration (Velodyne-style)

```xml
<sensor name="velodyne_VLP_16" type="ray">
  <always_on>true</always_on>
  <update_rate>10</update_rate>
  <ray>
    <scan>
      <horizontal>
        <samples>1800</samples>
        <resolution>1</resolution>
        <min_angle>-3.14159</min_angle>
        <max_angle>3.14159</max_angle>
      </horizontal>
      <vertical>
        <samples>16</samples>
        <resolution>1</resolution>
        <min_angle>-0.261799</min_angle> <!-- -15 degrees -->
        <max_angle>0.261799</max_angle>   <!-- 15 degrees -->
      </vertical>
    </scan>
    <range>
      <min>0.1</min>
      <max>100.0</max>
      <resolution>0.01</resolution>
    </range>
  </ray>
  <plugin filename="libgazebo_ros_velodyne_gpu_lidar.so" name="velodyne_VLP_16_controller">
    <topic_name>/velodyne_points</topic_name>
    <frame_name>velodyne</frame_name>
    <min_range>0.1</min_range>
    <max_range>100.0</max_range>
    <gaussian_noise>0.01</gaussian_noise>
  </plugin>
</sensor>
```

### LiDAR Processing Node

```python
# File: sensor_processing/lidar_processor.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2
from std_msgs.msg import String
import numpy as np
from sklearn.cluster import DBSCAN

class LiDARProcessor(Node):
    def __init__(self):
        super().__init__('lidar_processor')

        self.subscription = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)

        self.obstacle_pub = self.create_publisher(String, '/obstacle_detection', 10)

    def scan_callback(self, msg):
        # Convert laser scan to points
        points = self.laser_to_points(msg)

        # Detect obstacles
        obstacles = self.detect_obstacles(points)

        # Publish obstacle information
        obstacle_msg = String()
        obstacle_msg.data = f'Detected {len(obstacles)} obstacles'
        self.obstacle_pub.publish(obstacle_msg)

        self.get_logger().info(f'Processed scan: {len(obstacles)} obstacles detected')

    def laser_to_points(self, scan):
        points = []
        angle = scan.angle_min

        for range_val in scan.ranges:
            if scan.range_min <= range_val <= scan.range_max:
                x = range_val * np.cos(angle)
                y = range_val * np.sin(angle)
                points.append([x, y])
            angle += scan.angle_increment

        return np.array(points)

    def detect_obstacles(self, points):
        if len(points) < 2:
            return []

        # Use DBSCAN clustering to group nearby points
        clustering = DBSCAN(eps=0.3, min_samples=3).fit(points)
        labels = clustering.labels_

        # Group points by cluster (obstacle)
        obstacles = {}
        for i, label in enumerate(labels):
            if label != -1:  # -1 is noise in DBSCAN
                if label not in obstacles:
                    obstacles[label] = []
                obstacles[label].append(points[i])

        return obstacles

def main(args=None):
    rclpy.init(args=args)
    node = LiDARProcessor()

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

## IMU Simulation

### IMU Sensor Configuration

```xml
<sensor name="imu_sensor" type="imu">
  <always_on>true</always_on>
  <update_rate>100</update_rate>
  <imu>
    <angular_velocity>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
          <bias_mean>0.0000075</bias_mean>
          <bias_stddev>0.0000008</bias_stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
          <bias_mean>0.0000075</bias_mean>
          <bias_stddev>0.0000008</bias_stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
          <bias_mean>0.0000075</bias_mean>
          <bias_stddev>0.0000008</bias_stddev>
        </noise>
      </z>
    </angular_velocity>
    <linear_acceleration>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
          <bias_mean>0.1</bias_mean>
          <bias_stddev>0.001</bias_stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
          <bias_mean>0.1</bias_mean>
          <bias_stddev>0.001</bias_stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
          <bias_mean>0.1</bias_mean>
          <bias_stddev>0.001</bias_stddev>
        </noise>
      </z>
    </linear_acceleration>
  </imu>
  <plugin filename="libgazebo_ros_imu_sensor.so" name="imu_controller">
    <topic_name>/imu/data</topic_name>
    <body_name>base_link</body_name>
    <update_rate>100</update_rate>
    <gaussian_noise>0.01</gaussian_noise>
    <frame_name>imu_link</frame_name>
  </plugin>
</sensor>
```

### IMU Processing Node

```python
# File: sensor_processing/imu_processor.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32
import numpy as np
from tf_transformations import euler_from_quaternion

class IMUProcessor(Node):
    def __init__(self):
        super().__init__('imu_processor')

        self.subscription = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)

        self.orientation_pub = self.create_publisher(Float32, '/robot_orientation', 10)
        self.acceleration_pub = self.create_publisher(Float32, '/robot_acceleration', 10)

        self.prev_time = None
        self.prev_orientation = None

    def imu_callback(self, msg):
        # Extract orientation from quaternion
        orientation_q = msg.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)

        # Publish orientation (using yaw as primary orientation)
        orientation_msg = Float32()
        orientation_msg.data = yaw
        self.orientation_pub.publish(orientation_msg)

        # Calculate acceleration magnitude
        accel = msg.linear_acceleration
        accel_mag = np.sqrt(accel.x**2 + accel.y**2 + accel.z**2)

        # Publish acceleration magnitude
        accel_msg = Float32()
        accel_msg.data = accel_mag
        self.acceleration_pub.publish(accel_msg)

        # Calculate angular velocity magnitude
        ang_vel = msg.angular_velocity
        ang_vel_mag = np.sqrt(ang_vel.x**2 + ang_vel.y**2 + ang_vel.z**2)

        self.get_logger().info(f'Yaw: {yaw:.3f} rad, Accel: {accel_mag:.3f} m/s², Ang Vel: {ang_vel_mag:.3f} rad/s')

def main(args=None):
    rclpy.init(args=args)
    node = IMUProcessor()

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

## GPS Simulation

### GPS Sensor Configuration

```xml
<sensor name="gps_sensor" type="gps">
  <always_on>true</always_on>
  <update_rate>1</update_rate>
  <plugin filename="libgazebo_ros_gps.so" name="gps_controller">
    <topic_name>/gps/fix</topic_name>
    <frame_name>gps_link</frame_name>
    <update_rate>1</update_rate>
    <gaussian_noise>0.1</gaussian_noise>
  </plugin>
</sensor>
```

## Force/Torque Sensor Simulation

### Force/Torque Sensor in Joints

```xml
<joint name="wrist_ft_sensor" type="revolute">
  <parent link="forearm_link"/>
  <child link="wrist_link"/>
  <axis xyz="0 0 1"/>
  <limit lower="-3.14" upper="3.14" effort="100" velocity="1.0"/>

  <sensor name="wrist_force_torque" type="force_torque">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <force_torque>
      <frame>sensor</frame>
      <measure_direction>child_to_parent</measure_direction>
    </force_torque>
  </sensor>
</joint>

<plugin filename="libgazebo_ros_ft_sensor.so" name="ft_sensor_controller">
  <topic_name>/wrist/force_torque</topic_name>
  <joint_name>wrist_ft_sensor</joint_name>
</plugin>
```

## Multi-Sensor Fusion for Physical AI

### Sensor Fusion Node Example

```python
# File: sensor_processing/sensor_fusion.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu, Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import numpy as np
from collections import deque

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion')

        # Subscribers for different sensors
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)
        self.camera_sub = self.create_subscription(
            Image, '/camera/image_raw', self.camera_callback, 10)

        # Publisher for fused state
        self.state_pub = self.create_publisher(PoseStamped, '/fused_state', 10)

        # Buffer for sensor data
        self.scan_buffer = deque(maxlen=5)
        self.imu_buffer = deque(maxlen=10)

        self.bridge = CvBridge()
        self.last_pose = None

    def scan_callback(self, msg):
        self.scan_buffer.append(msg)
        self.update_fused_state()

    def imu_callback(self, msg):
        self.imu_buffer.append(msg)
        self.update_fused_state()

    def camera_callback(self, msg):
        # Process camera data if needed
        pass

    def update_fused_state(self):
        if not self.scan_buffer or not self.imu_buffer:
            return

        # Example: simple state estimation using sensor data
        latest_scan = self.scan_buffer[-1]
        latest_imu = self.imu_buffer[-1]

        # Calculate position from IMU integration (simplified)
        # In practice, you'd use more sophisticated fusion algorithms
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'

        # For now, just publish a placeholder
        pose_msg.pose.position.x = 0.0
        pose_msg.pose.position.y = 0.0
        pose_msg.pose.position.z = 0.0

        # Publish the fused state
        self.state_pub.publish(pose_msg)

def main(args=None):
    rclpy.init(args=args)
    node = SensorFusionNode()

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

## Physical AI Sensor Considerations

### Realism vs. Performance

When simulating sensors for Physical AI:

1. **Noise Modeling**: Include realistic noise patterns that match real sensors
2. **Latency**: Account for sensor processing and communication delays
3. **Field of View**: Match real sensor specifications
4. **Update Rates**: Use rates that match real hardware capabilities

### Environmental Effects

Sensors in Physical AI systems must account for environmental conditions:

```xml
<!-- Example: Camera with environmental effects -->
<sensor name="weather_camera" type="camera">
  <always_on>true</always_on>
  <update_rate>30</update_rate>
  <camera name="weather_cam">
    <horizontal_fov>1.047</horizontal_fov>
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>50.0</far>
    </clip>
    <noise>
      <type>gaussian</type>
      <mean>0.0</mean>
      <stddev>0.01</stddev>
    </noise>
  </camera>
  <!-- Add environmental effects plugin -->
  <plugin filename="libgazebo_ros_camera.so" name="weather_camera_controller">
    <frame_name>weather_camera_frame</frame_name>
    <topic_name>/weather_camera/image_raw</topic_name>
  </plugin>
</sensor>
```

## Validation and Calibration

### Comparing Simulated vs. Real Sensors

```python
# Example: Sensor validation node
class SensorValidator(Node):
    def __init__(self):
        super().__init__('sensor_validator')

        # Subscribe to both simulated and real sensors
        self.sim_scan_sub = self.create_subscription(
            LaserScan, '/sim_scan', self.sim_scan_callback, 10)
        self.real_scan_sub = self.create_subscription(
            LaserScan, '/real_scan', self.real_scan_callback, 10)

        self.error_pub = self.create_publisher(Float32, '/sensor_error', 10)

        self.scan_pairs = []

    def sim_scan_callback(self, msg):
        # Store simulated scan with timestamp
        pass

    def real_scan_callback(self, msg):
        # Store real scan and compare with closest simulated scan
        pass
```

## Best Practices for Physical AI Sensor Simulation

### Accuracy Considerations
- Use realistic noise models based on real sensor specifications
- Include sensor limitations (range, resolution, field of view)
- Model environmental effects (lighting, weather, etc.)

### Performance Optimization
- Use appropriate update rates for different sensor types
- Optimize sensor configurations based on application needs
- Balance realism with simulation performance

### Integration Testing
- Validate sensor data consistency across modalities
- Test sensor failure scenarios
- Ensure proper coordinate frame transformations

## Chapter Summary

Simulating sensors accurately is crucial for Physical AI development. This chapter covered configuring cameras, LiDAR, IMU, and other sensors in Gazebo, integrating them with ROS 2, and processing their data. Proper sensor simulation enables safe testing of perception and control algorithms before deployment to real hardware.

## Exercises

1. Create a simulation world with multiple sensor types and validate their ROS 2 topics.
2. Implement a simple sensor fusion algorithm combining camera and LiDAR data.
3. Add realistic noise models to your simulated sensors based on real hardware specifications.

## Next Steps

In the next chapter, we'll explore Unity for high-fidelity rendering and how it complements Gazebo for Physical AI applications requiring photorealistic simulation.