---
sidebar_position: 6
---

# Project: Warehouse Robot with Isaac

## Project Objectives

By completing this project, you will:
- Integrate Isaac Sim, Visual SLAM, Navigation2, and perception pipeline
- Create a complete intelligent robot system for warehouse operations
- Implement perception-aware navigation with object detection and tracking
- Build a system that can autonomously navigate, detect inventory, and report status
- Deploy and test the complete system in Isaac Sim

## Project Overview

In this project, we'll build an intelligent warehouse robot that operates in a simulated warehouse environment using NVIDIA Isaac. The robot will:
1. Navigate autonomously through warehouse aisles
2. Detect and identify inventory items using perception
3. Track inventory movement and location
4. Report inventory status to a central system
5. Handle dynamic obstacles and re-plan routes as needed

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Isaac Sim     │    │   Isaac ROS     │    │   Navigation    │
│   (Warehouse    │───▶│   Perception    │───▶│   (Nav2)        │
│   Environment)  │    │   (GPU)         │    │   (Path Plan,  │
└─────────────────┘    └─────────────────┘    │   Control)      │
                                              └─────────────────┘
         ▲                       ▲                       ▲
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    └─────────────────┘
│   Robot Model   │    │   SLAM &        │    │   Inventory     │
│   (URDF/USD)    │    │   Localization  │    │   Management    │
│                 │    │   (Isaac Visual │    │   (ROS Nodes)   │
└─────────────────┘    │   SLAM)         │    │                 │
                       └─────────────────┘    └─────────────────┘
```

## Implementation Steps

### Step 1: Create the Project Package

```bash
# Create project workspace
mkdir -p ~/warehouse_robot_ws/src
cd ~/warehouse_robot_ws/src

# Create warehouse robot package
ros2 pkg create --build-type ament_python warehouse_robot
cd warehouse_robot
```

### Step 2: Install Dependencies

```bash
# Install Isaac ROS packages
sudo apt update
sudo apt install ros-humble-isaac-ros-* ros-humble-navigation2 ros-humble-nav2-bringup

# Install additional dependencies
pip3 install opencv-python numpy scipy
```

### Step 3: Robot URDF Model

Create a warehouse robot model with sensors:

```xml
<!-- File: warehouse_robot/urdf/warehouse_robot.urdf -->
<?xml version="1.0"?>
<robot name="warehouse_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <mesh filename="package://warehouse_robot/meshes/robot_base.dae"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.3" radius="0.4"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="50"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Wheels -->
  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel"/>
    <origin xyz="0 0.4 -0.15" rpy="1.57 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <link name="left_wheel">
    <visual>
      <geometry>
        <cylinder length="0.05" radius="0.15"/>
      </geometry>
      <material name="black">
        <color rgba="0.1 0.1 0.1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.05" radius="0.15"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>

  <joint name="right_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="right_wheel"/>
    <origin xyz="0 -0.4 -0.15" rpy="1.57 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <link name="right_wheel">
    <visual>
      <geometry>
        <cylinder length="0.05" radius="0.15"/>
      </geometry>
      <material name="black">
        <color rgba="0.1 0.1 0.1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.05" radius="0.15"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Camera for perception -->
  <joint name="camera_joint" type="fixed">
    <parent link="base_link"/>
    <child link="camera_link"/>
    <origin xyz="0.3 0 0.2" rpy="0 0 0"/>
  </joint>

  <link name="camera_link">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
      <material name="red">
        <color rgba="0.8 0.2 0.2 1"/>
      </material>
    </visual>
  </link>

  <!-- IMU for SLAM -->
  <joint name="imu_joint" type="fixed">
    <parent link="base_link"/>
    <child link="imu_link"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
  </joint>

  <link name="imu_link"/>

  <!-- LiDAR for navigation -->
  <joint name="lidar_joint" type="fixed">
    <parent link="base_link"/>
    <child link="lidar_link"/>
    <origin xyz="0.2 0 0.3" rpy="0 0 0"/>
  </joint>

  <link name="lidar_link"/>

  <!-- Gazebo plugins -->
  <gazebo>
    <plugin filename="libgazebo_ros2_control.so" name="gazebo_ros2_control">
      <parameters>$(find warehouse_robot)/config/warehouse_robot_controllers.yaml</parameters>
    </plugin>
  </gazebo>

  <!-- Camera sensor -->
  <gazebo reference="camera_link">
    <sensor name="camera" type="camera">
      <always_on>true</always_on>
      <update_rate>30</update_rate>
      <camera name="warehouse_camera">
        <horizontal_fov>1.047</horizontal_fov>
        <image>
          <width>640</width>
          <height>480</height>
        </image>
        <clip>
          <near>0.1</near>
          <far>10.0</far>
        </clip>
      </camera>
      <plugin filename="libgazebo_ros_camera.so" name="camera_controller">
        <frame_name>camera_link</frame_name>
        <topic_name>/camera/image_raw</topic_name>
      </plugin>
    </sensor>
  </gazebo>

  <!-- IMU sensor -->
  <gazebo reference="imu_link">
    <sensor name="imu_sensor" type="imu">
      <always_on>true</always_on>
      <update_rate>100</update_rate>
      <plugin filename="libgazebo_ros_imu_sensor.so" name="imu_controller">
        <topic_name>/imu/data</topic_name>
        <body_name>imu_link</body_name>
        <update_rate>100</update_rate>
        <frame_name>imu_link</frame_name>
      </plugin>
    </sensor>
  </gazebo>

  <!-- LiDAR sensor -->
  <gazebo reference="lidar_link">
    <sensor name="lidar" type="ray">
      <always_on>true</always_on>
      <update_rate>10</update_rate>
      <ray>
        <scan>
          <horizontal>
            <samples>360</samples>
            <resolution>1</resolution>
            <min_angle>-3.14159</min_angle>
            <max_angle>3.14159</max_angle>
          </horizontal>
        </scan>
        <range>
          <min>0.1</min>
          <max>10.0</max>
          <resolution>0.01</resolution>
        </range>
      </ray>
      <plugin filename="libgazebo_ros_ray_sensor.so" name="lidar_controller">
        <ros_topic>/scan</ros_topic>
        <frame_name>lidar_link</frame_name>
      </plugin>
    </sensor>
  </gazebo>

</robot>
```

### Step 4: Controller Configuration

```yaml
# File: warehouse_robot/config/warehouse_robot_controllers.yaml
controller_manager:
  ros__parameters:
    update_rate: 100  # Hz

    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster

    diff_drive_controller:
      type: diff_drive_controller/DiffDriveController

diff_drive_controller:
  ros__parameters:
    left_wheel_names: ["left_wheel_joint"]
    right_wheel_names: ["right_wheel_joint"]

    wheel_separation: 0.8
    wheel_radius: 0.15

    use_stamped_vel: false

    # Publish rate
    publish_rate: 50.0
    odom_frame_id: odom
    base_frame_id: base_link
    pose_covariance_diagonal: [0.001, 0.001, 0.001, 0.001, 0.001, 0.01]
    twist_covariance_diagonal: [0.001, 0.001, 0.001, 0.001, 0.001, 0.01]
```

### Step 5: Isaac-Specific Configuration

```yaml
# File: warehouse_robot/config/isaac_warehouse_params.yaml
# Isaac Image Pipeline
isaac_ros_image_pipeline:
  ros__parameters:
    input_width: 640
    input_height: 480
    input_format: "rgb8"
    enable_rectification: true
    cuda_device: 0

# Isaac DetectNet
isaac_ros_detectnet:
  ros__parameters:
    model_name: "ssd_mobilenet_v2_coco"
    confidence_threshold: 0.7
    input_width: 640
    input_height: 480
    cuda_device: 0

# Isaac Visual SLAM
isaac_ros_visual_slam:
  ros__parameters:
    enable_imu_fusion: true
    map_frame: "map"
    tracking_frame: "camera_link"
    publish_odom_tf: true
    enable_localization: false
    enable_mapping: true

# Navigation2 components
amcl:
  ros__parameters:
    use_sim_time: True
    alpha1: 0.2
    alpha2: 0.2
    base_frame_id: "base_footprint"
    global_frame_id: "map"
    laser_topic: "/scan"
    max_particles: 2000
    min_particles: 500
    odom_frame_id: "odom"
    update_min_d: 0.25
    update_min_a: 0.2

amcl_map_client:
  ros__parameters:
    use_sim_time: True

amcl_rclcpp_node:
  ros__parameters:
    use_sim_time: True

bt_navigator:
  ros__parameters:
    use_sim_time: True
    global_frame: map
    robot_base_frame: base_footprint
    odom_topic: /odom
    bt_loop_duration: 10
    default_server_timeout: 20
    plugin_lib_names:
    - nav2_compute_path_to_pose_action_bt_node
    - nav2_follow_path_action_bt_node
    - nav2_back_up_action_bt_node
    - nav2_spin_action_bt_node
    - nav2_wait_action_bt_node
    - nav2_clear_costmap_service_bt_node
    - nav2_is_stuck_condition_bt_node
    - nav2_goal_reached_condition_bt_node
    - nav2_initial_pose_received_condition_bt_node
    - nav2_reinitialize_global_localization_service_bt_node
    - nav2_rate_controller_bt_node
    - nav2_distance_controller_bt_node
    - nav2_speed_controller_bt_node
    - nav2_truncate_path_action_bt_node
    - nav2_goal_updater_node_bt_node
    - nav2_recovery_node_bt_node
    - nav2_pipeline_sequence_bt_node
    - nav2_round_robin_node_bt_node
    - nav2_transform_available_condition_bt_node
    - nav2_time_expired_condition_bt_node
    - nav2_path_expiring_timer_condition
    - nav2_distance_traveled_condition_bt_node
    - nav2_single_trigger_bt_node
    - nav2_is_battery_low_condition_bt_node
    - nav2_navigate_through_poses_action_bt_node
    - nav2_navigate_to_pose_action_bt_node
    - nav2_remove_passed_goals_action_bt_node
    - nav2_planner_selector_bt_node
    - nav2_controller_selector_bt_node
    - nav2_goal_checker_selector_bt_node

bt_navigator_rclcpp_node:
  ros__parameters:
    use_sim_time: True

controller_server:
  ros__parameters:
    use_sim_time: True
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.5
    min_theta_velocity_threshold: 0.001
    progress_checker_plugin: "progress_checker"
    goal_checker_plugin: "goal_checker"
    controller_plugins: ["FollowPath"]

    progress_checker:
      plugin: "nav2_controller::SimpleProgressChecker"
      required_movement_radius: 0.5
      movement_time_allowance: 10.0

    goal_checker:
      plugin: "nav2_controller::SimpleGoalChecker"
      xy_goal_tolerance: 0.25
      yaw_goal_tolerance: 0.25
      stateful: True

    FollowPath:
      plugin: "nav2_rotation_shim_controller::RotationShimController"
      progress_checker_plugin: "progress_checker"
      goal_checker_plugin: "goal_checker"
      required_movement_radius: 0.5
      movement_time_allowance: 10.0
      velocity_deadband: 0.05
      simulate_ahead_time: 1.0
      max_rotational_vel: 1.0
      min_rotational_vel: 0.4
      rotational_acc_lim: 3.2

local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 5.0
      publish_frequency: 2.0
      global_frame: odom
      robot_base_frame: base_footprint
      use_sim_time: True
      rolling_window: true
      width: 3
      height: 3
      resolution: 0.05
      robot_radius: 0.4
      plugins: ["voxel_layer", "inflation_layer"]
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
      voxel_layer:
        plugin: "nav2_costmap_2d::VoxelLayer"
        enabled: True
        publish_voxel_map: True
        origin_z: 0.0
        z_resolution: 0.05
        z_voxels: 16
        max_obstacle_height: 2.0
        mark_threshold: 0
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0
      always_send_full_costmap: True
  local_costmap_client:
    ros__parameters:
      use_sim_time: True
  local_costmap_rclcpp_node:
    ros__parameters:
      use_sim_time: True

global_costmap:
  global_costmap:
    ros__parameters:
      update_frequency: 1.0
      publish_frequency: 1.0
      global_frame: map
      robot_base_frame: base_footprint
      use_sim_time: True
      robot_radius: 0.4
      resolution: 0.05
      track_unknown_space: true
      plugins: ["static_layer", "obstacle_layer", "inflation_layer"]
      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: True
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0
      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_subscribe_transient_local: True
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
      always_send_full_costmap: True
  global_costmap_client:
    ros__parameters:
      use_sim_time: True
  global_costmap_rclcpp_node:
    ros__parameters:
      use_sim_time: True

map_server:
  ros__parameters:
    use_sim_time: True
    yaml_filename: "warehouse_map.yaml"

map_saver:
  ros__parameters:
    use_sim_time: True
    save_map_timeout: 5.0
    free_thresh_default: 0.25
    occupied_thresh_default: 0.65

planner_server:
  ros__parameters:
    expected_planner_frequency: 20.0
    use_sim_time: True
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_navfn_planner/NavfnPlanner"
      tolerance: 0.5
      use_astar: false
      allow_unknown: true

planner_server_rclcpp_node:
  ros__parameters:
    use_sim_time: True

recoveries_server:
  ros__parameters:
    costmap_topic: local_costmap/costmap_raw
    footprint_topic: local_costmap/published_footprint
    cycle_frequency: 10.0
    recovery_plugins: ["spin", "backup", "wait"]
    spin:
      plugin: "nav2_recoveries/Spin"
      sim_frequency: 10
      angle_instep_thresh: 0.05
      angle_tolerance: 0.1
      max_rotational_vel: 1.0
      min_rotational_vel: 0.4
    backup:
      plugin: "nav2_recoveries/BackUp"
      sim_frequency: 10
      distance: 0.15
      forward_sampling_distance: 0.05
      move_time_allowance: 10.0
      max_translation_vel: 0.25
    wait:
      plugin: "nav2_recoveries/Wait"
      sim_frequency: 10
      backup_distance: 0.15
      time_allowance: 5.0

robot_state_publisher:
  ros__parameters:
    use_sim_time: True
```

### Step 6: Warehouse Management Node

```python
# File: warehouse_robot/warehouse_robot/warehouse_manager.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import Image, LaserScan
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import String, Float32
from tf2_ros import Buffer, TransformListener
import numpy as np
from collections import defaultdict
import json
from datetime import datetime

class WarehouseManager(Node):
    def __init__(self):
        super().__init__('warehouse_manager')

        # TF buffer for transformations
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscribers
        self.detections_sub = self.create_subscription(
            Detection2DArray,
            '/object_detections',
            self.detections_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/current_pose',
            self.pose_callback,
            10
        )

        # Publishers
        self.inventory_pub = self.create_publisher(
            String,
            '/warehouse/inventory_report',
            10
        )

        self.nav_goal_pub = self.create_publisher(
            PoseStamped,
            '/goal_pose',
            10
        )

        self.status_pub = self.create_publisher(
            String,
            '/warehouse/status',
            10
        )

        # State
        self.current_pose = None
        self.detected_objects = defaultdict(list)
        self.navigation_goals = []
        self.inventory_map = {}  # Object ID -> location
        self.scan_data = None

        # Warehouse layout
        self.warehouse_zones = {
            'receiving': (-5.0, 0.0),
            'storage_a': (5.0, 5.0),
            'storage_b': (5.0, -5.0),
            'shipping': (-5.0, -10.0)
        }

        # Navigation parameters
        self.safety_distance = 0.5
        self.inventory_check_interval = 30  # seconds
        self.last_inventory_check = self.get_clock().now()

        # Timer for periodic tasks
        self.inventory_timer = self.create_timer(
            self.inventory_check_interval,
            self.check_inventory
        )

        self.get_logger().info('Warehouse Manager Node Started')

    def pose_callback(self, msg):
        """Update current robot pose"""
        self.current_pose = msg.pose

    def scan_callback(self, msg):
        """Process laser scan for navigation"""
        self.scan_data = msg

        # Check for obstacles in path
        min_distance = min([r for r in msg.ranges if msg.range_min < r < msg.range_max])
        if min_distance < self.safety_distance:
            self.get_logger().warn(f'Obstacle detected at {min_distance:.2f}m')

    def detections_callback(self, msg):
        """Process object detections for inventory management"""
        current_time = self.get_clock().now()

        for detection in msg.detections:
            if detection.results:
                class_name = detection.results[0].hypothesis.class_id
                confidence = detection.results[0].hypothesis.score

                # Only process high-confidence detections
                if confidence > 0.7:
                    # Get object position in map frame
                    obj_position = self.get_object_position(detection, current_time)

                    if obj_position is not None:
                        # Store object detection
                        obj_id = f"{class_name}_{len(self.detected_objects[class_name])}"
                        self.detected_objects[class_name].append({
                            'id': obj_id,
                            'position': obj_position,
                            'confidence': confidence,
                            'timestamp': current_time
                        })

                        # Update inventory map
                        self.inventory_map[obj_id] = {
                            'type': class_name,
                            'position': obj_position,
                            'last_seen': current_time
                        }

                        self.get_logger().info(f'Detected {class_name} at {obj_position}')

    def get_object_position(self, detection, current_time):
        """Get object position in map frame"""
        if self.current_pose is None:
            return None

        # Calculate object position based on detection and robot pose
        # This is a simplified calculation
        # In practice, you'd use camera calibration and 3D reconstruction
        bbox = detection.bbox
        center_x = bbox.center.x
        center_y = bbox.center.y

        # Convert image coordinates to world coordinates (simplified)
        # This would require proper camera calibration and TF transforms
        obj_x = self.current_pose.position.x + (center_x - 320) * 0.01  # Rough conversion
        obj_y = self.current_pose.position.y + (center_y - 240) * 0.01  # Rough conversion

        return [obj_x, obj_y, 0.0]  # Assume z=0 for floor objects

    def check_inventory(self):
        """Periodically check inventory and generate reports"""
        current_time = self.get_clock().now()

        # Create inventory report
        inventory_report = {
            'timestamp': current_time.nanoseconds / 1e9,
            'robot_position': [0, 0, 0] if self.current_pose is None else [
                self.current_pose.position.x,
                self.current_pose.position.y,
                self.current_pose.position.z
            ],
            'objects_detected': {},
            'zones': self.warehouse_zones
        }

        # Count objects by type
        for obj_type, objects in self.detected_objects.items():
            active_objects = [
                obj for obj in objects
                if (current_time - obj['timestamp']).nanoseconds / 1e9 < 60  # Last 60 seconds
            ]
            inventory_report['objects_detected'][obj_type] = len(active_objects)

        # Publish inventory report
        report_msg = String()
        report_msg.data = json.dumps(inventory_report, indent=2)
        self.inventory_pub.publish(report_msg)

        self.get_logger().info(f'Inventory report: {inventory_report}')

    def navigate_to_zone(self, zone_name):
        """Navigate to a specific warehouse zone"""
        if zone_name in self.warehouse_zones:
            goal = PoseStamped()
            goal.header.stamp = self.get_clock().now().to_msg()
            goal.header.frame_id = 'map'
            x, y = self.warehouse_zones[zone_name]
            goal.pose.position.x = x
            goal.pose.position.y = y
            goal.pose.position.z = 0.0
            goal.pose.orientation.w = 1.0

            self.nav_goal_pub.publish(goal)
            self.get_logger().info(f'Navigating to {zone_name} at ({x}, {y})')

    def get_zone_status(self):
        """Get status of different warehouse zones"""
        status = {
            'zones': {},
            'total_objects': len(self.inventory_map),
            'last_update': self.get_clock().now().nanoseconds / 1e9
        }

        for zone_name, (zone_x, zone_y) in self.warehouse_zones.items():
            # Count objects near this zone
            zone_objects = 0
            for obj_info in self.inventory_map.values():
                obj_pos = obj_info['position']
                distance = np.sqrt((obj_pos[0] - zone_x)**2 + (obj_pos[1] - zone_y)**2)
                if distance < 2.0:  # Within 2m of zone
                    zone_objects += 1

            status['zones'][zone_name] = {
                'position': [zone_x, zone_y],
                'object_count': zone_objects
            }

        return status

def main(args=None):
    rclpy.init(args=args)
    manager = WarehouseManager()

    try:
        rclpy.spin(manager)
    except KeyboardInterrupt:
        pass
    finally:
        manager.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 7: Main Launch File

```python
# File: warehouse_robot/launch/warehouse_robot.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    params_file = LaunchConfiguration('params_file', default='isaac_warehouse_params.yaml')

    # Get package paths
    pkg_nav2_bringup = FindPackageShare('nav2_bringup')
    pkg_isaac_perception = FindPackageShare('isaac_ros_detectnet')
    pkg_warehouse = FindPackageShare('warehouse_robot')

    # Robot state publisher
    robot_state_publisher = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('warehouse_robot'),
                'launch',
                'robot_state_publisher.launch.py'
            ])
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time
        }.items()
    )

    # Isaac perception pipeline
    perception = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('isaac_ros_image_pipeline'),
                'launch',
                'image_pipeline.launch.py'
            ])
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time
        }.items()
    )

    # Isaac object detection
    detectnet = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('isaac_ros_detectnet'),
                'launch',
                'detectnet.launch.py'
            ])
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time
        }.items()
    )

    # Isaac Visual SLAM
    visual_slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('isaac_ros_visual_slam'),
                'launch',
                'visual_slam.launch.py'
            ])
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time
        }.items()
    )

    # Navigation2
    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                pkg_nav2_bringup,
                'launch',
                'navigation_launch.py'
            ])
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'params_file': PathJoinSubstitution([
                pkg_warehouse,
                'config',
                params_file
            ])
        }.items()
    )

    # Warehouse manager
    warehouse_manager = Node(
        package='warehouse_robot',
        executable='warehouse_manager',
        parameters=[PathJoinSubstitution([
            pkg_warehouse,
            'config',
            params_file
        ])],
        remappings=[
            ('/object_detections', '/detectnet/detections')
        ]
    )

    # RViz
    rviz_config_file = PathJoinSubstitution([
        pkg_nav2_bringup,
        'rviz',
        'nav2_default_view.rviz'
    ])

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Isaac Sim (if running separately)
    # This would typically be launched from Omniverse Launcher

    # Define startup order
    startup_sequence = []

    # Start robot state publisher
    startup_sequence.append(robot_state_publisher)

    # Then perception
    startup_sequence.append(RegisterEventHandler(
        OnProcessExit(
            target_action=robot_state_publisher,
            on_exit=[perception]
        )
    ))

    # Then detection
    startup_sequence.append(RegisterEventHandler(
        OnProcessExit(
            target_action=perception,
            on_exit=[detectnet]
        )
    ))

    # Then SLAM
    startup_sequence.append(RegisterEventHandler(
        OnProcessExit(
            target_action=detectnet,
            on_exit=[visual_slam]
        )
    ))

    # Then navigation
    startup_sequence.append(RegisterEventHandler(
        OnProcessExit(
            target_action=visual_slam,
            on_exit=[navigation]
        )
    ))

    # Then warehouse manager
    startup_sequence.append(RegisterEventHandler(
        OnProcessExit(
            target_action=navigation,
            on_exit=[warehouse_manager]
        )
    ))

    # Finally RViz
    startup_sequence.append(RegisterEventHandler(
        OnProcessExit(
            target_action=warehouse_manager,
            on_exit=[rviz]
        )
    ))

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation time'
        ),
        DeclareLaunchArgument(
            'params_file',
            default_value='isaac_warehouse_params.yaml',
            description='Parameters file'
        )
    ] + startup_sequence)
```

### Step 8: Warehouse World File

```xml
<!-- File: warehouse_robot/worlds/warehouse_world.sdf -->
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="warehouse">
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>

    <light name="sun" type="directional">
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.4 0.2 -0.9</direction>
    </light>

    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Warehouse structure -->
    <!-- Outer walls -->
    <model name="north_wall">
      <pose>0 10 1 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry><box><size>20 0.2 2</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>20 0.2 2</size></box></geometry>
          <material><ambient>0.5 0.5 0.5 1</ambient></material>
        </visual>
      </link>
    </model>

    <model name="south_wall">
      <pose>0 -10 1 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry><box><size>20 0.2 2</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>20 0.2 2</size></box></geometry>
          <material><ambient>0.5 0.5 0.5 1</ambient></material>
        </visual>
      </link>
    </model>

    <model name="east_wall">
      <pose>10 0 1 0 0 1.5708</pose>
      <link name="link">
        <collision name="collision">
          <geometry><box><size>20 0.2 2</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>20 0.2 2</size></box></geometry>
          <material><ambient>0.5 0.5 0.5 1</ambient></material>
        </visual>
      </link>
    </model>

    <model name="west_wall">
      <pose>-10 0 1 0 0 1.5708</pose>
      <link name="link">
        <collision name="collision">
          <geometry><box><size>20 0.2 2</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>20 0.2 2</size></box></geometry>
          <material><ambient>0.5 0.5 0.5 1</ambient></material>
        </visual>
      </link>
    </model>

    <!-- Warehouse zones -->
    <!-- Aisle markers -->
    <model name="aisle_marker_1">
      <pose>-5 0 0.05 0 0 0</pose>
      <link name="link">
        <visual name="visual">
          <geometry><box><size>0.1 15 0.1</size></box></geometry>
          <material><ambient>1 1 0 1</ambient></material>
        </visual>
      </link>
    </model>

    <model name="aisle_marker_2">
      <pose>5 0 0.05 0 0 0</pose>
      <link name="link">
        <visual name="visual">
          <geometry><box><size>0.1 15 0.1</size></box></geometry>
          <material><ambient>1 1 0 1</ambient></material>
        </visual>
      </link>
    </model>

    <!-- Inventory items -->
    <model name="inventory_item_1">
      <pose>-3 3 0.2 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry><box><size>0.3 0.3 0.4</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>0.3 0.3 0.4</size></box></geometry>
          <material><ambient>0.8 0.2 0.2 1</ambient></material>
        </visual>
        <inertial>
          <mass>5.0</mass>
          <inertia><ixx>0.1</ixx><iyy>0.1</iyy><izz>0.1</izz></inertia>
        </inertial>
      </link>
    </model>

    <model name="inventory_item_2">
      <pose>3 -3 0.2 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry><box><size>0.4 0.4 0.5</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>0.4 0.4 0.5</size></box></geometry>
          <material><ambient>0.2 0.8 0.2 1</ambient></material>
        </visual>
        <inertial>
          <mass>8.0</mass>
          <inertia><ixx>0.2</ixx><iyy>0.2</iyy><izz>0.2</izz></inertia>
        </inertial>
      </link>
    </model>

    <model name="inventory_item_3">
      <pose>-7 7 0.2 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry><cylinder><radius>0.2</radius><length>0.6</length></cylinder></geometry>
        </collision>
        <visual name="visual">
          <geometry><cylinder><radius>0.2</radius><length>0.6</length></geometry>
          <material><ambient>0.2 0.2 0.8 1</ambient></material>
        </visual>
        <inertial>
          <mass>3.0</mass>
          <inertia><ixx>0.05</ixx><iyy>0.05</iyy><izz>0.1</izz></inertia>
        </inertial>
      </link>
    </model>

    <!-- Loading dock -->
    <model name="loading_dock">
      <pose>-9 -9 0.1 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry><box><size>2 4 0.2</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>2 4 0.2</size></box></geometry>
          <material><ambient>0.6 0.6 0.6 1</ambient></material>
        </visual>
      </link>
    </model>

    <!-- Goal marker -->
    <model name="goal_marker">
      <pose>8 8 0.1 0 0 0</pose>
      <link name="link">
        <visual name="visual">
          <geometry><cylinder><radius>0.5</radius><length>0.2</length></cylinder></geometry>
          <material><ambient>0.2 0.8 0.2 0.5</ambient></material>
        </visual>
      </link>
    </model>
  </world>
</sdf>
```

### Step 9: Package Configuration

Update the package.xml file:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>warehouse_robot</name>
  <version>0.0.0</version>
  <description>Package for warehouse robot project using Isaac</description>
  <maintainer email="user@example.com">user</maintainer>
  <license>Apache-2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>sensor_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>vision_msgs</depend>
  <depend>tf2_ros</depend>
  <depend>launch</depend>
  <depend>launch_ros</depend>
  <depend>robot_state_publisher</depend>
  <depend>gazebo_ros</depend>
  <depend>isaac_ros_image_pipeline</depend>
  <depend>isaac_ros_detectnet</depend>
  <depend>isaac_ros_visual_slam</depend>
  <depend>navigation2</depend>
  <depend>nav2_bringup</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

Update the setup.py file:

```python
from setuptools import setup
import os
from glob import glob

package_name = 'warehouse_robot'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.urdf')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf')),
        (os.path.join('share', package_name, 'meshes'), glob('meshes/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='Package for warehouse robot project using Isaac',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'warehouse_manager = warehouse_robot.warehouse_manager:main',
        ],
    },
)
```

### Step 10: Running the System

To run the complete warehouse robot system:

```bash
# Build the package
cd ~/warehouse_robot_ws
colcon build --packages-select warehouse_robot
source install/setup.bash

# Run Isaac Sim with warehouse world (separately)
# Launch through Omniverse Launcher with warehouse_world.sdf

# Run the complete system
ros2 launch warehouse_robot warehouse_robot.launch.py

# In another terminal, send navigation goals
ros2 run warehouse_robot warehouse_manager --ros-args -p 'zone:=storage_a'
```

## Physical AI Concepts Demonstrated

### Multi-Modal Perception
- Camera-based object detection and classification
- LiDAR-based navigation and obstacle avoidance
- IMU-based localization and SLAM

### Integrated Navigation
- Visual SLAM for map building
- Navigation2 for path planning and execution
- Perception-aware obstacle avoidance

### Real-World Application
- Warehouse inventory management
- Autonomous navigation in structured environments
- Multi-sensor fusion for robust operation

## Troubleshooting

### Common Issues and Solutions

1. **Perception Performance**
   - Ensure GPU is properly configured
   - Check CUDA installation and drivers
   - Verify Isaac ROS packages are installed

2. **Navigation Issues**
   - Validate robot calibration
   - Check TF tree completeness
   - Verify sensor data quality

3. **SLAM Problems**
   - Ensure sufficient visual features
   - Check IMU integration
   - Validate camera calibration

4. **System Integration**
   - Verify all components are running
   - Check topic remappings
   - Monitor system performance

## Chapter Summary

This project integrated all the Isaac concepts learned in Module 3 into a comprehensive warehouse robot system. The system demonstrates how to combine Isaac Sim, Visual SLAM, Navigation2, and perception pipeline to create an intelligent robot capable of autonomous operation in a structured environment. The modular design allows for testing different components and evaluating their performance in a realistic scenario.

## Exercises

1. Extend the system to handle dynamic obstacles in the warehouse.
2. Implement more sophisticated inventory tracking with RFID or barcode scanning.
3. Add multi-robot coordination for larger warehouse operations.

## Next Steps

In the next chapter, we'll assess your understanding of Isaac and Physical AI concepts through practical challenges and exercises.