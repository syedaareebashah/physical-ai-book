---
sidebar_position: 6
---

# Project: Autonomous Navigation in Simulation

## Project Objectives

By completing this project, you will:
- Integrate Gazebo simulation with ROS 2 navigation stack
- Implement perception and planning systems for autonomous navigation
- Create a complete simulation environment for navigation testing
- Validate navigation performance using evaluation metrics
- Deploy the system from simulation to potential real hardware

## Project Overview

In this project, we'll build an autonomous navigation system that operates in Gazebo simulation. The system will:
1. Localize itself in the simulated environment
2. Plan paths to specified goals
3. Navigate while avoiding obstacles
4. Evaluate its performance using metrics
5. Provide a framework for testing different navigation algorithms

## System Architecture

```
Goal Input → Path Planner → Local Planner → Robot Controller → Sensor Feedback
     ↑                                                              ↓
     └─────────────────── Map & Localization ←──────────────────────┘
```

## Implementation Steps

### Step 1: Create the Project Package

```bash
# Create project workspace
mkdir -p ~/nav_simulation_ws/src
cd ~/nav_simulation_ws/src

# Create navigation package
ros2 pkg create --build-type ament_python nav_simulation
cd nav_simulation
```

### Step 2: Install Dependencies

```bash
# Install navigation stack
sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup

# Install simulation packages
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros2-control
```

### Step 3: Robot URDF Model

First, let's create a simple differential drive robot model:

```xml
<!-- File: nav_simulation/urdf/diff_robot.urdf -->
<?xml version="1.0"?>
<robot name="diff_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.2" radius="0.3"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 0.8 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.2" radius="0.3"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Left wheel -->
  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel"/>
    <origin xyz="0 0.3 -0.1" rpy="1.57 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <link name="left_wheel">
    <visual>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Right wheel -->
  <joint name="right_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="right_wheel"/>
    <origin xyz="0 -0.3 -0.1" rpy="1.57 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <link name="right_wheel">
    <visual>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Camera -->
  <joint name="camera_joint" type="fixed">
    <parent link="base_link"/>
    <child link="camera_link"/>
    <origin xyz="0.2 0 0.1" rpy="0 0 0"/>
  </joint>

  <link name="camera_link">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
      <material name="red">
        <color rgba="0.8 0 0 1"/>
      </material>
    </visual>
  </link>

  <!-- LiDAR -->
  <joint name="lidar_joint" type="fixed">
    <parent link="base_link"/>
    <child link="lidar_link"/>
    <origin xyz="0 0 0.2" rpy="0 0 0"/>
  </joint>

  <link name="lidar_link">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.05"/>
      </geometry>
      <material name="green">
        <color rgba="0 0.8 0 1"/>
      </material>
    </visual>
  </link>

  <!-- Gazebo plugins -->
  <gazebo>
    <plugin filename="libgazebo_ros2_control.so" name="gazebo_ros2_control">
      <parameters>$(find nav_simulation)/config/diff_robot_controllers.yaml</parameters>
    </plugin>
  </gazebo>

  <!-- Camera sensor -->
  <gazebo reference="camera_link">
    <sensor name="camera" type="camera">
      <always_on>true</always_on>
      <update_rate>30</update_rate>
      <camera name="head_camera">
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

Create the controller configuration file:

```yaml
# File: nav_simulation/config/diff_robot_controllers.yaml
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

    wheel_separation: 0.6
    wheel_radius: 0.1

    use_stamped_vel: false

    # Publish rate
    publish_rate: 50.0
    odom_frame_id: odom
    base_frame_id: base_link
    pose_covariance_diagonal: [0.001, 0.001, 0.001, 0.001, 0.001, 0.01]
    twist_covariance_diagonal: [0.001, 0.001, 0.001, 0.001, 0.001, 0.01]
```

### Step 5: Navigation Configuration

Create the navigation configuration:

```yaml
# File: nav_simulation/config/nav2_params.yaml
amcl:
  ros__parameters:
    use_sim_time: True
    alpha1: 0.2
    alpha2: 0.2
    alpha3: 0.2
    alpha4: 0.2
    alpha5: 0.2
    base_frame_id: "base_link"
    beam_skip_distance: 0.5
    beam_skip_error_threshold: 0.9
    beam_skip_threshold: 0.3
    do_beamskip: false
    global_frame_id: "map"
    lambda_short: 0.1
    laser_likelihood_max_dist: 2.0
    laser_max_range: 100.0
    laser_min_range: -1.0
    laser_model_type: "likelihood_field"
    max_beams: 60
    max_particles: 2000
    min_particles: 500
    odom_frame_id: "odom"
    pf_err: 0.05
    pf_z: 0.99
    recovery_alpha_fast: 0.0
    recovery_alpha_slow: 0.0
    resample_interval: 1
    robot_model_type: "nav2_amcl::DifferentialMotionModel"
    save_pose_rate: 0.5
    sigma_hit: 0.2
    tf_broadcast: true
    transform_tolerance: 1.0
    update_min_a: 0.2
    update_min_d: 0.25
    z_hit: 0.5
    z_max: 0.05
    z_rand: 0.5
    z_short: 0.05
    scan_topic: scan

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
    robot_base_frame: base_link
    odom_topic: /odom
    bt_loop_duration: 10
    default_server_timeout: 20
    enable_groot_monitoring: True
    groot_zmq_publisher_port: 1666
    groot_zmq_server_port: 1667
    # Specify the path where the BT XML files are located
    plugin_lib_names:
    - nav2_compute_path_to_pose_action_bt_node
    - nav2_follow_path_action_bt_node
    - nav2_back_up_action_bt_node
    - nav2_spin_action_bt_node
    - nav2_wait_action_bt_node
    - nav2_clear_costmap_service_bt_node
    - nav2_is_stuck_condition_bt_node
    - nav2_goal_reached_condition_bt_node
    - nav2_goal_updated_condition_bt_node
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

    # Progress checker parameters
    progress_checker:
      plugin: "nav2_controller::SimpleProgressChecker"
      required_movement_radius: 0.5
      movement_time_allowance: 10.0

    # Goal checker parameters
    goal_checker:
      plugin: "nav2_controller::SimpleGoalChecker"
      xy_goal_tolerance: 0.25
      yaw_goal_tolerance: 0.25
      stateful: True

    # Controller parameters
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
      robot_base_frame: base_link
      use_sim_time: True
      rolling_window: true
      width: 3
      height: 3
      resolution: 0.05
      robot_radius: 0.3
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
      robot_base_frame: base_link
      use_sim_time: True
      robot_radius: 0.3
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
    yaml_filename: "turtlebot3_world.yaml"

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

waypoint_follower:
  ros__parameters:
    loop_rate: 20
    stop_on_failure: false
    waypoint_task_executor_plugin: "wait_at_waypoint"
    wait_at_waypoint:
      plugin: "nav2_waypoint_follower::WaitAtWaypoint"
      enabled: True
      waypoint_pause_duration: 200
```

### Step 6: Simulation Launch File

Create the main launch file:

```python
# File: nav_simulation/launch/navigation_simulation.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    world = LaunchConfiguration('world', default='empty.sdf')

    # Launch Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'world': world,
            'verbose': 'true'
        }.items()
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': open(PathJoinSubstitution([
                FindPackageShare('nav_simulation'),
                'urdf',
                'diff_robot.urdf'
            ]).perform({})).read()
        }]
    )

    # Spawn robot in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'diff_robot',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.0'
        ],
        output='screen'
    )

    # Launch navigation
    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('nav2_bringup'),
                'launch',
                'navigation_launch.py'
            ])
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'params_file': PathJoinSubstitution([
                FindPackageShare('nav_simulation'),
                'config',
                'nav2_params.yaml'
            ])
        }.items()
    )

    # Launch RViz
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', PathJoinSubstitution([
            FindPackageShare('nav2_bringup'),
            'rviz',
            'nav2_default_view.rviz'
        ])],
        parameters=[{'use_sim_time': use_sim_time}]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'world',
            default_value='empty.sdf',
            description='Choose one of the world files from `/usr/share/gazebo-11/worlds`'
        ),
        gazebo,
        robot_state_publisher,
        spawn_entity,
        navigation,
        rviz
    ])
```

### Step 7: Custom World File

Create a world file for navigation testing:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="navigation_world">
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
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane><normal>0 0 1</normal></plane>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <plane><normal>0 0 1</normal></plane>
          </geometry>
          <material>
            <ambient>0.7 0.7 0.7 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Walls creating a maze-like structure -->
    <model name="wall_1">
      <pose>0 5 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box><size>10 0.2 1</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>10 0.2 1</size></box>
          </geometry>
          <material><ambient>0.5 0.5 0.5 1</ambient></material>
        </visual>
      </link>
    </model>

    <model name="wall_2">
      <pose>0 -5 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box><size>10 0.2 1</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>10 0.2 1</size></box>
          </geometry>
          <material><ambient>0.5 0.5 0.5 1</ambient></material>
        </visual>
      </link>
    </model>

    <model name="wall_3">
      <pose>5 0 0.5 0 0 1.5708</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box><size>10 0.2 1</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>10 0.2 1</size></box>
          </geometry>
          <material><ambient>0.5 0.5 0.5 1</ambient></material>
        </visual>
      </link>
    </model>

    <model name="wall_4">
      <pose>-5 0 0.5 0 0 1.5708</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box><size>10 0.2 1</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>10 0.2 1</size></box>
          </geometry>
          <material><ambient>0.5 0.5 0.5 1</ambient></material>
        </visual>
      </link>
    </model>

    <!-- Interior obstacles -->
    <model name="obstacle_1">
      <pose>2 2 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box><size>1 1 1</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>1 1 1</size></box>
          </geometry>
          <material><ambient>0.8 0.2 0.2 1</ambient></material>
        </visual>
      </link>
    </model>

    <model name="obstacle_2">
      <pose>-2 -2 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <cylinder><radius>0.5</radius><length>1</length></cylinder>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <cylinder><radius>0.5</radius><length>1</length></cylinder>
          </geometry>
          <material><ambient>0.2 0.8 0.2 1</ambient></material>
        </visual>
      </link>
    </model>

    <model name="obstacle_3">
      <pose>0 3 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <sphere><radius>0.4</radius></sphere>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <sphere><radius>0.4</radius></sphere>
          </geometry>
          <material><ambient>0.2 0.2 0.8 1</ambient></material>
        </visual>
      </link>
    </model>

    <!-- Goal marker -->
    <model name="goal_marker">
      <pose>4 4 0.1 0 0 0</pose>
      <link name="link">
        <visual name="visual">
          <geometry>
            <cylinder><radius>0.3</radius><length>0.2</length></cylinder>
          </geometry>
          <material><ambient>0.2 0.8 0.2 0.5</ambient></material>
        </visual>
      </link>
    </model>
  </world>
</sdf>
```

### Step 8: Performance Evaluation Node

Create a node to evaluate navigation performance:

```python
# File: nav_simulation/nav_simulation/navigation_evaluator.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, Bool
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import numpy as np
from collections import deque
import time

class NavigationEvaluator(Node):
    def __init__(self):
        super().__init__('navigation_evaluator')

        # Parameters
        self.declare_parameter('evaluation_duration', 300)  # 5 minutes
        self.declare_parameter('goal_tolerance', 0.5)
        self.declare_parameter('collision_threshold', 0.3)

        # TF2 setup for localization
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.path_sub = self.create_subscription(
            Path, '/plan', self.path_callback, 10)

        # Publishers
        self.collision_pub = self.create_publisher(Bool, '/collision', 10)
        self.performance_pub = self.create_publisher(Float32, '/navigation_performance', 10)
        self.distance_pub = self.create_publisher(Float32, '/traveled_distance', 10)

        # Evaluation parameters
        self.evaluation_duration = self.get_parameter('evaluation_duration').value
        self.goal_tolerance = self.get_parameter('goal_tolerance').value
        self.collision_threshold = self.get_parameter('collision_threshold').value

        # State tracking
        self.start_time = self.get_clock().now()
        self.start_position = None
        self.current_position = None
        self.previous_position = None
        self.total_distance = 0.0
        self.collision_count = 0
        self.path_efficiency = 0.0
        self.collision_detected = False

        # Goal position (will be set when navigation starts)
        self.goal_position = None

        # Setup evaluation timer
        self.eval_timer = self.create_timer(1.0, self.evaluate_performance)

        self.get_logger().info('Navigation Evaluator Node Started')

    def odom_callback(self, msg):
        self.current_position = np.array([msg.pose.pose.position.x,
                                         msg.pose.pose.position.y])

        if self.start_position is None:
            self.start_position = self.current_position.copy()

        # Calculate distance traveled
        if self.previous_position is not None:
            step_distance = np.linalg.norm(self.current_position - self.previous_position)
            self.total_distance += step_distance

        self.previous_position = self.current_position.copy()

        # Publish traveled distance
        distance_msg = Float32()
        distance_msg.data = self.total_distance
        self.distance_pub.publish(distance_msg)

    def scan_callback(self, msg):
        # Check for potential collisions
        if min(msg.ranges) < self.collision_threshold:
            self.collision_count += 1
            self.collision_detected = True

            collision_msg = Bool()
            collision_msg.data = True
            self.collision_pub.publish(collision_msg)

    def path_callback(self, msg):
        # Calculate path efficiency (if we have both planned path and actual path)
        if len(msg.poses) > 0 and self.current_position is not None:
            # Calculate planned path length
            planned_length = 0.0
            for i in range(1, len(msg.poses)):
                p1 = np.array([msg.poses[i-1].pose.position.x,
                              msg.poses[i-1].pose.position.y])
                p2 = np.array([msg.poses[i].pose.position.x,
                              msg.poses[i].pose.position.y])
                planned_length += np.linalg.norm(p2 - p1)

            # Calculate efficiency as ratio of direct distance to actual path
            if self.start_position is not None:
                direct_distance = np.linalg.norm(self.current_position - self.start_position)
                if planned_length > 0:
                    self.path_efficiency = direct_distance / planned_length

    def evaluate_performance(self):
        if self.current_position is None:
            return

        # Calculate time elapsed
        current_time = self.get_clock().now()
        time_elapsed = (current_time - self.start_time).nanoseconds / 1e9

        # Calculate metrics
        success_rate = 0.0
        efficiency_score = 0.0
        safety_score = 0.0

        # Success: reached goal within tolerance
        if self.goal_position is not None:
            distance_to_goal = np.linalg.norm(self.current_position - self.goal_position)
            if distance_to_goal <= self.goal_tolerance:
                success_rate = 1.0

        # Efficiency: path length vs direct distance
        if self.start_position is not None and self.goal_position is not None:
            direct_distance = np.linalg.norm(self.goal_position - self.start_position)
            if direct_distance > 0:
                efficiency_score = direct_distance / max(self.total_distance, direct_distance)

        # Safety: inverse of collision frequency
        max_expected_time = self.evaluation_duration
        expected_collision_free_periods = time_elapsed / 10.0  # per 10 seconds
        safety_score = max(0, 1.0 - (self.collision_count / max(1, expected_collision_free_periods)))

        # Overall performance score (weighted combination)
        performance_score = (success_rate * 0.4 +
                           efficiency_score * 0.3 +
                           safety_score * 0.3)

        # Publish performance score
        perf_msg = Float32()
        perf_msg.data = performance_score
        self.performance_pub.publish(perf_msg)

        # Log performance metrics
        self.get_logger().info(f'Performance Metrics - '
                              f'Time: {time_elapsed:.1f}s, '
                              f'Distance: {self.total_distance:.2f}m, '
                              f'Collisions: {self.collision_count}, '
                              f'Score: {performance_score:.3f}')

        # Check if evaluation period is over
        if time_elapsed > self.evaluation_duration:
            self.get_logger().info(f'Evaluation completed after {time_elapsed:.1f}s')
            self.get_logger().info(f'Final Score: {performance_score:.3f}')
            # Could trigger next test scenario here

    def set_goal(self, goal_x, goal_y):
        """Set the goal position for evaluation"""
        self.goal_position = np.array([goal_x, goal_y])

def main(args=None):
    rclpy.init(args=args)
    evaluator = NavigationEvaluator()

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

### Step 9: Package Configuration

Update the package.xml file:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>nav_simulation</name>
  <version>0.0.0</version>
  <description>Package for autonomous navigation simulation project</description>
  <maintainer email="user@example.com">user</maintainer>
  <license>Apache-2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>nav_msgs</depend>
  <depend>sensor_msgs</depend>
  <depend>tf2_ros</depend>
  <depend>launch</depend>
  <depend>launch_ros</depend>
  <depend>robot_state_publisher</depend>
  <depend>gazebo_ros</depend>
  <depend>gazebo_plugins</depend>

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

package_name = 'nav_simulation'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.urdf')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='Package for autonomous navigation simulation project',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'navigation_evaluator = nav_simulation.navigation_evaluator:main',
        ],
    },
)
```

### Step 10: Running the System

To run the complete navigation simulation:

```bash
# Build the package
cd ~/nav_simulation_ws
colcon build --packages-select nav_simulation
source install/setup.bash

# Run the simulation
ros2 launch nav_simulation navigation_simulation.launch.py world:=path/to/your/world.sdf

# In another terminal, send a navigation goal
ros2 run nav2_msgs navigate_to_pose --ros-args -p 'pose.position.x:=4.0' -p 'pose.position.y:=4.0' -p 'pose.orientation.w:=1.0'

# Or use the Rviz interface to set goals graphically
```

## Physical AI Concepts Demonstrated

### Multi-Sensor Integration
- LiDAR for obstacle detection
- IMU for localization (simulated)
- Camera for visual perception (simulated)

### Navigation Stack Integration
- AMCL for localization
- Nav2 for path planning and execution
- Costmap for obstacle avoidance

### Performance Evaluation
- Quantitative metrics for navigation performance
- Safety assessment through collision detection
- Path efficiency measurement

## Troubleshooting

### Common Issues and Solutions

1. **Robot not moving**
   - Check controller configuration and connection
   - Verify TF tree is properly published
   - Ensure navigation stack is running

2. **Localization problems**
   - Verify initial pose is set correctly
   - Check sensor data quality
   - Adjust AMCL parameters if needed

3. **Path planning failures**
   - Check costmap configuration
   - Verify map quality and resolution
   - Adjust planner parameters

4. **Performance issues**
   - Reduce simulation complexity if needed
   - Optimize costmap resolution
   - Check CPU/memory usage

## Chapter Summary

This project integrated all the simulation concepts learned in Module 2 into a comprehensive autonomous navigation system. The system demonstrates how to combine Gazebo simulation, ROS 2 navigation stack, sensor simulation, and performance evaluation in a cohesive Physical AI application. The modular design allows for testing different navigation algorithms and evaluating their performance in various simulated environments.

## Exercises

1. Modify the robot model to include additional sensors (e.g., IMU, GPS simulation).
2. Implement different navigation strategies and compare their performance.
3. Create additional test environments with varying complexity levels.

## Next Steps

In the next chapter, we'll assess your understanding of simulation concepts through practical challenges and exercises.