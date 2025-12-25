---
sidebar_position: 2
---

# Gazebo Fundamentals

## Chapter Objectives

By the end of this chapter, you will be able to:
- Launch and configure Gazebo simulation environments
- Create and customize simulation worlds
- Spawn and control robots in Gazebo
- Configure physics properties and parameters
- Integrate Gazebo with ROS 2 for Physical AI applications

## Gazebo Architecture

Gazebo is built on a client-server architecture:

### Server Components
- **Gazebo Server**: Handles physics simulation, sensors, and plugins
- **Physics Engine**: Simulates rigid body dynamics (ODE, Bullet, DART)
- **Sensor System**: Emulates various sensors with realistic properties

### Client Components
- **Gazebo Client**: Provides the graphical user interface
- **Model Database**: Stores robot and environment models (Fuel)

## Launching Gazebo

### Basic Launch
```bash
# Launch Gazebo with default empty world
gz sim -r

# Launch with a specific world file
gz sim -r -v 4 empty.sdf  # Higher verbosity for debugging
```

For ROS 2 integration:
```bash
# Launch through ROS 2 launch system
ros2 launch gazebo_ros gazebo.launch.py

# Launch with custom world
ros2 launch gazebo_ros gazebo.launch.py world:=/path/to/world.sdf
```

## World Files and SDF Format

Gazebo uses SDF (Simulation Description Format) to define simulation worlds:

### Basic World Structure
```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="my_world">
    <!-- Physics engine configuration -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>

    <!-- Environment lighting -->
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.4 0.2 -0.9</direction>
    </light>

    <!-- Ground plane -->
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <material>
            <ambient>0.7 0.7 0.7 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
            <specular>0.7 0.7 0.7 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <!-- Your robot will go here -->
  </world>
</sdf>
```

## Creating Custom Worlds

### Simple Obstacle World
```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="obstacle_course">
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

    <!-- Obstacles -->
    <model name="obstacle_1">
      <pose>2 0 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box><size>1 0.2 1</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>1 0.2 1</size></box>
          </geometry>
          <material>
            <ambient>0.8 0.2 0.2 1</ambient>
            <diffuse>0.8 0.2 0.2 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>0.1667</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.1667</iyy>
            <iyz>0</iyz>
            <izz>0.1667</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <!-- More obstacles can be added -->
  </world>
</sdf>
```

## Spawning Robots in Gazebo

### Using Command Line
```bash
# Spawn a model from the model database
gz model -f model.sdf -m robot_name --model-name robot_name

# For ROS 2
ros2 run gazebo_ros spawn_entity.py -file /path/to/robot.urdf -entity my_robot -x 0 -y 0 -z 1
```

### Using ROS 2 Service Calls
```python
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity

class RobotSpawner(Node):
    def __init__(self):
        super().__init__('robot_spawner')
        self.cli = self.create_client(SpawnEntity, '/spawn_entity')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

    def spawn_robot(self, robot_xml, robot_name, initial_pose):
        req = SpawnEntity.Request()
        req.name = robot_name
        req.xml = robot_xml
        req.initial_pose.position.x = initial_pose[0]
        req.initial_pose.position.y = initial_pose[1]
        req.initial_pose.position.z = initial_pose[2]

        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            self.get_logger().info(f'Successfully spawned {robot_name}')
        else:
            self.get_logger().error(f'Failed to spawn {robot_name}')
```

## Physics Configuration

### Understanding Physics Parameters

```xml
<physics type="ode">
  <!-- Time step for physics updates -->
  <max_step_size>0.001</max_step_size>

  <!-- Real-time factor (1.0 = real-time, >1 = faster than real-time) -->
  <real_time_factor>1.0</real_time_factor>

  <!-- Physics updates per second -->
  <real_time_update_rate>1000</real_time_update_rate>

  <!-- Gravity -->
  <gravity>0 0 -9.8</gravity>

  <!-- ODE-specific parameters -->
  <ode>
    <solver>
      <type>quick</type>
      <iters>10</iters>
      <sor>1.3</sor>
    </solver>
    <constraints>
      <cfm>0.0</cfm>
      <erp>0.2</erp>
      <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

### Performance vs. Accuracy Trade-offs
- **Smaller time steps**: More accurate but slower simulation
- **Higher real-time factor**: Faster simulation but potentially unstable
- **More solver iterations**: Better stability but slower performance

## Sensor Simulation in Gazebo

### Camera Sensor
```xml
<sensor name="camera" type="camera">
  <always_on>true</always_on>
  <update_rate>30</update_rate>
  <camera name="head">
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
  <plugin filename="libgazebo_ros_camera.so" name="camera_controller">
    <frame_name>camera_frame</frame_name>
    <topic_name>/camera/image_raw</topic_name>
  </plugin>
</sensor>
```

### LiDAR Sensor
```xml
<sensor name="lidar" type="ray">
  <always_on>true</always_on>
  <update_rate>10</update_rate>
  <ray>
    <scan>
      <horizontal>
        <samples>360</samples>
        <resolution>1.0</resolution>
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
    <frame_name>lidar_frame</frame_name>
  </plugin>
</sensor>
```

## ROS 2 Integration

### Gazebo ROS Packages
The `gazebo_ros_pkgs` provide the bridge between Gazebo and ROS 2:

```xml
<!-- In your robot's URDF/Xacro -->
<gazebo>
  <plugin filename="libgazebo_ros2_control.so" name="gazebo_ros2_control">
    <parameters>$(find my_robot_package)/config/my_robot_controllers.yaml</parameters>
  </plugin>
</gazebo>
```

### Controller Configuration
```yaml
# config/my_robot_controllers.yaml
controller_manager:
  ros__parameters:
    update_rate: 100  # Hz

    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster

    arm_controller:
      type: position_controllers/JointGroupPositionController

arm_controller:
  ros__parameters:
    joints:
      - joint1
      - joint2
      - joint3
```

## Physical AI Simulation Scenarios

### Navigation Simulation
Setting up a world for navigation testing:

```xml
<!-- In your world file -->
<!-- Add various obstacles with different shapes and sizes -->
<model name="wall_1">
  <pose>5 0 1 0 0 0</pose>
  <link name="link">
    <collision name="collision">
      <geometry>
        <box><size>0.1 10 2</size></box>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <box><size>0.1 10 2</size></box>
      </geometry>
      <material><ambient>0.5 0.5 0.5 1</ambient></material>
    </visual>
  </link>
</model>
```

### Manipulation Simulation
Creating objects for manipulation tasks:

```xml
<!-- Add objects that can be grasped -->
<model name="graspable_object">
  <pose>1 1 0.5 0 0 0</pose>
  <link name="link">
    <collision name="collision">
      <geometry>
        <cylinder><radius>0.05</radius><length>0.2</length></cylinder>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <cylinder><radius>0.05</radius><length>0.2</length></cylinder>
      </geometry>
      <material><ambient>0.2 0.8 0.2 1</ambient></material>
    </visual>
    <inertial>
      <mass>0.1</mass>
      <inertia><ixx>0.001</ixx><iyy>0.001</iyy><izz>0.0005</izz></inertia>
    </inertial>
  </link>
</model>
```

## Best Practices for Physical AI Simulation

### Model Accuracy
- Use realistic mass and inertia properties
- Include appropriate friction coefficients
- Model sensor noise and limitations
- Validate simulation against real-world data

### Performance Optimization
- Use simplified collision geometry for planning
- Optimize visual geometry separately from collision geometry
- Adjust physics parameters based on simulation requirements
- Use appropriate update rates for different sensors

### Safety Considerations
- Implement simulation bounds to prevent robot wandering
- Include emergency stop mechanisms in simulation
- Test failure scenarios safely in simulation

## Chapter Summary

Gazebo provides a powerful simulation environment for Physical AI applications with native ROS 2 integration. Understanding SDF world files, physics configuration, and sensor simulation enables the creation of realistic environments for testing robotic systems. Proper integration with ROS 2 allows seamless transition between simulation and real hardware.

## Exercises

1. Create a simple world file with a robot and several obstacles.
2. Configure a camera sensor in Gazebo and verify it publishes ROS 2 messages.
3. Adjust physics parameters to optimize simulation performance vs. accuracy.

## Next Steps

In the next chapter, we'll explore simulating various types of sensors for Physical AI applications, including cameras, LiDAR, IMU, and force/torque sensors.