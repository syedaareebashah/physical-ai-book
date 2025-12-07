---
sidebar_position: 5
---

# Building Test Environments

## Chapter Objectives

By the end of this chapter, you will be able to:
- Design comprehensive test environments for Physical AI systems
- Create obstacle courses and navigation challenges
- Build manipulation and interaction scenarios
- Implement multi-environment testing frameworks
- Validate robot performance across diverse environments

## Environment Design Principles for Physical AI

### Realism vs. Complexity Balance

When building test environments for Physical AI, consider:

1. **Deployment Fidelity**: How closely does the environment match real deployment scenarios?
2. **Computational Efficiency**: Can the environment run in real-time for testing?
3. **Variety**: Does the environment test diverse scenarios and edge cases?
4. **Measurability**: Can you quantify robot performance in the environment?

### Environmental Complexity Levels

#### Level 1: Basic Validation
- Simple, open spaces
- Basic geometric obstacles
- Static environments
- Single-task scenarios

#### Level 2: Intermediate Challenges
- Complex geometries
- Multiple simultaneous tasks
- Dynamic elements
- Varying lighting conditions

#### Level 3: Advanced Testing
- Real-world inspired scenarios
- Multiple robots interaction
- Environmental disturbances
- Long-term autonomy tests

## Gazebo Environment Construction

### Creating Modular World Components

Building reusable environment components:

```xml
<!-- reusable_wall.sdf -->
<sdf version="1.7">
  <model name="reusable_wall">
    <pose>0 0 0 0 0 0</pose>
    <link name="wall_link">
      <collision name="collision">
        <geometry>
          <box>
            <size>3 0.2 2</size>
          </box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box>
            <size>3 0.2 2</size>
          </box>
        </geometry>
        <material>
          <ambient>0.5 0.5 0.5 1</ambient>
          <diffuse>0.5 0.5 0.5 1</diffuse>
          <specular>0.5 0.5 0.5 1</specular>
        </material>
      </visual>
      <inertial>
        <mass>10.0</mass>
        <inertia>
          <ixx>1.0</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>1.0</iyy>
          <iyz>0</iyz>
          <izz>1.0</izz>
        </inertia>
      </inertial>
    </link>
  </model>
</sdf>
```

### Navigation Test Environments

Creating challenging navigation scenarios:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="navigation_challenge">
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

    <!-- Outer walls -->
    <model name="outer_wall_north">
      <pose>0 5 1 0 0 0</pose>
      <include>
        <uri>model://reusable_wall</uri>
      </include>
    </model>

    <model name="outer_wall_south">
      <pose>0 -5 1 0 0 3.14159</pose>
      <include>
        <uri>model://reusable_wall</uri>
      </include>
    </model>

    <model name="outer_wall_east">
      <pose>5 0 1 0 0 1.5708</pose>
      <include>
        <uri>model://reusable_wall</uri>
      </include>
    </model>

    <model name="outer_wall_west">
      <pose>-5 0 1 0 0 -1.5708</pose>
      <include>
        <uri>model://reusable_wall</uri>
      </include>
    </model>

    <!-- Interior obstacles -->
    <model name="narrow_passage_wall1">
      <pose>0 2 1 0 0 0</pose>
      <include>
        <uri>model://reusable_wall</uri>
      </include>
    </model>

    <model name="narrow_passage_wall2">
      <pose>0 -2 1 0 0 0</pose>
      <include>
        <uri>model://reusable_wall</uri>
      </include>
    </model>

    <!-- Create a gap in the middle -->
    <model name="gap_filler">
      <pose>0 0 1 0 0 0</pose>
      <link name="link">
        <visual name="visual">
          <geometry>
            <box><size>0.5 0.1 0.1</size>
          </geometry>
        </visual>
      </link>
    </model>

    <!-- Moving obstacles -->
    <model name="moving_obstacle">
      <pose>2 0 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <sphere><radius>0.3</radius></sphere>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <sphere><radius>0.3</radius></sphere>
          </geometry>
          <material><ambient>0.8 0.2 0.2 1</ambient></material>
        </visual>
        <inertial>
          <mass>1.0</mass>
          <inertia><ixx>0.1</ixx><iyy>0.1</iyy><izz>0.1</izz></inertia>
        </inertial>
      </link>
      <!-- Add plugin for movement -->
      <plugin filename="libgazebo_ros_p3d.so" name="moving_obstacle_controller">
        <alwaysOn>true</alwaysOn>
        <updateRate>30</updateRate>
        <bodyName>link</bodyName>
        <topicName>/moving_obstacle/pose</topicName>
        <gaussianNoise>0.0</gaussianNoise>
        <frameName>map</frameName>
      </plugin>
    </model>

    <!-- Goal marker -->
    <model name="goal_marker">
      <pose>4 0 0.1 0 0 0</pose>
      <link name="link">
        <visual name="visual">
          <geometry>
            <cylinder><radius>0.2</radius><length>0.2</length></cylinder>
          </geometry>
          <material><ambient>0.2 0.8 0.2 1</ambient></material>
        </visual>
      </link>
    </model>
  </world>
</sdf>
```

### Manipulation Test Environments

Creating environments for manipulation tasks:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="manipulation_lab">
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

    <!-- Workbench -->
    <model name="workbench">
      <pose>0 0 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box><size>2 1 1</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>2 1 1</size></box>
          </geometry>
          <material><ambient>0.6 0.4 0.2 1</ambient></material>
        </visual>
        <inertial>
          <mass>50.0</mass>
          <inertia>
            <ixx>10.0</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>10.0</iyy>
            <iyz>0</iyz>
            <izz>10.0</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <!-- Objects to manipulate -->
    <model name="red_block">
      <pose>-0.5 0.2 1.05 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box><size>0.1 0.1 0.1</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>0.1 0.1 0.1</size></box>
          </geometry>
          <material><ambient>0.8 0.2 0.2 1</ambient></material>
        </visual>
        <inertial>
          <mass>0.1</mass>
          <inertia>
            <ixx>0.0001</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.0001</iyy>
            <iyz>0</iyz>
            <izz>0.0001</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <model name="blue_cylinder">
      <pose>-0.2 0.2 1.05 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <cylinder><radius>0.05</radius><length>0.1</length></cylinder>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <cylinder><radius>0.05</radius><length>0.1</length></cylinder>
          </geometry>
          <material><ambient>0.2 0.2 0.8 1</ambient></material>
        </visual>
        <inertial>
          <mass>0.05</mass>
          <inertia>
            <ixx>0.0001</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.0001</iyy>
            <iyz>0</iyz>
            <izz>0.0002</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <!-- Target zones -->
    <model name="target_zone_red">
      <pose>0.5 0.2 0.51 0 0 0</pose>
      <link name="link">
        <visual name="visual">
          <geometry>
            <box><size>0.3 0.3 0.02</size></box>
          </geometry>
          <material><ambient>0.8 0.2 0.2 0.5</ambient></material>
        </visual>
      </link>
    </model>

    <model name="target_zone_blue">
      <pose>0.5 -0.2 0.51 0 0 0</pose>
      <link name="link">
        <visual name="visual">
          <geometry>
            <box><size>0.3 0.3 0.02</size></box>
          </geometry>
          <material><ambient>0.2 0.2 0.8 0.5</ambient></material>
        </visual>
      </link>
    </model>

    <!-- Storage bin -->
    <model name="storage_bin">
      <pose>0.8 0 0.6 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box><size>0.4 0.4 0.2</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>0.4 0.4 0.2</size></box>
          </geometry>
          <material><ambient>0.4 0.4 0.4 1</ambient></material>
        </visual>
      </link>
    </model>
  </world>
</sdf>
```

## Unity Environment Construction

### Procedural Environment Generation

Creating environments algorithmically in Unity:

```csharp
// Example: Procedural maze generator for navigation testing
using UnityEngine;
using System.Collections.Generic;

public class MazeGenerator : MonoBehaviour
{
    [Header("Maze Parameters")]
    public int width = 20;
    public int height = 20;
    public GameObject wallPrefab;
    public GameObject floorPrefab;

    [Header("Navigation Goals")]
    public Transform startMarker;
    public Transform goalMarker;

    private bool[,] mazeGrid;
    private List<Vector3> pathPositions;

    void Start()
    {
        GenerateMaze();
        PlaceStartAndGoal();
    }

    void GenerateMaze()
    {
        mazeGrid = new bool[width, height];
        pathPositions = new List<Vector3>();

        // Initialize grid (all walls)
        for (int x = 0; x < width; x++)
        {
            for (int z = 0; z < height; z++)
            {
                mazeGrid[x, z] = true; // true = wall, false = path
            }
        }

        // Generate maze using recursive backtracking
        GenerateMazeRecursive(1, 1);

        // Create the actual game objects
        CreateMazeObjects();
    }

    void GenerateMazeRecursive(int x, int z)
    {
        mazeGrid[x, z] = false; // Mark as path
        pathPositions.Add(new Vector3(x, 0, z));

        // Shuffle directions
        List<Vector2Int> directions = new List<Vector2Int>
        {
            new Vector2Int(2, 0),   // Right
            new Vector2Int(-2, 0),  // Left
            new Vector2Int(0, 2),   // Up
            new Vector2Int(0, -2)   // Down
        };

        directions.Shuffle(); // Extension method to shuffle list

        foreach (Vector2Int dir in directions)
        {
            int newX = x + dir.x;
            int newZ = z + dir.y;

            if (newX > 0 && newX < width - 1 && newZ > 0 && newZ < height - 1 && mazeGrid[newX, newZ])
            {
                // Remove wall between current and new cell
                mazeGrid[x + dir.x / 2, z + dir.y / 2] = false;
                pathPositions.Add(new Vector3(x + dir.x / 2, 0, z + dir.y / 2));

                GenerateMazeRecursive(newX, newZ);
            }
        }
    }

    void CreateMazeObjects()
    {
        // Create floor
        GameObject floor = Instantiate(floorPrefab);
        floor.transform.position = new Vector3(width / 2f, -0.5f, height / 2f);
        floor.transform.localScale = new Vector3(width, 1, height);

        // Create walls
        for (int x = 0; x < width; x++)
        {
            for (int z = 0; z < height; z++)
            {
                if (mazeGrid[x, z])
                {
                    GameObject wall = Instantiate(wallPrefab);
                    wall.transform.position = new Vector3(x, 0, z);
                }
            }
        }
    }

    void PlaceStartAndGoal()
    {
        // Place start at beginning of path
        if (pathPositions.Count > 0)
        {
            startMarker.position = pathPositions[0] + Vector3.up;
        }

        // Place goal at end of path
        if (pathPositions.Count > 0)
        {
            goalMarker.position = pathPositions[pathPositions.Count - 1] + Vector3.up;
        }
    }
}

// Extension method for shuffling lists
public static class ListExtensions
{
    public static void Shuffle<T>(this IList<T> list)
    {
        System.Random rng = new System.Random();
        int n = list.Count;
        while (n > 1)
        {
            n--;
            int k = rng.Next(n + 1);
            T value = list[k];
            list[k] = list[n];
            list[n] = value;
        }
    }
}
```

### Dynamic Environment Elements

Creating environments with changing elements:

```csharp
// Example: Dynamic obstacle system
using UnityEngine;
using System.Collections;

public class DynamicEnvironment : MonoBehaviour
{
    [Header("Dynamic Elements")]
    public GameObject[] movingObstacles;
    public Transform[] obstaclePaths;
    public float moveSpeed = 2.0f;

    [Header("Environmental Changes")]
    public GameObject[] toggleObjects;
    public float changeInterval = 10.0f;

    [Header("Weather System")]
    public Light sunLight;
    public AnimationCurve lightIntensityCurve;
    public float dayNightCycleDuration = 120.0f;

    private float timeSinceChange = 0f;
    private float dayNightTime = 0f;

    void Start()
    {
        StartCoroutine(MoveObstacles());
        StartCoroutine(ChangeEnvironment());
    }

    IEnumerator MoveObstacles()
    {
        while (true)
        {
            foreach (GameObject obstacle in movingObstacles)
            {
                // Move obstacle along its path
                MoveObstacle(obstacle);
            }
            yield return new WaitForSeconds(0.1f);
        }
    }

    void MoveObstacle(GameObject obstacle)
    {
        // Simple back-and-forth movement
        float moveDistance = moveSpeed * Time.deltaTime;
        obstacle.transform.Translate(Vector3.forward * moveDistance);

        // Check if reached end of path and reverse direction
        if (Vector3.Distance(obstacle.transform.position, obstaclePaths[0].position) > 10f)
        {
            obstacle.transform.Rotate(0, 180, 0); // Turn around
        }
    }

    IEnumerator ChangeEnvironment()
    {
        while (true)
        {
            yield return new WaitForSeconds(changeInterval);
            ToggleEnvironmentElements();
        }
    }

    void ToggleEnvironmentElements()
    {
        foreach (GameObject obj in toggleObjects)
        {
            obj.SetActive(!obj.activeSelf);
        }
    }

    void Update()
    {
        // Update day/night cycle
        dayNightTime += Time.deltaTime / dayNightCycleDuration;
        if (dayNightTime > 1) dayNightTime = 0;

        float intensity = lightIntensityCurve.Evaluate(dayNightTime);
        sunLight.intensity = intensity;
    }
}
```

## Environment Validation and Metrics

### Performance Metrics Framework

Creating systems to measure robot performance:

```python
# File: environment_evaluation/evaluation_metrics.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, Bool
import numpy as np
from collections import deque

class EnvironmentEvaluator(Node):
    def __init__(self):
        super().__init__('environment_evaluator')

        # Subscribers
        self.pose_sub = self.create_subscription(
            PoseStamped, '/robot_pose', self.pose_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.cmd_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_callback, 10)

        # Publishers
        self.completion_pub = self.create_publisher(Bool, '/task_completion', 10)
        self.performance_pub = self.create_publisher(Float32, '/performance_score', 10)
        self.collision_pub = self.create_publisher(Bool, '/collision_detected', 10)

        # Parameters
        self.declare_parameter('goal_position', [5.0, 0.0])
        self.declare_parameter('navigation_timeout', 300)  # 5 minutes
        self.declare_parameter('min_safe_distance', 0.3)

        # State tracking
        self.current_pose = None
        self.start_time = self.get_clock().now()
        self.collision_threshold = self.get_parameter('min_safe_distance').value
        self.goal_position = self.get_parameter('goal_position').value
        self.cmd_history = deque(maxlen=100)
        self.pose_history = deque(maxlen=200)

    def pose_callback(self, msg):
        self.current_pose = msg.pose
        self.pose_history.append((msg.header.stamp, msg.pose))

        if self.current_pose:
            self.evaluate_performance()

    def scan_callback(self, msg):
        # Check for collisions based on laser scan
        min_distance = min(msg.ranges) if msg.ranges else float('inf')

        collision_msg = Bool()
        collision_msg.data = min_distance < self.collision_threshold
        self.collision_pub.publish(collision_msg)

    def cmd_callback(self, msg):
        self.cmd_history.append(msg)

    def evaluate_performance(self):
        if not self.current_pose:
            return

        # Calculate distance to goal
        pos = self.current_pose.position
        distance_to_goal = np.sqrt(
            (pos.x - self.goal_position[0])**2 +
            (pos.y - self.goal_position[1])**2
        )

        # Check if goal reached
        if distance_to_goal < 0.5:  # 50cm tolerance
            completion_msg = Bool()
            completion_msg.data = True
            self.completion_pub.publish(completion_msg)
            self.get_logger().info(f'Goal reached! Time: {(self.get_clock().now() - self.start_time).nanoseconds / 1e9:.2f}s')
            return

        # Calculate performance score
        # Factors: distance to goal, time taken, collision avoidance, smoothness
        time_elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        distance_score = 1.0 / (1.0 + distance_to_goal)  # Higher score for closer to goal
        time_score = max(0, 1.0 - (time_elapsed / self.get_parameter('navigation_timeout').value))

        # Smoothness score based on command history
        smoothness_score = self.calculate_smoothness_score()

        performance_score = (distance_score * 0.4 + time_score * 0.3 + smoothness_score * 0.3)

        perf_msg = Float32()
        perf_msg.data = performance_score
        self.performance_pub.publish(perf_msg)

    def calculate_smoothness_score(self):
        if len(self.cmd_history) < 2:
            return 1.0

        # Calculate command variation (lower variation = smoother)
        linear_vels = [cmd.linear.x for cmd in self.cmd_history]
        angular_vels = [cmd.angular.z for cmd in self.cmd_history]

        linear_variation = np.std(linear_vels) if len(set(linear_vels)) > 1 else 0
        angular_variation = np.std(angular_vels) if len(set(angular_vels)) > 1 else 0

        # Normalize to 0-1 range (lower variation = higher score)
        smoothness = 1.0 / (1.0 + linear_variation + angular_variation)
        return min(smoothness, 1.0)

def main(args=None):
    rclpy.init(args=args)
    evaluator = EnvironmentEvaluator()

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

## Multi-Environment Testing Framework

### Environment Switching System

```python
# File: environment_evaluation/environment_manager.py
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from std_msgs.msg import String
import subprocess
import time

class EnvironmentManager(Node):
    def __init__(self):
        super().__init__('environment_manager')

        # Services for environment control
        self.switch_env_srv = self.create_service(
            Trigger, 'switch_environment', self.switch_environment_callback)
        self.reset_env_srv = self.create_service(
            Trigger, 'reset_environment', self.reset_environment_callback)

        # Publisher for environment status
        self.status_pub = self.create_publisher(String, '/environment_status', 10)

        # Available environments
        self.environments = {
            'simple_navigation': 'worlds/simple_nav.sdf',
            'complex_navigation': 'worlds/complex_nav.sdf',
            'manipulation': 'worlds/manipulation.sdf',
            'dynamic_obstacles': 'worlds/dynamic.sdf'
        }

        self.current_env = None

    def switch_environment_callback(self, request, response):
        # This would typically involve restarting Gazebo with a new world
        response.success = True
        response.message = f"Environment switching not implemented in this example"
        return response

    def reset_environment_callback(self, request, response):
        # Reset environment to initial state
        response.success = True
        response.message = "Environment reset complete"
        return response

def main(args=None):
    rclpy.init(args=args)
    manager = EnvironmentManager()

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

## Physical AI Environment Considerations

### Real-World Scenario Replication

When building environments for Physical AI:

1. **Deployment Environments**: Match the actual deployment location
2. **Human Interaction**: Include realistic human movement patterns
3. **Environmental Variations**: Day/night, weather, lighting changes
4. **Dynamic Elements**: Moving obstacles, changing layouts

### Safety and Validation

```python
# Example: Safety validation system
class SafetyValidator(Node):
    def __init__(self):
        super().__init__('safety_validator')

        self.pose_sub = self.create_subscription(
            PoseStamped, '/robot_pose', self.pose_callback, 10)
        self.safety_pub = self.create_publisher(Bool, '/safety_violation', 10)

        # Define safety zones
        self.safety_zones = [
            {'center': [0, 0], 'radius': 2.0, 'name': 'restricted_area'},
            {'center': [5, 5], 'radius': 1.0, 'name': 'no_entry_zone'}
        ]

    def pose_callback(self, msg):
        pos = msg.pose.position
        robot_pos = [pos.x, pos.y]

        for zone in self.safety_zones:
            distance = np.sqrt((robot_pos[0] - zone['center'][0])**2 +
                             (robot_pos[1] - zone['center'][1])**2)

            if distance < zone['radius']:
                self.get_logger().warn(f'Safety violation in {zone["name"]}')
                safety_msg = Bool()
                safety_msg.data = True
                self.safety_pub.publish(safety_msg)
                return
```

## Best Practices for Environment Design

### Modularity
- Create reusable environment components
- Use parameterized environments for flexibility
- Implement environment inheritance systems

### Scalability
- Design environments that can be easily modified
- Use configuration files for environment parameters
- Implement progressive difficulty systems

### Validation
- Validate environments against real-world scenarios
- Test with multiple robot configurations
- Measure computational performance requirements

## Chapter Summary

Building comprehensive test environments is crucial for Physical AI development. This chapter covered creating modular environments in both Gazebo and Unity, implementing performance metrics systems, and designing validation frameworks. Well-designed environments enable thorough testing of Physical AI systems before deployment to real hardware.

## Exercises

1. Create a Gazebo world with multiple rooms and obstacles for navigation testing.
2. Implement a Unity scene with dynamic elements that change over time.
3. Design an evaluation system that measures robot performance in your environments.

## Next Steps

In the next chapter, we'll work on a project that integrates all the simulation concepts learned in Module 2, creating an autonomous navigation system in simulation.