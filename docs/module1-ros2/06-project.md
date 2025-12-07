---
sidebar_position: 6
---

# Project: Voice-Controlled Robot Arm

## Project Objectives

By completing this project, you will:
- Integrate multiple ROS 2 concepts into a cohesive Physical AI application
- Create a voice recognition system that controls robot movements
- Implement sensor feedback for safe operation
- Design a modular system architecture for Physical AI applications
- Practice debugging and testing techniques for robotic systems

## Project Overview

In this project, we'll build a voice-controlled robot arm that responds to spoken commands. The system will:
1. Listen for voice commands using speech recognition
2. Parse commands to determine desired arm movements
3. Execute movements using ROS 2 control interfaces
4. Provide visual and audio feedback
5. Include safety mechanisms to prevent dangerous movements

## System Architecture

```
Voice Input → Speech Recognition → Command Parser → Motion Planning → Robot Control → Feedback
     ↑                                                                                     ↓
     └─────────────────────────────── Safety Monitor ←─────────────────────────────────────┘
```

## Implementation Steps

### Step 1: Create the Project Package

```bash
# Create the project workspace
mkdir -p ~/voice_robot_ws/src
cd ~/voice_robot_ws/src

# Create the package
ros2 pkg create --build-type ament_python voice_robot_control
cd voice_robot_control
```

### Step 2: Install Dependencies

```bash
pip3 install speechrecognition pyaudio numpy transforms3d
```

### Step 3: Voice Recognition Node

```python
# File: voice_robot_control/voice_robot_control/voice_recognition.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import speech_recognition as sr
import threading
import queue

class VoiceRecognitionNode(Node):
    def __init__(self):
        super().__init__('voice_recognition_node')
        self.publisher = self.create_publisher(String, 'voice_command', 10)

        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Set up for continuous listening
        self.listening = True
        self.command_queue = queue.Queue()

        # Start voice recognition thread
        self.voice_thread = threading.Thread(target=self.listen_continuously)
        self.voice_thread.daemon = True
        self.voice_thread.start()

        # Timer to process recognized commands
        self.timer = self.create_timer(0.1, self.process_commands)

        self.get_logger().info('Voice Recognition Node Started')

    def listen_continuously(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)

        while self.listening:
            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=3)

                # Use Google's speech recognition
                command = self.recognizer.recognize_google(audio).lower()
                self.command_queue.put(command)

            except sr.WaitTimeoutError:
                pass  # No speech detected, continue listening
            except sr.UnknownValueError:
                self.get_logger().info('Could not understand audio')
            except sr.RequestError as e:
                self.get_logger().error(f'Speech recognition error: {e}')

    def process_commands(self):
        while not self.command_queue.empty():
            command = self.command_queue.get()
            self.get_logger().info(f'Recognized command: {command}')

            # Publish the command
            msg = String()
            msg.data = command
            self.publisher.publish(msg)

    def destroy_node(self):
        self.listening = False
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = VoiceRecognitionNode()

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

### Step 4: Command Parser Node

```python
# File: voice_robot_control/voice_robot_control/command_parser.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState
import re

class CommandParserNode(Node):
    def __init__(self):
        super().__init__('command_parser_node')

        # Subscribers
        self.voice_sub = self.create_subscription(
            String, 'voice_command', self.voice_callback, 10)

        # Publishers
        self.goal_pub = self.create_publisher(Point, 'arm_goal', 10)
        self.joint_pub = self.create_publisher(JointState, 'joint_commands', 10)
        self.feedback_pub = self.create_publisher(String, 'system_feedback', 10)

        self.get_logger().info('Command Parser Node Started')

    def voice_callback(self, msg):
        command = msg.data.lower()
        self.get_logger().info(f'Processing command: {command}')

        # Parse different types of commands
        if 'move arm to' in command:
            self.parse_position_command(command)
        elif 'pick up' in command:
            self.parse_pickup_command(command)
        elif 'put down' in command:
            self.parse_putdown_command(command)
        elif 'home position' in command:
            self.parse_home_command(command)
        elif 'wave' in command:
            self.parse_wave_command(command)
        else:
            self.send_feedback(f'Unknown command: {command}')

    def parse_position_command(self, command):
        # Extract coordinates from command like "move arm to x 10 y 20 z 30"
        x_match = re.search(r'x\s+([+-]?\d*\.?\d+)', command)
        y_match = re.search(r'y\s+([+-]?\d*\.?\d+)', command)
        z_match = re.search(r'z\s+([+-]?\d*\.?\d+)', command)

        if x_match and y_match and z_match:
            goal = Point()
            goal.x = float(x_match.group(1))
            goal.y = float(y_match.group(1))
            goal.z = float(z_match.group(1))

            self.goal_pub.publish(goal)
            self.send_feedback(f'Moving arm to position: ({goal.x}, {goal.y}, {goal.z})')
        else:
            self.send_feedback('Could not parse coordinates from command')

    def parse_pickup_command(self, command):
        # Parse pickup command
        object_name = command.replace('pick up', '').strip()
        if object_name:
            self.send_feedback(f'Attempting to pick up {object_name}')
            # Publish command to execute pickup sequence
        else:
            self.send_feedback('Pickup command needs object name')

    def parse_putdown_command(self, command):
        # Parse putdown command
        self.send_feedback('Executing put down sequence')
        # Publish command to execute put down sequence

    def parse_home_command(self, command):
        # Move to home position
        home_pos = Point()
        home_pos.x = 0.0
        home_pos.y = 0.0
        home_pos.z = 0.0

        self.goal_pub.publish(home_pos)
        self.send_feedback('Moving to home position')

    def parse_wave_command(self, command):
        # Execute waving motion
        self.send_feedback('Executing wave motion')
        # Publish joint commands for waving motion

    def send_feedback(self, message):
        feedback_msg = String()
        feedback_msg.data = message
        self.feedback_pub.publish(feedback_msg)

def main(args=None):
    rclpy.init(args=args)
    node = CommandParserNode()

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

### Step 5: Motion Planning Node

```python
# File: voice_robot_control/voice_robot_control/motion_planner.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState
from std_msgs.msg import String
import numpy as np
from transforms3d.euler import euler2mat

class MotionPlannerNode(Node):
    def __init__(self):
        super().__init__('motion_planner_node')

        # Subscribers
        self.goal_sub = self.create_subscription(
            Point, 'arm_goal', self.goal_callback, 10)

        # Publishers
        self.joint_pub = self.create_publisher(JointState, 'joint_commands', 10)
        self.feedback_pub = self.create_publisher(String, 'system_feedback', 10)

        # Robot parameters (simplified 3-DOF arm)
        self.link_lengths = [0.5, 0.4, 0.3]  # Link lengths in meters

        self.get_logger().info('Motion Planner Node Started')

    def goal_callback(self, msg):
        self.get_logger().info(f'Planning motion to goal: ({msg.x}, {msg.y}, {msg.z})')

        # Check if goal is reachable
        goal_distance = np.sqrt(msg.x**2 + msg.y**2 + msg.z**2)
        max_reach = sum(self.link_lengths)

        if goal_distance > max_reach:
            self.send_feedback('Goal position is out of reach')
            return

        # Calculate inverse kinematics (simplified)
        joint_angles = self.calculate_inverse_kinematics(msg.x, msg.y, msg.z)

        if joint_angles is not None:
            self.publish_joint_commands(joint_angles)
            self.send_feedback(f'Motion planned: Joint angles {joint_angles}')
        else:
            self.send_feedback('Could not calculate motion to goal position')

    def calculate_inverse_kinematics(self, x, y, z):
        """
        Simplified inverse kinematics for a 3-DOF arm
        This is a basic implementation - real systems would use more sophisticated algorithms
        """
        try:
            # Calculate distance from base to target in XY plane
            r = np.sqrt(x**2 + y**2)

            # Height
            h = z

            # Calculate joint angles using geometric approach
            l1, l2, l3 = self.link_lengths

            # For a 3-DOF arm: base rotation, shoulder, elbow
            theta1 = np.arctan2(y, x)  # Base rotation

            # Calculate remaining distances
            d = np.sqrt(r**2 + h**2)  # Distance from base to target

            # Check if target is reachable
            if d > l2 + l3:
                return None  # Target too far

            if d < abs(l2 - l3):
                return None  # Target too close

            # Calculate shoulder and elbow angles
            cos_theta3 = (l2**2 + l3**2 - d**2) / (2 * l2 * l3)
            if abs(cos_theta3) > 1:
                return None

            theta3 = np.arccos(cos_theta3)  # Elbow angle
            alpha = np.arctan2(h, r)
            beta = np.arccos((l2**2 + d**2 - l3**2) / (2 * l2 * d))

            theta2 = alpha - beta  # Shoulder angle

            # Convert to joint angles
            joint_angles = [theta1, theta2, theta3]

            # Check joint limits (example limits)
            joint_limits = [(-np.pi, np.pi), (-np.pi/2, np.pi/2), (-np.pi, np.pi)]

            for i, (angle, (min_limit, max_limit)) in enumerate(zip(joint_angles, joint_limits)):
                if angle < min_limit or angle > max_limit:
                    self.get_logger().warn(f'Joint {i} angle {angle} exceeds limits')
                    # Adjust or return None based on safety policy
                    return None

            return joint_angles

        except Exception as e:
            self.get_logger().error(f'Error in inverse kinematics: {e}')
            return None

    def publish_joint_commands(self, joint_angles):
        joint_msg = JointState()
        joint_msg.name = ['joint1', 'joint2', 'joint3']
        joint_msg.position = joint_angles
        joint_msg.velocity = [0.0, 0.0, 0.0]  # Start with zero velocity
        joint_msg.effort = [0.0, 0.0, 0.0]

        self.joint_pub.publish(joint_msg)

    def send_feedback(self, message):
        feedback_msg = String()
        feedback_msg.data = message
        self.feedback_pub.publish(feedback_msg)

def main(args=None):
    rclpy.init(args=args)
    node = MotionPlannerNode()

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

### Step 6: Safety Monitor Node

```python
# File: voice_robot_control/voice_robot_control/safety_monitor.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from geometry_msgs.msg import Point
import numpy as np

class SafetyMonitorNode(Node):
    def __init__(self):
        super().__init__('safety_monitor_node')

        # Subscribers
        self.joint_sub = self.create_subscription(
            JointState, 'joint_commands', self.joint_command_callback, 10)
        self.goal_sub = self.create_subscription(
            Point, 'arm_goal', self.goal_command_callback, 10)

        # Publishers
        self.safety_pub = self.create_publisher(String, 'safety_alert', 10)
        self.emergency_pub = self.create_publisher(String, 'emergency_stop', 10)

        # Safety parameters
        self.joint_limits = {
            'joint1': (-2.0, 2.0),   # radians
            'joint2': (-1.5, 1.5),
            'joint3': (-2.5, 2.5)
        }

        self.velocity_limits = [1.0, 1.0, 1.0]  # rad/s

        self.get_logger().info('Safety Monitor Node Started')

    def joint_command_callback(self, msg):
        # Check joint limits
        for i, (name, position) in enumerate(zip(msg.name, msg.position)):
            if name in self.joint_limits:
                min_limit, max_limit = self.joint_limits[name]
                if position < min_limit or position > max_limit:
                    self.send_safety_alert(f'Joint {name} position {position} exceeds limits [{min_limit}, {max_limit}]')

        # Check velocity limits if provided
        if msg.velocity:
            for i, velocity in enumerate(msg.velocity):
                if abs(velocity) > self.velocity_limits[i]:
                    self.send_safety_alert(f'Joint {i} velocity {velocity} exceeds limit {self.velocity_limits[i]}')

    def goal_command_callback(self, msg):
        # Check if goal is in safe workspace
        distance = np.sqrt(msg.x**2 + msg.y**2 + msg.z**2)
        if distance > 2.0:  # Max safe reach
            self.send_safety_alert(f'Goal position ({msg.x}, {msg.y}, {msg.z}) too far: distance {distance:.2f}m')

    def send_safety_alert(self, message):
        self.get_logger().warn(f'SAFETY ALERT: {message}')

        alert_msg = String()
        alert_msg.data = message
        self.safety_pub.publish(alert_msg)

def main(args=None):
    rclpy.init(args=args)
    node = SafetyMonitorNode()

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

### Step 7: Package Configuration

Update `package.xml`:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>voice_robot_control</name>
  <version>0.0.0</version>
  <description>Package for voice-controlled robot arm</description>
  <maintainer email="user@example.com">user</maintainer>
  <license>Apache-2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>sensor_msgs</end>
  <depend>geometry_msgs</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

Update `setup.py`:

```python
from setuptools import setup
import os
from glob import glob

package_name = 'voice_robot_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='Package for voice-controlled robot arm',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'voice_recognition = voice_robot_control.voice_recognition:main',
            'command_parser = voice_robot_control.command_parser:main',
            'motion_planner = voice_robot_control.motion_planner:main',
            'safety_monitor = voice_robot_control.safety_monitor:main',
        ],
    },
)
```

### Step 8: Launch File

Create `launch/voice_robot.launch.py`:

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='voice_robot_control',
            executable='voice_recognition',
            name='voice_recognition_node',
            output='screen',
        ),
        Node(
            package='voice_robot_control',
            executable='command_parser',
            name='command_parser_node',
            output='screen',
        ),
        Node(
            package='voice_robot_control',
            executable='motion_planner',
            name='motion_planner_node',
            output='screen',
        ),
        Node(
            package='voice_robot_control',
            executable='safety_monitor',
            name='safety_monitor_node',
            output='screen',
        ),
    ])
```

## Running the System

### Build and Run

```bash
# Build the package
cd ~/voice_robot_ws
colcon build --packages-select voice_robot_control
source install/setup.bash

# Run the system
ros2 launch voice_robot_control voice_robot.launch.py
```

### Test Commands

Once running, try these voice commands:
- "Move arm to x 0.5 y 0.3 z 0.2"
- "Home position"
- "Wave"
- "Pick up object"

## Physical AI Concepts Demonstrated

### Multi-Modal Interaction
- Voice input processing
- Sensor feedback integration
- Real-time command parsing

### Safety-First Design
- Joint limit monitoring
- Workspace boundary checking
- Emergency stop capabilities

### Modular Architecture
- Separate nodes for different functions
- Clear communication interfaces
- Independent development and testing

## Troubleshooting

### Common Issues and Solutions

1. **Microphone Access**
   - Ensure proper permissions for microphone access
   - Check that PyAudio is properly installed

2. **Speech Recognition Accuracy**
   - Use a quiet environment
   - Speak clearly and at consistent volume
   - Consider using alternative speech recognition APIs

3. **Joint Limit Violations**
   - Verify robot kinematic parameters
   - Check inverse kinematics calculations
   - Adjust joint limits as needed

4. **Communication Issues**
   - Verify topic names and message types
   - Check that all nodes are properly connected
   - Use `ros2 topic list` and `ros2 node list` for debugging

## Chapter Summary

This project integrates multiple ROS 2 concepts into a comprehensive Physical AI application. It demonstrates voice interaction, motion planning, safety monitoring, and modular system design. The voice-controlled robot arm showcases how ROS 2 enables complex Physical AI systems that interact naturally with humans.

## Exercises

1. Add gesture recognition using a camera to complement voice commands.
2. Implement a learning component that improves command recognition over time.
3. Add haptic feedback to enhance the human-robot interaction experience.

## Next Steps

In the next chapter, we'll assess your understanding of ROS 2 concepts through practical challenges and exercises.