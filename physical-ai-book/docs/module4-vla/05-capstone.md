---
sidebar_position: 5
---

# CAPSTONE: Autonomous Humanoid Assistant

## Project Objectives

By completing this CAPSTONE project, you will:
- Integrate all VLA (Vision-Language-Action) concepts into a complete Physical AI system
- Build an autonomous humanoid assistant capable of natural interaction
- Implement multimodal perception, reasoning, and action execution
- Demonstrate advanced Physical AI capabilities in simulation and/or reality
- Validate the complete pipeline from voice commands to physical actions

## Project Overview

In this CAPSTONE project, we'll build an autonomous humanoid assistant that can:
1. Understand natural language voice commands
2. Perceive and understand its environment through vision
3. Reason about complex tasks and their execution
4. Navigate and manipulate objects in physical space
5. Provide natural feedback and maintain conversation context

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Voice Input   │    │  Vision Input   │    │   LLM Reasoning │
│   (Speech Rec)  │    │  (Cameras,      │    │   (GPT-4,       │
│                 │    │  Perception)     │    │   Cognitive     │
└─────────────────┘    └─────────────────┘    │   Planning)     │
         │                       │             └─────────────────┘
         └───────────────────────┼───────────────────┘
                                 │
                    ┌─────────────────┐
                    │  Multimodal     │
                    │  Fusion &       │
                    │  Decision Making │
                    └─────────────────┘
                                 │
                    ┌─────────────────┐
                    │  Action Planner │
                    │  & Executor     │
                    └─────────────────┘
                                 │
                    ┌─────────────────┐
                    │  Robot Control  │
                    │  (Navigation,   │
                    │   Manipulation) │
                    └─────────────────┘
```

## Implementation Steps

### Step 1: Create the Project Package

```bash
# Create project workspace
mkdir -p ~/humanoid_assistant_ws/src
cd ~/humanoid_assistant_ws/src

# Create humanoid assistant package
ros2 pkg create --build-type ament_python humanoid_assistant
cd humanoid_assistant
```

### Step 2: Install Dependencies

```bash
# Install required packages
pip3 install openai transformers torch torchvision torchaudio
pip3 install speechrecognition pyaudio vosk
pip3 install opencv-python numpy scipy
pip3 install webrtcvad

# Install ROS2 packages
sudo apt update
sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup
sudo apt install ros-humble-isaac-ros-*  # If using Isaac
```

### Step 3: Main System Node

```python
# File: humanoid_assistant/humanoid_assistant/main_system.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist, PoseStamped
from vision_msgs.msg import Detection2DArray
import json
import threading
import time
from typing import Dict, Any, Optional

class HumanoidAssistantNode(Node):
    def __init__(self):
        super().__init__('humanoid_assistant_node')

        # Initialize components
        self.voice_processor = VoiceProcessor(self)
        self.vision_processor = VisionProcessor(self)
        self.reasoning_engine = ReasoningEngine(self)
        self.action_executor = ActionExecutor(self)

        # State management
        self.system_state = {
            'current_task': None,
            'task_status': 'idle',
            'conversation_context': [],
            'environment_map': {},
            'last_interaction': time.time()
        }

        # Publishers
        self.status_pub = self.create_publisher(String, '/assistant/status', 10)
        self.feedback_pub = self.create_publisher(String, '/assistant/feedback', 10)

        # Subscribers
        self.voice_cmd_sub = self.create_subscription(
            String, '/voice/command', self.voice_command_callback, 10)
        self.vision_data_sub = self.create_subscription(
            Detection2DArray, '/object/detections', self.vision_callback, 10)
        self.nav_status_sub = self.create_subscription(
            String, '/navigation/status', self.navigation_callback, 10)

        # Timer for system monitoring
        self.monitor_timer = self.create_timer(1.0, self.system_monitor)

        self.get_logger().info('Humanoid Assistant System Started')

    def voice_command_callback(self, msg):
        """Process voice command through the complete pipeline"""
        command = msg.data
        self.get_logger().info(f'Received voice command: {command}')

        # Add to conversation context
        self.system_state['conversation_context'].append({
            'type': 'user_input',
            'content': command,
            'timestamp': time.time()
        })

        # Process through VLA pipeline
        self.process_voice_command(command)

    def vision_callback(self, msg):
        """Process vision data"""
        # Update environment map with detected objects
        objects = []
        for detection in msg.detections:
            if detection.results:
                obj_info = {
                    'class': detection.results[0].hypothesis.class_id,
                    'confidence': detection.results[0].hypothesis.score,
                    'bbox': {
                        'x': detection.bbox.center.x,
                        'y': detection.bbox.center.y,
                        'width': detection.bbox.size_x,
                        'height': detection.bbox.size_y
                    }
                }
                objects.append(obj_info)

        self.system_state['environment_map']['objects'] = objects

    def navigation_callback(self, msg):
        """Process navigation status"""
        try:
            nav_status = json.loads(msg.data)
            self.system_state['navigation_status'] = nav_status
        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in navigation status')

    def process_voice_command(self, command: str):
        """Process voice command through complete VLA pipeline"""
        self.system_state['task_status'] = 'processing'

        # 1. Natural Language Understanding
        self.get_logger().info('Understanding command...')
        parsed_command = self.reasoning_engine.parse_command(command)

        # 2. Context Integration
        self.get_logger().info('Integrating context...')
        contextual_command = self.reasoning_engine.integrate_context(
            parsed_command, self.system_state
        )

        # 3. Task Planning
        self.get_logger().info('Planning task...')
        action_plan = self.reasoning_engine.plan_task(contextual_command)

        # 4. Action Execution
        self.get_logger().info('Executing plan...')
        self.system_state['current_task'] = contextual_command
        self.system_state['task_status'] = 'executing'

        execution_result = self.action_executor.execute_plan(action_plan)

        # 5. Feedback Generation
        self.get_logger().info('Generating feedback...')
        feedback = self.reasoning_engine.generate_feedback(
            contextual_command, execution_result
        )

        # Publish feedback
        feedback_msg = String()
        feedback_msg.data = feedback
        self.feedback_pub.publish(feedback_msg)

        # Update system state
        self.system_state['task_status'] = 'completed'
        self.system_state['last_interaction'] = time.time()

        self.get_logger().info(f'Task completed. Feedback: {feedback}')

    def system_monitor(self):
        """Monitor system status and publish updates"""
        status_msg = String()
        status_msg.data = json.dumps({
            'state': self.system_state['task_status'],
            'current_task': self.system_state['current_task'],
            'last_interaction': self.system_state['last_interaction'],
            'active_components': {
                'voice': self.voice_processor.is_active(),
                'vision': self.vision_processor.is_active(),
                'reasoning': self.reasoning_engine.is_active(),
                'action': self.action_executor.is_active()
            }
        })
        self.status_pub.publish(status_msg)

class VoiceProcessor:
    def __init__(self, node):
        self.node = node
        self.active = True

    def is_active(self):
        return self.active

class VisionProcessor:
    def __init__(self, node):
        self.node = node
        self.active = True

    def is_active(self):
        return self.active

class ReasoningEngine:
    def __init__(self, node):
        self.node = node
        self.active = True

    def is_active(self):
        return self.active

    def parse_command(self, command: str) -> Dict[str, Any]:
        """Parse natural language command"""
        # In real implementation, this would use NLP/LLM processing
        return {
            'original': command,
            'intent': self.classify_intent(command),
            'entities': self.extract_entities(command),
            'action_type': self.determine_action_type(command)
        }

    def classify_intent(self, command: str) -> str:
        """Classify the intent of the command"""
        command_lower = command.lower()

        if any(word in command_lower for word in ['go to', 'navigate', 'move to', 'bring me']):
            return 'navigation'
        elif any(word in command_lower for word in ['pick', 'grasp', 'take', 'get']):
            return 'manipulation'
        elif any(word in command_lower for word in ['find', 'look', 'see', 'where']):
            return 'perception'
        else:
            return 'communication'

    def extract_entities(self, command: str) -> Dict[str, str]:
        """Extract entities from command"""
        # Simple entity extraction
        entities = {}
        words = command.lower().split()

        # Look for objects
        object_keywords = ['cup', 'bottle', 'book', 'phone', 'laptop', 'box']
        for word in words:
            if word in object_keywords:
                entities['object'] = word
                break

        # Look for locations
        location_keywords = ['kitchen', 'living room', 'bedroom', 'office', 'table', 'chair']
        for word in words:
            if word in location_keywords:
                entities['location'] = word
                break

        return entities

    def determine_action_type(self, command: str) -> str:
        """Determine the primary action type"""
        intent = self.classify_intent(command)

        if intent == 'navigation':
            if 'bring me' in command.lower():
                return 'fetch_object'
            else:
                return 'navigate'
        elif intent == 'manipulation':
            return 'manipulate_object'
        elif intent == 'perception':
            return 'perceive_environment'
        else:
            return 'communicate'

    def integrate_context(self, parsed_command: Dict[str, Any], system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate command with system context"""
        contextual_command = parsed_command.copy()
        contextual_command['context'] = {
            'environment': system_state.get('environment_map', {}),
            'previous_interactions': system_state['conversation_context'][-3:],  # Last 3 interactions
            'robot_capabilities': ['navigation', 'basic_manipulation', 'voice_feedback'],
            'current_pose': system_state.get('current_pose')
        }
        return contextual_command

    def plan_task(self, contextual_command: Dict[str, Any]) -> Dict[str, Any]:
        """Plan the task based on contextual command"""
        intent = contextual_command['intent']

        if intent == 'navigation':
            if contextual_command['action_type'] == 'fetch_object':
                # Plan: navigate to object -> pick up -> navigate to user -> deliver
                plan = {
                    'task_type': 'fetch_and_deliver',
                    'steps': [
                        {
                            'action': 'locate_object',
                            'parameters': contextual_command['entities'],
                            'description': 'Locate the requested object'
                        },
                        {
                            'action': 'navigate_to_object',
                            'parameters': contextual_command['entities'],
                            'description': 'Navigate to the object location'
                        },
                        {
                            'action': 'grasp_object',
                            'parameters': contextual_command['entities'],
                            'description': 'Grasp the object'
                        },
                        {
                            'action': 'navigate_to_user',
                            'parameters': {},
                            'description': 'Navigate back to the user'
                        },
                        {
                            'action': 'deliver_object',
                            'parameters': {},
                            'description': 'Deliver the object to the user'
                        }
                    ]
                }
            else:
                # Simple navigation task
                plan = {
                    'task_type': 'navigation',
                    'steps': [
                        {
                            'action': 'navigate',
                            'parameters': contextual_command['entities'],
                            'description': f'Navigate to {contextual_command["entities"].get("location", "destination")}'
                        }
                    ]
                }
        elif intent == 'manipulation':
            plan = {
                'task_type': 'manipulation',
                'steps': [
                    {
                        'action': 'locate_object',
                        'parameters': contextual_command['entities'],
                        'description': f'Locate the {contextual_command["entities"].get("object", "object")}'
                    },
                    {
                        'action': 'manipulate_object',
                        'parameters': contextual_command['entities'],
                        'description': f'Manipulate the {contextual_command["entities"].get("object", "object")}'
                    }
                ]
            }
        else:
            # Default communication task
            plan = {
                'task_type': 'communication',
                'steps': [
                    {
                        'action': 'respond',
                        'parameters': {'message': contextual_command['original']},
                        'description': 'Provide appropriate response'
                    }
                ]
            }

        return plan

    def generate_feedback(self, contextual_command: Dict[str, Any], execution_result: Dict[str, Any]) -> str:
        """Generate natural language feedback"""
        if execution_result.get('success', False):
            if contextual_command['action_type'] == 'fetch_object':
                return "I've successfully brought you the item you requested."
            elif contextual_command['action_type'] == 'navigate':
                return "I've reached the destination successfully."
            else:
                return "Task completed successfully."
        else:
            error_msg = execution_result.get('error', 'Unknown error occurred')
            return f"I'm sorry, but I encountered an issue: {error_msg}. Could you please rephrase your request?"

class ActionExecutor:
    def __init__(self, node):
        self.node = node
        self.active = True

        # Create publishers for different action types
        self.nav_goal_pub = node.create_publisher(PoseStamped, '/goal_pose', 10)
        self.cmd_vel_pub = node.create_publisher(Twist, '/cmd_vel', 10)
        self.manipulation_pub = node.create_publisher(String, '/manipulation/command', 10)

    def is_active(self):
        return self.active

    def execute_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the planned task"""
        success = True
        error_msg = None

        try:
            for step in plan['steps']:
                step_success = self.execute_step(step)
                if not step_success:
                    success = False
                    error_msg = f"Failed at step: {step['description']}"
                    break

        except Exception as e:
            success = False
            error_msg = str(e)

        return {
            'success': success,
            'error': error_msg,
            'completed_steps': len([s for s in plan['steps'] if self.execute_step(s, dry_run=True)]),
            'total_steps': len(plan['steps'])
        }

    def execute_step(self, step: Dict[str, Any], dry_run: bool = False) -> bool:
        """Execute a single step"""
        action = step['action']

        if dry_run:
            # For dry run, just check if action is valid
            return action in ['locate_object', 'navigate_to_object', 'grasp_object',
                             'navigate_to_user', 'deliver_object', 'navigate', 'manipulate_object', 'respond']

        self.node.get_logger().info(f'Executing step: {step["description"]}')

        if action == 'navigate':
            return self.execute_navigation(step['parameters'])
        elif action == 'grasp_object':
            return self.execute_grasp(step['parameters'])
        elif action == 'locate_object':
            return self.execute_perception(step['parameters'])
        elif action == 'respond':
            return self.execute_communication(step['parameters'])
        else:
            self.node.get_logger().warn(f'Unknown action: {action}')
            return False

    def execute_navigation(self, params: Dict[str, Any]) -> bool:
        """Execute navigation action"""
        # In real implementation, this would send navigation goals
        # For simulation, we'll just return success
        self.node.get_logger().info(f'Navigating to {params}')

        # Publish a dummy navigation goal
        goal_msg = PoseStamped()
        goal_msg.header.stamp = self.node.get_clock().now().to_msg()
        goal_msg.header.frame_id = 'map'
        goal_msg.pose.position.x = 1.0  # Example coordinates
        goal_msg.pose.position.y = 1.0
        goal_msg.pose.orientation.w = 1.0

        self.nav_goal_pub.publish(goal_msg)
        return True

    def execute_grasp(self, params: Dict[str, Any]) -> bool:
        """Execute grasp action"""
        # In real implementation, this would send manipulation commands
        self.node.get_logger().info(f'Grasping object: {params}')

        # Publish manipulation command
        manip_msg = String()
        manip_msg.data = json.dumps({
            'action': 'grasp',
            'object': params.get('object', 'unknown')
        })
        self.manipulation_pub.publish(manip_msg)
        return True

    def execute_perception(self, params: Dict[str, Any]) -> bool:
        """Execute perception action"""
        # In real implementation, this would trigger object detection
        self.node.get_logger().info(f'Perceiving environment for: {params}')
        return True

    def execute_communication(self, params: Dict[str, Any]) -> bool:
        """Execute communication action"""
        # In real implementation, this would trigger speech synthesis
        message = params.get('message', 'Hello')
        self.node.get_logger().info(f'Communicating: {message}')
        return True

def main(args=None):
    rclpy.init(args=args)
    node = HumanoidAssistantNode()

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

### Step 4: Voice Processing Component

```python
# File: humanoid_assistant/humanoid_assistant/voice_processor.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
import speech_recognition as sr
import threading
import queue
import time

class VoiceProcessorNode(Node):
    def __init__(self):
        super().__init__('voice_processor_node')

        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Configure recognizer
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True

        # Audio processing
        self.listening = False
        self.audio_queue = queue.Queue()
        self.command_queue = queue.Queue()

        # Publishers
        self.command_pub = self.create_publisher(String, '/voice/command', 10)
        self.listening_pub = self.create_publisher(Bool, '/voice/listening', 10)

        # Subscribers
        self.wake_word_sub = self.create_subscription(
            Bool, '/voice/wake_word_detected', self.wake_word_callback, 10)

        # Start audio processing thread
        self.audio_thread = threading.Thread(target=self.audio_processing_loop, daemon=True)
        self.audio_thread.start()

        # Timer for voice activity
        self.voice_timer = self.create_timer(0.5, self.check_voice_activity)

        self.get_logger().info('Voice Processor Node Started')

    def wake_word_callback(self, msg):
        """Handle wake word detection"""
        if msg.data:
            self.listening = True
            self.get_logger().info('Listening activated by wake word')

    def audio_processing_loop(self):
        """Main audio processing loop"""
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)

        while rclpy.ok():
            if self.listening:
                try:
                    with self.microphone as source:
                        # Listen for audio with timeout
                        audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)

                    # Recognize speech
                    try:
                        command = self.recognizer.recognize_google(audio).lower()
                        self.get_logger().info(f'Recognized: {command}')

                        # Publish command
                        cmd_msg = String()
                        cmd_msg.data = command
                        self.command_pub.publish(cmd_msg)

                        # Deactivate listening after successful recognition
                        self.listening = False

                    except sr.UnknownValueError:
                        self.get_logger().info('Could not understand audio')
                    except sr.RequestError as e:
                        self.get_logger().error(f'Speech recognition error: {e}')

                except sr.WaitTimeoutError:
                    # No speech detected, continue listening
                    pass
                except Exception as e:
                    self.get_logger().error(f'Audio processing error: {e}')

            time.sleep(0.1)  # Small delay to prevent busy waiting

    def check_voice_activity(self):
        """Check and publish listening status"""
        listening_msg = Bool()
        listening_msg.data = self.listening
        self.listening_pub.publish(listening_msg)

def main(args=None):
    rclpy.init(args=args)
    node = VoiceProcessorNode()

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

### Step 5: Vision Processing Component

```python
# File: humanoid_assistant/humanoid_assistant/vision_processor.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D
from std_msgs.msg import String
import cv2
from cv_bridge import CvBridge
import numpy as np
from typing import List, Dict, Any

class VisionProcessorNode(Node):
    def __init__(self):
        super().__init__('vision_processor_node')

        # Initialize CV bridge
        self.cv_bridge = CvBridge()

        # Publishers
        self.detections_pub = self.create_publisher(Detection2DArray, '/object/detections', 10)
        self.visualization_pub = self.create_publisher(Image, '/vision/visualization', 10)

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)

        # State
        self.current_image = None

        self.get_logger().info('Vision Processor Node Started')

    def image_callback(self, msg):
        """Process incoming images for object detection"""
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            self.current_image = cv_image

            # Perform object detection (simplified - in real system would use actual detector)
            detections = self.detect_objects(cv_image)

            # Create detection array message
            detection_array = Detection2DArray()
            detection_array.header = msg.header

            for detection in detections:
                detection_msg = Detection2D()
                detection_msg.header = msg.header
                detection_msg.bbox.center.x = detection['bbox']['center_x']
                detection_msg.bbox.center.y = detection['bbox']['center_y']
                detection_msg.bbox.size_x = detection['bbox']['width']
                detection_msg.bbox.size_y = detection['bbox']['height']

                # Add result
                from vision_msgs.msg import ObjectHypothesisWithPose
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = detection['class']
                hypothesis.hypothesis.score = detection['confidence']
                detection_msg.results.append(hypothesis)

                detection_array.detections.append(detection_msg)

            # Publish detections
            self.detections_pub.publish(detection_array)

            # Create and publish visualization
            viz_image = self.create_visualization(cv_image, detections)
            viz_msg = self.cv_bridge.cv2_to_imgmsg(viz_image, "bgr8")
            viz_msg.header = msg.header
            self.visualization_pub.publish(viz_msg)

        except Exception as e:
            self.get_logger().error(f'Vision processing error: {e}')

    def detect_objects(self, image):
        """Detect objects in image (simplified implementation)"""
        # In a real system, this would use a trained object detection model
        # For this example, we'll simulate detections

        # Convert to grayscale for simple blob detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply threshold to find potential objects
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Find contours (potential objects)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Filter small contours
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)

                # Determine object class based on shape/size
                aspect_ratio = w / h
                if 0.8 <= aspect_ratio <= 1.2:
                    obj_class = "object"  # Could be cup, box, etc.
                elif aspect_ratio > 1.5:
                    obj_class = "book"  # Could be book, paper
                else:
                    obj_class = "object"

                detection = {
                    'class': obj_class,
                    'confidence': min(0.9, area / 10000),  # Normalize confidence
                    'bbox': {
                        'center_x': x + w / 2,
                        'center_y': y + h / 2,
                        'width': w,
                        'height': h
                    }
                }
                detections.append(detection)

        return detections

    def create_visualization(self, image, detections):
        """Create visualization with bounding boxes"""
        viz_image = image.copy()

        for detection in detections:
            bbox = detection['bbox']
            x = int(bbox['center_x'] - bbox['width'] / 2)
            y = int(bbox['center_y'] - bbox['height'] / 2)
            w = int(bbox['width'])
            h = int(bbox['height'])

            # Draw bounding box
            cv2.rectangle(viz_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Add label
            label = f"{detection['class']}: {detection['confidence']:.2f}"
            cv2.putText(viz_image, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return viz_image

def main(args=None):
    rclpy.init(args=args)
    node = VisionProcessorNode()

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

### Step 6: Main Launch File

```python
# File: humanoid_assistant/launch/humanoid_assistant.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os

def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    enable_voice = LaunchConfiguration('enable_voice', default='true')
    enable_vision = LaunchConfiguration('enable_vision', default='true')

    # Main system node
    main_system = Node(
        package='humanoid_assistant',
        executable='main_system',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Voice processing node
    voice_processor = Node(
        package='humanoid_assistant',
        executable='voice_processor',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen',
        condition=IfCondition(enable_voice)
    )

    # Vision processing node
    vision_processor = Node(
        package='humanoid_assistant',
        executable='vision_processor',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen',
        condition=IfCondition(enable_vision)
    )

    # Navigation stack (if using Nav2)
    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('nav2_bringup'),
                'launch',
                'navigation_launch.py'
            ])
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time
        }.items()
    )

    # RViz for visualization
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', PathJoinSubstitution([
            FindPackageShare('humanoid_assistant'),
            'rviz',
            'assistant_view.rviz'
        ])],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(LaunchConfiguration('enable_rviz', default='true'))
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time'
        ),
        DeclareLaunchArgument(
            'enable_voice',
            default_value='true',
            description='Enable voice processing'
        ),
        DeclareLaunchArgument(
            'enable_vision',
            default_value='true',
            description='Enable vision processing'
        ),
        DeclareLaunchArgument(
            'enable_rviz',
            default_value='true',
            description='Enable RViz visualization'
        ),
        main_system,
        voice_processor,
        vision_processor,
        navigation,
        rviz
    ])

# Add the missing imports
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.actions import IncludeLaunchDescription
```

### Step 7: Configuration Files

```yaml
# File: humanoid_assistant/config/assistant_params.yaml
humanoid_assistant:
  ros__parameters:
    # Voice processing parameters
    voice:
      wake_word: "hey assistant"
      sensitivity: 0.5
      response_delay: 0.5  # seconds

    # Vision processing parameters
    vision:
      detection_threshold: 0.7
      tracking_enabled: true
      fov_horizontal: 60  # degrees
      fov_vertical: 45

    # Navigation parameters
    navigation:
      max_speed: 0.5  # m/s
      min_distance: 0.3  # m from obstacles
      goal_tolerance: 0.2  # m

    # Interaction parameters
    interaction:
      max_conversation_length: 10  # turns
      context_window: 30  # seconds
      feedback_enabled: true
```

### Step 8: Package Configuration

Update the package.xml file:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>humanoid_assistant</name>
  <version>0.0.0</version>
  <description>Autonomous humanoid assistant using VLA (Vision-Language-Action)</description>
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

package_name = 'humanoid_assistant'

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
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='Autonomous humanoid assistant using VLA (Vision-Language-Action)',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'main_system = humanoid_assistant.main_system:main',
            'voice_processor = humanoid_assistant.voice_processor:main',
            'vision_processor = humanoid_assistant.vision_processor:main',
        ],
    },
)
```

### Step 9: Running the System

To run the complete humanoid assistant system:

```bash
# Build the package
cd ~/humanoid_assistant_ws
colcon build --packages-select humanoid_assistant
source install/setup.bash

# Run the complete system
ros2 launch humanoid_assistant humanoid_assistant.launch.py

# In another terminal, send voice commands
echo "{'data': 'Please go to the kitchen and bring me a cup'}" | ros2 topic pub /voice/command std_msgs/String --once
```

## Physical AI Concepts Demonstrated

### 1. Multimodal Integration
- Voice commands processed through natural language understanding
- Visual perception for environment awareness
- Action execution based on integrated understanding

### 2. Cognitive Reasoning
- Task decomposition from high-level commands
- Context-aware decision making
- Adaptive behavior based on environment

### 3. Real-time Processing
- Synchronized processing of multiple modalities
- Real-time response to user commands
- Continuous environment monitoring

### 4. Human-Robot Interaction
- Natural language interface
- Contextual conversation management
- Appropriate feedback and responses

## Advanced Features Implementation

### Context Management

```python
# Enhanced context manager for the assistant
class ContextManager:
    def __init__(self, max_history=10):
        self.conversation_history = []
        self.object_memory = {}
        self.location_memory = {}
        self.max_history = max_history

    def update_conversation(self, user_input, system_response):
        """Update conversation history"""
        entry = {
            'timestamp': time.time(),
            'user_input': user_input,
            'system_response': system_response,
            'follow_up': self.extract_follow_up_info(user_input)
        }

        self.conversation_history.append(entry)

        # Maintain history size
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)

    def extract_follow_up_info(self, text):
        """Extract information that might be referenced later"""
        # Look for demonstratives, spatial references, etc.
        follow_up_indicators = ['this', 'that', 'these', 'those', 'it', 'there', 'here']
        words = text.lower().split()

        # In a real system, this would use more sophisticated NLP
        return [word for word in words if word in follow_up_indicators]

    def resolve_references(self, command):
        """Resolve pronouns and references in command"""
        # Simple reference resolution
        if 'it' in command.lower() and self.conversation_history:
            # Assume 'it' refers to the last mentioned object
            last_response = self.conversation_history[-1]['system_response']
            # Extract object from last response
            import re
            obj_match = re.search(r'(cup|bottle|book|object)', last_response.lower())
            if obj_match:
                return command.lower().replace('it', obj_match.group(1))

        return command
```

### Safety and Validation

```python
# Safety validation for the assistant
class SafetyValidator:
    def __init__(self):
        self.safety_rules = {
            'navigation': [
                lambda goal: self.check_navigation_safety(goal),
                lambda goal: self.check_accessible(goal)
            ],
            'manipulation': [
                lambda obj: self.check_manipulation_safety(obj),
                lambda obj: self.check_object_safety(obj)
            ],
            'communication': [
                lambda msg: self.check_appropriate_content(msg)
            ]
        }

    def validate_action(self, action_type, parameters):
        """Validate action against safety rules"""
        if action_type in self.safety_rules:
            for rule in self.safety_rules[action_type]:
                try:
                    if not rule(parameters):
                        return False, f"Action violates safety rule for {action_type}"
                except Exception as e:
                    return False, f"Safety check error: {e}"

        return True, "Action is safe"

    def check_navigation_safety(self, goal):
        """Check if navigation goal is safe"""
        # In real system, check against map, obstacles, restricted areas
        return True  # Placeholder

    def check_manipulation_safety(self, obj):
        """Check if manipulation is safe"""
        # In real system, check object properties, location, etc.
        return True  # Placeholder
```

## Performance Optimization

### Caching and Efficiency

```python
# Performance optimization for the assistant
from functools import lru_cache
import time

class PerformanceOptimizer:
    def __init__(self):
        self.response_cache = {}
        self.cache_ttl = 300  # 5 minutes

    @lru_cache(maxsize=128)
    def cached_command_processing(self, command_hash):
        """Cache expensive command processing"""
        # This would cache the results of expensive LLM calls
        pass

    def should_cache_response(self, command, response):
        """Determine if response should be cached"""
        # Cache responses to common, factual questions
        common_questions = ['what is your name', 'what can you do', 'hello']
        return any(q in command.lower() for q in common_questions)
```

## Troubleshooting

### Common Issues and Solutions

1. **Voice Recognition Issues**
   - Ensure microphone permissions and quality
   - Adjust energy thresholds based on environment
   - Use noise cancellation techniques

2. **Vision Processing Delays**
   - Optimize image resolution and processing frequency
   - Use GPU acceleration for deep learning models
   - Implement efficient object detection algorithms

3. **LLM Integration Problems**
   - Verify API keys and rate limits
   - Implement proper error handling and fallbacks
   - Optimize prompt engineering for better results

4. **System Integration Issues**
   - Check ROS2 network configuration
   - Verify message type compatibility
   - Monitor system resource usage

## Chapter Summary

The CAPSTONE project demonstrates the complete integration of Vision-Language-Action concepts into a functional Physical AI system. The autonomous humanoid assistant showcases how multiple modalities can be combined to create natural, intuitive human-robot interaction. The system processes voice commands, perceives its environment, reasons about tasks, and executes appropriate actions while maintaining safety and context awareness.

## Exercises

1. Extend the system to handle multi-step commands like "Go to the kitchen, pick up the red cup, and bring it to the living room."
2. Implement more sophisticated object recognition and manipulation capabilities.
3. Add emotional intelligence features that adapt responses based on user tone and context.

## Next Steps

In the next chapter, we'll explore advanced topics and future directions in Physical AI, including emerging technologies, research frontiers, and practical deployment considerations.