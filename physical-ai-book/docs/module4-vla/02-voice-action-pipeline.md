---
sidebar_position: 2
---

# Voice-to-Action Pipeline

## Chapter Objectives

By the end of this chapter, you will be able to:
- Design and implement a complete voice-to-action pipeline for Physical AI
- Integrate speech recognition with LLM processing
- Create robust voice command interpretation systems
- Implement multi-modal feedback mechanisms
- Handle voice command ambiguity and errors gracefully

## Voice-to-Action Architecture

### Complete Pipeline Overview

```
Voice Input → Speech Recognition → Natural Language Processing → LLM Interpretation → Action Planning → Robot Execution → Feedback
     ↑                                                                                                                     ↓
     └─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Pipeline Components

1. **Audio Input**: Microphone array and audio preprocessing
2. **Speech Recognition**: Converting speech to text
3. **Natural Language Understanding**: Parsing and semantic analysis
4. **LLM Processing**: Advanced interpretation and reasoning
5. **Action Planning**: Converting commands to executable actions
6. **Execution**: Physical robot action execution
7. **Feedback**: Audio/visual confirmation and status updates

## Audio Input and Preprocessing

### Audio Capture Configuration

```python
# File: voice_pipeline/audio_capture.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
import pyaudio
import numpy as np
import threading
import queue
import webrtcvad
from collections import deque

class AudioCaptureNode(Node):
    def __init__(self):
        super().__init__('audio_capture_node')

        # Audio configuration
        self.rate = 16000  # Sample rate
        self.chunk_size = 1024  # Samples per chunk
        self.channels = 1  # Mono
        self.format = pyaudio.paInt16  # 16-bit samples

        # Voice Activity Detection
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(2)  # Aggressive VAD

        # Audio buffers
        self.audio_queue = queue.Queue()
        self.voice_buffer = deque(maxlen=32000)  # 2 seconds at 16kHz
        self.listening = False

        # Publishers
        self.audio_text_pub = self.create_publisher(String, '/audio/text', 10)
        self.listening_pub = self.create_publisher(Bool, '/audio/listening', 10)

        # Start audio capture thread
        self.audio_thread = threading.Thread(target=self.capture_audio, daemon=True)
        self.audio_thread.start()

        # Timer for VAD
        self.vad_timer = self.create_timer(0.1, self.check_voice_activity)

        self.get_logger().info('Audio Capture Node Started')

    def capture_audio(self):
        """Capture audio from microphone"""
        audio = pyaudio.PyAudio()

        stream = audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )

        self.get_logger().info('Audio capture started')

        while rclpy.ok():
            try:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)

                # Add to voice buffer for VAD
                self.voice_buffer.extend(audio_data)

                # Put raw audio in queue for processing
                self.audio_queue.put(audio_data)

            except Exception as e:
                self.get_logger().error(f'Audio capture error: {e}')
                break

        stream.stop_stream()
        stream.close()
        audio.terminate()

    def check_voice_activity(self):
        """Check for voice activity and publish listening status"""
        if len(self.voice_buffer) >= 320:  # 20ms at 16kHz
            # Get 20ms chunk for VAD
            chunk = list(self.voice_buffer)[-320:]
            chunk_bytes = np.array(chunk, dtype=np.int16).tobytes()

            try:
                is_speech = self.vad.is_speech(chunk_bytes, self.rate)
                self.listening = is_speech

                # Publish listening status
                listening_msg = Bool()
                listening_msg.data = is_speech
                self.listening_pub.publish(listening_msg)

                if is_speech:
                    self.get_logger().debug('Voice activity detected')
            except Exception as e:
                self.get_logger().error(f'VAD error: {e}')

    def get_audio_chunk(self):
        """Get audio chunk from queue"""
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None

def main(args=None):
    rclpy.init(args=args)
    node = AudioCaptureNode()

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

### Audio Preprocessing

```python
# File: voice_pipeline/audio_preprocessing.py
import numpy as np
from scipy import signal
import librosa

class AudioPreprocessor:
    def __init__(self):
        self.sample_rate = 16000
        self.noise_threshold = 0.01  # Threshold for noise detection

    def preprocess_audio(self, audio_data):
        """Preprocess audio data for better recognition"""
        # Convert to float32
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0

        # Apply noise reduction
        audio_data = self.reduce_noise(audio_data)

        # Normalize audio
        audio_data = self.normalize_audio(audio_data)

        # Apply pre-emphasis filter
        audio_data = self.pre_emphasis_filter(audio_data)

        return audio_data

    def reduce_noise(self, audio_data):
        """Simple noise reduction"""
        # Calculate RMS energy
        rms = np.sqrt(np.mean(audio_data**2))

        # Apply noise threshold
        if rms < self.noise_threshold:
            # Apply noise reduction
            audio_data = self.spectral_subtraction(audio_data)

        return audio_data

    def spectral_subtraction(self, audio_data, n_fft=512):
        """Simple spectral subtraction noise reduction"""
        # This is a simplified version
        # In practice, you'd use more sophisticated noise reduction
        return audio_data

    def normalize_audio(self, audio_data):
        """Normalize audio to consistent level"""
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val
        return audio_data

    def pre_emphasis_filter(self, audio_data, coeff=0.97):
        """Apply pre-emphasis filter"""
        return np.append(audio_data[0], audio_data[1:] - coeff * audio_data[:-1])

    def vad_filter(self, audio_data, threshold=0.01):
        """Voice Activity Detection based on energy"""
        energy = np.mean(audio_data**2)
        return energy > threshold
```

## Speech Recognition Integration

### Online Speech Recognition

```python
# File: voice_pipeline/speech_recognition.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from sensor_msgs.msg import AudioData
import speech_recognition as sr
import threading
import queue
import time

class SpeechRecognitionNode(Node):
    def __init__(self):
        super().__init__('speech_recognition_node')

        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Configure recognizer
        self.recognizer.energy_threshold = 300  # Adjust for environment
        self.recognizer.dynamic_energy_threshold = True

        # Audio buffer
        self.audio_queue = queue.Queue()
        self.listening = False

        # Subscribers
        self.audio_sub = self.create_subscription(
            AudioData, '/audio/raw', self.audio_callback, 10)
        self.listening_sub = self.create_subscription(
            Bool, '/audio/listening', self.listening_callback, 10)

        # Publishers
        self.text_pub = self.create_publisher(String, '/voice/command', 10)

        # Start recognition thread
        self.recognition_thread = threading.Thread(
            target=self.recognition_worker, daemon=True)
        self.recognition_thread.start()

        self.get_logger().info('Speech Recognition Node Started')

    def audio_callback(self, msg):
        """Receive audio data"""
        # Convert AudioData message to AudioData object
        audio_data = sr.AudioData(msg.data, 16000, 2)  # Assuming 16kHz, 16-bit
        self.audio_queue.put(audio_data)

    def listening_callback(self, msg):
        """Update listening state"""
        self.listening = msg.data

    def recognition_worker(self):
        """Process audio in background"""
        while rclpy.ok():
            try:
                if not self.audio_queue.empty() and self.listening:
                    audio_data = self.audio_queue.get(timeout=1.0)

                    try:
                        # Recognize speech
                        text = self.recognizer.recognize_google(audio_data)
                        self.get_logger().info(f'Recognized: {text}')

                        # Publish recognized text
                        text_msg = String()
                        text_msg.data = text
                        self.text_pub.publish(text_msg)

                    except sr.UnknownValueError:
                        self.get_logger().info('Could not understand audio')
                    except sr.RequestError as e:
                        self.get_logger().error(f'Could not request results: {e}')

            except queue.Empty:
                time.sleep(0.1)  # Small delay to prevent busy waiting
            except Exception as e:
                self.get_logger().error(f'Recognition error: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = SpeechRecognitionNode()

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

### Offline Speech Recognition

```python
# File: voice_pipeline/offline_recognition.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import vosk
import json
import wave

class OfflineSpeechRecognitionNode(Node):
    def __init__(self):
        super().__init__('offline_speech_recognition_node')

        # Initialize Vosk model (requires model download)
        # Download model from https://alphacephei.com/vosk/models
        try:
            self.model = vosk.Model("path/to/vosk-model")  # Download model first
            self.recognizer = vosk.KaldiRecognizer(self.model, 16000)
        except:
            self.get_logger().error('Vosk model not found. Please download and configure.')
            self.model = None
            self.recognizer = None

        # Subscribers and publishers
        self.audio_sub = self.create_subscription(
            String, '/audio/processed', self.audio_callback, 10)
        self.text_pub = self.create_publisher(String, '/voice/command', 10)

        self.get_logger().info('Offline Speech Recognition Node Started')

    def audio_callback(self, msg):
        """Process audio for offline recognition"""
        if self.recognizer is None:
            return

        # Process audio chunk
        if self.recognizer.AcceptWaveform(msg.data):
            result = self.recognizer.Result()
            result_dict = json.loads(result)

            if 'text' in result_dict and result_dict['text']:
                # Publish recognized text
                text_msg = String()
                text_msg.data = result_dict['text']
                self.text_pub.publish(text_msg)
                self.get_logger().info(f'Recognized (offline): {result_dict["text"]}')

def main(args=None):
    rclpy.init(args=args)
    node = OfflineSpeechRecognitionNode()

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

## Natural Language Understanding

### Command Parsing

```python
# File: voice_pipeline/nlu_processor.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import re
import json
from typing import Dict, List, Optional

class NLUProcessorNode(Node):
    def __init__(self):
        super().__init__('nlu_processor_node')

        # Subscribers
        self.voice_sub = self.create_subscription(
            String, '/voice/command', self.voice_callback, 10)

        # Publishers
        self.interpretation_pub = self.create_publisher(
            String, '/command/interpretation', 10)

        # Command patterns
        self.command_patterns = {
            'move': [
                r'move\s+(forward|backward|ahead|back)',
                r'go\s+(forward|backward|ahead|back)',
                r'forward',
                r'backward',
                r'back',
                r'ahead'
            ],
            'turn': [
                r'turn\s+(left|right)',
                r'rotate\s+(left|right)',
                r'pivot\s+(left|right)'
            ],
            'stop': [
                r'stop',
                r'halt',
                r'freeze'
            ],
            'pickup': [
                r'pick up',
                r'grab',
                r'take',
                r'collect'
            ],
            'place': [
                r'place',
                r'put',
                r'drop',
                r'release'
            ]
        }

        # Distance patterns
        self.distance_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:meters?|m)',
            r'(\d+(?:\.\d+)?)\s*(?:feet|ft)',
            r'(\d+(?:\.\d+)?)\s*(?:steps?)'
        ]

        # Direction patterns
        self.direction_patterns = [
            r'(\d+(?:\.\d+)?)\s*degrees?',
            r'(\d+(?:\.\d+)?)\s*deg',
            r'quarter turn',
            r'half turn',
            r'full turn'
        ]

        self.get_logger().info('NLU Processor Node Started')

    def voice_callback(self, msg):
        """Process voice command"""
        command_text = msg.data.lower()
        self.get_logger().info(f'Processing command: {command_text}')

        # Parse command
        interpretation = self.parse_command(command_text)

        # Publish interpretation
        interpretation_msg = String()
        interpretation_msg.data = json.dumps(interpretation)
        self.interpretation_pub.publish(interpretation_msg)

    def parse_command(self, text: str) -> Dict:
        """Parse natural language command into structured format"""
        interpretation = {
            'action': None,
            'parameters': {},
            'confidence': 0.0,
            'original_text': text
        }

        # Find main action
        for action, patterns in self.command_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    interpretation['action'] = action
                    interpretation['confidence'] = 0.9
                    break
            if interpretation['action']:
                break

        # Extract parameters
        if interpretation['action']:
            interpretation['parameters'] = self.extract_parameters(text, interpretation['action'])

        return interpretation

    def extract_parameters(self, text: str, action: str) -> Dict:
        """Extract parameters from command text"""
        params = {}

        # Extract distance
        for pattern in self.distance_patterns:
            match = re.search(pattern, text)
            if match:
                distance = float(match.group(1))
                if 'meters' in match.group(0) or 'm' in match.group(0):
                    params['distance'] = distance
                elif 'feet' in match.group(0) or 'ft' in match.group(0):
                    params['distance'] = distance * 0.3048  # Convert to meters
                elif 'step' in match.group(0):
                    params['distance'] = distance * 0.76  # Average step length
                break

        # Extract direction (for turns)
        if action == 'turn':
            for pattern in self.direction_patterns:
                match = re.search(pattern, text)
                if match:
                    if 'quarter turn' in text:
                        params['angle'] = 90
                    elif 'half turn' in text:
                        params['angle'] = 180
                    elif 'full turn' in text:
                        params['angle'] = 360
                    else:
                        params['angle'] = float(match.group(1))
                    break

            # Extract direction (left/right)
            if 'left' in text:
                params['direction'] = 'left'
            elif 'right' in text:
                params['direction'] = 'right'

        # Extract object (for pickup/place)
        if action in ['pickup', 'place']:
            # Look for object names in the text
            object_patterns = [
                r'pick up (.+)',
                r'grab (.+)',
                r'take (.+)',
                r'place (.+)',
                r'put (.+)'
            ]

            for pattern in object_patterns:
                match = re.search(pattern, text)
                if match:
                    params['object'] = match.group(1).strip()
                    break

        return params

def main(args=None):
    rclpy.init(args=args)
    node = NLUProcessorNode()

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

## LLM Integration for Advanced Understanding

### LLM Command Interpreter

```python
# File: voice_pipeline/llm_interpreter.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import openai
from typing import Dict, Any

class LLMInterpreterNode(Node):
    def __init__(self):
        super().__init__('llm_interpreter_node')

        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key='your-api-key')  # Replace with actual key

        # Subscribers
        self.nlu_sub = self.create_subscription(
            String, '/command/interpretation', self.nlu_callback, 10)

        # Publishers
        self.action_plan_pub = self.create_publisher(
            String, '/action/plan', 10)

        self.get_logger().info('LLM Interpreter Node Started')

    def nlu_callback(self, msg):
        """Process NLU output with LLM for advanced interpretation"""
        try:
            interpretation = json.loads(msg.data)
            original_text = interpretation['original_text']

            # Use LLM to enhance interpretation
            enhanced_interpretation = self.enhance_interpretation(original_text)

            # Publish enhanced action plan
            plan_msg = String()
            plan_msg.data = json.dumps(enhanced_interpretation)
            self.action_plan_pub.publish(plan_msg)

        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in NLU message')

    def enhance_interpretation(self, command: str) -> Dict[str, Any]:
        """Use LLM to enhance command interpretation with context and reasoning"""
        prompt = f"""
        Interpret this robot command: "{command}"

        Provide a structured JSON response with:
        1. action: the primary action to perform
        2. parameters: specific parameters for the action
        3. context: environmental context considerations
        4. safety: safety considerations for execution
        5. alternatives: alternative interpretations if ambiguous
        6. confidence: confidence level (0-1)

        Consider common sense, physics, and safety in your interpretation.
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )

            # Parse response (in practice, ensure proper JSON format)
            content = response.choices[0].message.content

            # Clean up response if needed
            if content.startswith('```json'):
                content = content[7:content.rfind('```')]
            elif content.startswith('```'):
                content = content[3:content.rfind('```')]

            return json.loads(content)

        except Exception as e:
            self.get_logger().error(f'LLM interpretation error: {e}')
            # Return fallback interpretation
            return {
                'action': 'unknown',
                'parameters': {},
                'context': 'unknown',
                'safety': 'unknown',
                'alternatives': [],
                'confidence': 0.0
            }

def main(args=None):
    rclpy.init(args=args)
    node = LLMInterpreterNode()

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

## Action Planning and Execution

### Action Planner

```python
# File: voice_pipeline/action_planner.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path
import json
from typing import List, Dict, Any

class ActionPlannerNode(Node):
    def __init__(self):
        super().__init__('action_planner_node')

        # Subscribers
        self.action_plan_sub = self.create_subscription(
            String, '/action/plan', self.action_plan_callback, 10)

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.nav_goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.status_pub = self.create_publisher(String, '/voice/status', 10)

        self.get_logger().info('Action Planner Node Started')

    def action_plan_callback(self, msg):
        """Process action plan and execute"""
        try:
            plan = json.loads(msg.data)
            action = plan.get('action', 'unknown')

            self.get_logger().info(f'Executing action: {action}')

            if action == 'move':
                self.execute_move_action(plan.get('parameters', {}))
            elif action == 'turn':
                self.execute_turn_action(plan.get('parameters', {}))
            elif action == 'stop':
                self.execute_stop_action()
            elif action == 'navigate':
                self.execute_navigation_action(plan.get('parameters', {}))
            elif action == 'pickup':
                self.execute_pickup_action(plan.get('parameters', {}))
            elif action == 'place':
                self.execute_place_action(plan.get('parameters', {}))
            else:
                self.get_logger().warn(f'Unknown action: {action}')
                self.publish_status(f'Unknown command: {action}')

        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in action plan')

    def execute_move_action(self, params: Dict[str, Any]):
        """Execute move action"""
        cmd = Twist()

        distance = params.get('distance', 1.0)  # Default 1 meter
        duration = distance / 0.5  # Assuming 0.5 m/s speed

        # Simple move forward
        cmd.linear.x = 0.5  # 0.5 m/s
        cmd.angular.z = 0.0

        # Publish command for duration
        self.publish_command_for_duration(cmd, duration)
        self.publish_status(f'Moving forward {distance} meters')

    def execute_turn_action(self, params: Dict[str, Any]):
        """Execute turn action"""
        cmd = Twist()

        angle = params.get('angle', 90)  # Default 90 degrees
        direction = params.get('direction', 'left')

        # Convert angle to radians and calculate duration
        angle_rad = angle * 3.14159 / 180.0
        duration = angle_rad / 0.5  # Assuming 0.5 rad/s angular velocity

        cmd.linear.x = 0.0
        cmd.angular.z = 0.5 if direction == 'left' else -0.5

        self.publish_command_for_duration(cmd, duration)
        self.publish_status(f'Turning {direction} {angle} degrees')

    def execute_stop_action(self):
        """Execute stop action"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0

        self.cmd_vel_pub.publish(cmd)
        self.publish_status('Stopping')

    def execute_navigation_action(self, params: Dict[str, Any]):
        """Execute navigation action"""
        # This would integrate with navigation stack
        goal_x = params.get('x', 0.0)
        goal_y = params.get('y', 0.0)

        goal_msg = PoseStamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = 'map'
        goal_msg.pose.position.x = goal_x
        goal_msg.pose.position.y = goal_y
        goal_msg.pose.position.z = 0.0
        goal_msg.pose.orientation.w = 1.0

        self.nav_goal_pub.publish(goal_msg)
        self.publish_status(f'Navigating to ({goal_x}, {goal_y})')

    def execute_pickup_action(self, params: Dict[str, Any]):
        """Execute pickup action"""
        # This would integrate with manipulation stack
        object_name = params.get('object', 'unknown')
        self.publish_status(f'Attempting to pick up {object_name}')
        # In real implementation, this would call manipulation services

    def execute_place_action(self, params: Dict[str, Any]):
        """Execute place action"""
        # This would integrate with manipulation stack
        object_name = params.get('object', 'unknown')
        self.publish_status(f'Attempting to place {object_name}')
        # In real implementation, this would call manipulation services

    def publish_command_for_duration(self, cmd: Twist, duration: float):
        """Publish command for specified duration"""
        start_time = self.get_clock().now()
        end_time = start_time + rclpy.time.Duration(seconds=duration)

        while self.get_clock().now() < end_time:
            self.cmd_vel_pub.publish(cmd)
            rclpy.spin_once(self, timeout_sec=0.1)

        # Stop after duration
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)

    def publish_status(self, status: str):
        """Publish status message"""
        status_msg = String()
        status_msg.data = status
        self.status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ActionPlannerNode()

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

## Voice Command Validation

### Safety and Validation System

```python
# File: voice_pipeline/command_validator.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
import json
from typing import Dict, Any, Tuple

class CommandValidatorNode(Node):
    def __init__(self):
        super().__init__('command_validator_node')

        # Subscribers
        self.action_plan_sub = self.create_subscription(
            String, '/action/plan', self.action_plan_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)

        # Publishers
        self.validated_plan_pub = self.create_publisher(
            String, '/action/validated_plan', 10)
        self.warning_pub = self.create_publisher(
            String, '/voice/warning', 10)

        # State
        self.laser_data = None
        self.safety_threshold = 0.5  # 50cm safety distance

        self.get_logger().info('Command Validator Node Started')

    def scan_callback(self, msg):
        """Update laser scan data"""
        self.laser_data = msg

    def action_plan_callback(self, msg):
        """Validate action plan against safety constraints"""
        try:
            plan = json.loads(msg.data)

            # Validate plan
            is_safe, reason = self.validate_plan(plan)

            if is_safe:
                # Publish validated plan
                validated_msg = String()
                validated_msg.data = msg.data
                self.validated_plan_pub.publish(validated_msg)
            else:
                # Issue warning and modify plan if possible
                self.get_logger().warn(f'Unsafe command: {reason}')
                self.publish_warning(f'Unsafe command: {reason}')

                # Try to modify plan to be safe
                safe_plan = self.modify_plan_for_safety(plan)
                if safe_plan:
                    validated_msg = String()
                    validated_msg.data = json.dumps(safe_plan)
                    self.validated_plan_pub.publish(validated_msg)

        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in action plan')

    def validate_plan(self, plan: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate plan against safety constraints"""
        action = plan.get('action', 'unknown')

        if action in ['move', 'navigate']:
            if self.laser_data:
                # Check if path is clear
                min_distance = min(self.laser_data.ranges) if self.laser_data.ranges else float('inf')

                if min_distance < self.safety_threshold:
                    return False, f'Obstacle detected at {min_distance:.2f}m, minimum safe distance is {self.safety_threshold}m'

        elif action == 'turn':
            # Check if turning would cause collision
            if self.laser_data:
                # For left turn, check left side; for right turn, check right side
                params = plan.get('parameters', {})
                direction = params.get('direction', 'left')

                if direction == 'left':
                    # Check left side (first quarter of ranges)
                    left_ranges = self.laser_data.ranges[:len(self.laser_data.ranges)//4]
                else:  # right
                    # Check right side (last quarter of ranges)
                    right_idx = 3 * len(self.laser_data.ranges) // 4
                    left_ranges = self.laser_data.ranges[right_idx:]

                min_left_distance = min(left_ranges) if left_ranges else float('inf')

                if min_left_distance < self.safety_threshold:
                    return False, f'Obstacle on {direction} side at {min_left_distance:.2f}m'

        return True, "Plan is safe"

    def modify_plan_for_safety(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Modify plan to make it safe, if possible"""
        action = plan.get('action', 'unknown')

        if action == 'move':
            # Reduce distance if obstacle detected
            if self.laser_data:
                min_distance = min(self.laser_data.ranges) if self.laser_data.ranges else float('inf')

                if min_distance < self.safety_threshold:
                    # Modify distance to be safe
                    safe_distance = max(0.1, min_distance - 0.1)  # 10cm buffer
                    modified_plan = plan.copy()
                    if 'parameters' not in modified_plan:
                        modified_plan['parameters'] = {}
                    modified_plan['parameters']['distance'] = safe_distance
                    modified_plan['modified'] = True
                    modified_plan['modification_reason'] = f'Reduced distance for safety (was too close to obstacle)'
                    return modified_plan

        return None  # No safe modification possible

    def publish_warning(self, warning: str):
        """Publish warning message"""
        warning_msg = String()
        warning_msg.data = warning
        self.warning_pub.publish(warning_msg)

def main(args=None):
    rclpy.init(args=args)
    node = CommandValidatorNode()

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

## Complete Voice Pipeline Launch

### Main Launch File

```python
# File: voice_pipeline/launch/voice_pipeline.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    # Audio capture node
    audio_capture = Node(
        package='voice_pipeline',
        executable='audio_capture',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Speech recognition node
    speech_recognition = Node(
        package='voice_pipeline',
        executable='speech_recognition',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # NLU processor node
    nlu_processor = Node(
        package='voice_pipeline',
        executable='nlu_processor',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # LLM interpreter node
    llm_interpreter = Node(
        package='voice_pipeline',
        executable='llm_interpreter',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Command validator node
    command_validator = Node(
        package='voice_pipeline',
        executable='command_validator',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Action planner node
    action_planner = Node(
        package='voice_pipeline',
        executable='action_planner',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time'
        ),
        audio_capture,
        speech_recognition,
        nlu_processor,
        llm_interpreter,
        command_validator,
        action_planner
    ])
```

## Performance Optimization

### Audio Processing Optimization

```python
# File: voice_pipeline/optimization.py
import numpy as np
from collections import deque
import threading
import time

class OptimizedAudioProcessor:
    def __init__(self):
        self.sample_rate = 16000
        self.buffer_size = 8192  # Process in chunks
        self.audio_buffer = deque(maxlen=self.buffer_size)
        self.processing_lock = threading.Lock()

        # Pre-allocated arrays to avoid allocation during processing
        self.temp_array = np.zeros(self.buffer_size, dtype=np.float32)
        self.processed_array = np.zeros(self.buffer_size, dtype=np.float32)

    def process_audio_chunk(self, chunk):
        """Optimized audio processing with minimal allocation"""
        with self.processing_lock:
            # Convert to float32 if needed
            if chunk.dtype != np.float32:
                chunk = chunk.astype(np.float32) / 32768.0

            # Copy to temp array to avoid modifying original
            np.copyto(self.temp_array[:len(chunk)], chunk)

            # Apply processing
            processed = self.apply_processing(self.temp_array[:len(chunk)])

            return processed

    def apply_processing(self, audio_data):
        """Apply optimized processing steps"""
        # Normalize
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val

        # Apply simple noise reduction
        audio_data = self.simple_noise_reduction(audio_data)

        return audio_data

    def simple_noise_reduction(self, audio_data, threshold=0.01):
        """Simple noise reduction without complex algorithms"""
        # Create a simple mask based on threshold
        mask = np.abs(audio_data) > threshold
        return audio_data * mask.astype(np.float32)
```

## Best Practices for Voice-to-Action Pipelines

### 1. Error Handling
- Implement robust error handling at each pipeline stage
- Provide graceful degradation when components fail
- Log errors for debugging and improvement

### 2. Performance Optimization
- Use appropriate buffer sizes to balance latency and throughput
- Optimize audio processing to run in real-time
- Consider using separate threads for different pipeline stages

### 3. Safety Considerations
- Always validate LLM-generated actions before execution
- Implement safety checks based on sensor data
- Provide emergency stop capabilities

### 4. User Experience
- Provide clear audio feedback when listening
- Confirm understanding before executing commands
- Handle ambiguous commands gracefully

## Chapter Summary

The voice-to-action pipeline enables natural human-robot interaction by converting spoken commands into physical robot actions. The pipeline involves audio capture, speech recognition, natural language understanding, LLM processing, action planning, and execution. Proper integration requires attention to real-time performance, safety validation, and user experience considerations.

## Exercises

1. Implement a complete voice-to-action pipeline with speech recognition and command execution.
2. Add voice feedback to confirm command understanding.
3. Implement safety validation that prevents unsafe actions based on sensor data.

## Next Steps

In the next chapter, we'll explore cognitive planning with LLMs, learning how to use language models for higher-level reasoning and decision making in Physical AI systems.