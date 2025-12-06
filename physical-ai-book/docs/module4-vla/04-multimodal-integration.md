---
sidebar_position: 4
---

# Multimodal Integration

## Chapter Objectives

By the end of this chapter, you will be able to:
- Design and implement multimodal systems that combine vision, language, and action
- Integrate multiple sensory modalities for enhanced Physical AI capabilities
- Create unified representations that bridge different modalities
- Build systems that understand and respond to complex multimodal inputs
- Implement cross-modal reasoning and decision making

## Multimodal Integration Overview

### What is Multimodal Integration?

Multimodal integration in Physical AI refers to the ability to process, understand, and act upon information from multiple sensory modalities simultaneously. This includes:

- **Visual Modality**: Cameras, LiDAR, depth sensors
- **Language Modality**: Speech, text, natural language commands
- **Action Modality**: Physical manipulation, navigation, interaction
- **Other Modalities**: Audio, tactile, thermal, etc.

### The VLA Framework

The Vision-Language-Action (VLA) framework provides a unified approach to multimodal integration:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│    Vision       │    │   Language      │    │     Action      │
│   (Perception)  │◄──►│   (Cognition)   │◄──►│   (Execution)   │
│   • Cameras     │    │   • LLMs        │    │   • Navigation  │
│   • LiDAR       │    │   • NLP         │    │   • Manipulation│
│   • Depth       │    │   • Dialogue    │    │   • Control     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  Cross-Modal    │
                    │  Integration    │
                    │  (Fusion,       │
                    │   Reasoning,    │
                    │   Grounding)    │
                    └─────────────────┘
```

## Cross-Modal Representations

### Unified Embedding Spaces

```python
# File: multimodal_integration/embedding_fusion.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from typing import Dict, Any, List, Optional
import cv2
from cv_bridge import CvBridge

class MultimodalEmbeddingNode(Node):
    def __init__(self):
        super().__init__('multimodal_embedding_node')

        # Initialize CLIP model for vision-language integration
        try:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        except:
            self.get_logger().warn('CLIP model not available, using placeholder')
            self.clip_model = None
            self.clip_processor = None

        # Initialize CV bridge
        self.cv_bridge = CvBridge()

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.text_sub = self.create_subscription(
            String, '/user/command', self.text_callback, 10)
        self.detections_sub = self.create_subscription(
            Detection2DArray, '/object/detections', self.detections_callback, 10)

        # Publishers
        self.fusion_pub = self.create_publisher(
            String, '/multimodal/fusion_output', 10)

        # State
        self.current_image = None
        self.current_text = None
        self.current_detections = None

        self.get_logger().info('Multimodal Embedding Node Started')

    def image_callback(self, msg):
        """Process image and extract visual embeddings"""
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            self.current_image = cv_image

            # Extract visual features using CLIP
            if self.clip_model is not None:
                inputs = self.clip_processor(images=cv_image, return_tensors="pt", padding=True)
                visual_features = self.clip_model.get_image_features(**inputs)
                self.process_visual_features(visual_features, cv_image)

        except Exception as e:
            self.get_logger().error(f'Image processing error: {e}')

    def text_callback(self, msg):
        """Process text and extract language embeddings"""
        self.current_text = msg.data

        # Extract text features using CLIP
        if self.clip_model is not None:
            try:
                inputs = self.clip_processor(text=msg.data, return_tensors="pt", padding=True)
                text_features = self.clip_model.get_text_features(**inputs)
                self.process_text_features(text_features, msg.data)
            except Exception as e:
                self.get_logger().error(f'Text processing error: {e}')

    def detections_callback(self, msg):
        """Process object detections"""
        self.current_detections = msg

    def process_visual_features(self, features, image):
        """Process visual features and prepare for fusion"""
        # Convert to numpy for easier handling
        visual_embedding = features.detach().cpu().numpy()

        # Store with spatial information
        height, width = image.shape[:2]
        spatial_features = self.extract_spatial_features(image)

        return {
            'embedding': visual_embedding,
            'spatial': spatial_features,
            'resolution': (height, width)
        }

    def process_text_features(self, features, text):
        """Process text features and prepare for fusion"""
        text_embedding = features.detach().cpu().numpy()

        # Extract semantic information
        semantic_features = self.extract_semantic_features(text)

        return {
            'embedding': text_embedding,
            'semantic': semantic_features,
            'original_text': text
        }

    def extract_spatial_features(self, image):
        """Extract spatial features from image"""
        # Simple spatial features: edges, corners, regions
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Corner detection
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)

        return {
            'edges': edges,
            'corners': corners,
            'center_of_mass': self.calculate_center_of_mass(edges)
        }

    def extract_semantic_features(self, text):
        """Extract semantic features from text"""
        # Simple keyword extraction and semantic analysis
        keywords = self.extract_keywords(text)
        intent = self.classify_intent(text)

        return {
            'keywords': keywords,
            'intent': intent,
            'entities': self.extract_entities(text)
        }

    def extract_keywords(self, text):
        """Extract important keywords from text"""
        # Simple keyword extraction (in practice, use NLP libraries)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = text.lower().replace(',', ' ').replace('.', ' ').split()
        return [word for word in words if word not in stop_words]

    def classify_intent(self, text):
        """Classify intent of text command"""
        navigation_keywords = ['go', 'move', 'navigate', 'to', 'toward', 'towards']
        manipulation_keywords = ['pick', 'place', 'grasp', 'take', 'put', 'drop']
        perception_keywords = ['see', 'find', 'look', 'detect', 'where', 'what']

        text_lower = text.lower()

        if any(keyword in text_lower for keyword in navigation_keywords):
            return 'navigation'
        elif any(keyword in text_lower for keyword in manipulation_keywords):
            return 'manipulation'
        elif any(keyword in text_lower for keyword in perception_keywords):
            return 'perception'
        else:
            return 'other'

    def extract_entities(self, text):
        """Extract named entities from text"""
        # Simple entity extraction (in practice, use NER models)
        entities = []
        words = text.split()

        # Look for potential object names
        object_indicators = ['the', 'a', 'an']
        for i, word in enumerate(words):
            if word.lower() in object_indicators and i + 1 < len(words):
                entities.append(words[i + 1])

        return entities

    def fuse_modalities(self, visual_data, text_data):
        """Fuse visual and text modalities"""
        if visual_data is None or text_data is None:
            return None

        # Simple fusion: compute similarity between visual and text embeddings
        visual_emb = visual_data['embedding']
        text_emb = text_data['embedding']

        # Normalize embeddings
        visual_emb_norm = visual_emb / np.linalg.norm(visual_emb)
        text_emb_norm = text_emb / np.linalg.norm(text_emb)

        # Compute similarity
        similarity = np.dot(visual_emb_norm, text_emb_norm.T)

        # Create fused representation
        fused_data = {
            'similarity': float(similarity[0][0]) if hasattr(similarity, '__len__') else float(similarity),
            'visual_context': visual_data,
            'text_context': text_data,
            'grounding_confidence': self.calculate_grounding_confidence(visual_data, text_data)
        }

        return fused_data

    def calculate_grounding_confidence(self, visual_data, text_data):
        """Calculate confidence in visual-language grounding"""
        # Simple confidence calculation based on keyword matching
        keywords = text_data['semantic']['keywords']
        # In a real system, this would use more sophisticated grounding

        # For now, return a placeholder confidence
        return 0.8

    def calculate_center_of_mass(self, binary_image):
        """Calculate center of mass of non-zero pixels"""
        y, x = np.nonzero(binary_image)
        if len(x) > 0 and len(y) > 0:
            return (int(np.mean(x)), int(np.mean(y)))
        return (0, 0)

def main(args=None):
    rclpy.init(args=args)
    node = MultimodalEmbeddingNode()

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

## Vision-Language Grounding

### Object Grounding and Referencing

```python
# File: multimodal_integration/vision_language_grounding.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray, Detection2D
from geometry_msgs.msg import Point
import numpy as np
import cv2
from cv_bridge import CvBridge
import json
from typing import Dict, Any, List, Optional

class VisionLanguageGroundingNode(Node):
    def __init__(self):
        super().__init__('vision_language_grounding_node')

        # Initialize CV bridge
        self.cv_bridge = CvBridge()

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.command_sub = self.create_subscription(
            String, '/user/command', self.command_callback, 10)
        self.detections_sub = self.create_subscription(
            Detection2DArray, '/object/detections', self.detections_callback, 10)

        # Publishers
        self.grounded_objects_pub = self.create_publisher(
            String, '/multimodal/grounded_objects', 10)
        self.visualization_pub = self.create_publisher(
            Image, '/multimodal/grounding_viz', 10)

        # State
        self.current_image = None
        self.current_detections = None
        self.pending_command = None

        self.get_logger().info('Vision-Language Grounding Node Started')

    def image_callback(self, msg):
        """Process incoming image for grounding"""
        try:
            self.current_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Image conversion error: {e}')

    def command_callback(self, msg):
        """Process command and perform grounding"""
        command = msg.data
        self.pending_command = command

        # Perform grounding if we have both image and detections
        if self.current_image is not None and self.current_detections is not None:
            self.perform_grounding(command)

    def detections_callback(self, msg):
        """Process object detections"""
        self.current_detections = msg

        # Process pending command if available
        if self.pending_command is not None and self.current_image is not None:
            self.perform_grounding(self.pending_command)
            self.pending_command = None

    def perform_grounding(self, command: str):
        """Perform vision-language grounding"""
        if self.current_detections is None:
            return

        # Parse command to identify target object
        target_object = self.parse_command_for_object(command)

        if not target_object:
            self.get_logger().info('No target object identified in command')
            return

        # Find matching detection
        matched_detection = self.find_matching_detection(target_object)

        if matched_detection:
            # Create grounding result
            grounding_result = {
                'command': command,
                'target_object': target_object,
                'detection': self.detection_to_dict(matched_detection),
                'confidence': self.calculate_grounding_confidence(target_object, matched_detection),
                'spatial_relationship': self.analyze_spatial_relationship(matched_detection)
            }

            # Publish grounding result
            grounding_msg = String()
            grounding_msg.data = json.dumps(grounding_result)
            self.grounded_objects_pub.publish(grounding_msg)

            # Create visualization
            viz_image = self.create_grounding_visualization(target_object, matched_detection)
            viz_msg = self.cv_bridge.cv2_to_imgmsg(viz_image, "bgr8")
            self.visualization_pub.publish(viz_msg)

            self.get_logger().info(f'Grounded "{target_object}" with confidence {grounding_result["confidence"]:.2f}')

    def parse_command_for_object(self, command: str) -> Optional[str]:
        """Parse command to identify target object"""
        # Simple object extraction (in practice, use NLP)
        command_lower = command.lower()

        # Look for object references
        object_indicators = ['the', 'a', 'an']
        words = command_lower.split()

        # Common object categories that might be in commands
        object_categories = [
            'box', 'bottle', 'cup', 'chair', 'table', 'person', 'robot',
            'object', 'item', 'thing', 'book', 'phone', 'laptop', 'monitor'
        ]

        for word in words:
            if word in object_categories:
                return word

        # Look for color + object patterns
        color_object_patterns = [
            'red box', 'blue cup', 'green bottle', 'white chair', 'black table'
        ]

        for pattern in color_object_patterns:
            if pattern in command_lower:
                return pattern

        return None

    def find_matching_detection(self, target_object: str) -> Optional[Detection2D]:
        """Find detection that matches target object"""
        if self.current_detections is None:
            return None

        target_lower = target_object.lower()

        # Score each detection based on similarity to target
        best_match = None
        best_score = 0.0

        for detection in self.current_detections.detections:
            if detection.results:
                class_name = detection.results[0].hypothesis.class_id.lower()
                confidence = detection.results[0].hypothesis.score

                # Calculate match score
                score = self.calculate_match_score(target_lower, class_name, confidence)

                if score > best_score:
                    best_score = score
                    best_match = detection

        return best_match if best_score > 0.3 else None  # Threshold

    def calculate_match_score(self, target: str, detected_class: str, confidence: float) -> float:
        """Calculate match score between target and detected object"""
        score = 0.0

        # Exact match
        if target == detected_class:
            score = confidence
        # Partial match (e.g., "box" vs "cardboard box")
        elif target in detected_class or detected_class in target:
            score = confidence * 0.8
        # Semantic similarity could be added here
        else:
            # Check for semantic similarity using simple rules
            semantic_matches = {
                'person': ['human', 'man', 'woman', 'person'],
                'bottle': ['container', 'vessel', 'jug'],
                'cup': ['mug', 'glass', 'vessel'],
                'box': ['container', 'crate', 'carton']
            }

            for semantic_target, semantic_variants in semantic_matches.items():
                if target == semantic_target and detected_class in semantic_variants:
                    score = confidence * 0.7
                    break
                elif detected_class == semantic_target and target in semantic_variants:
                    score = confidence * 0.7
                    break

        return score

    def calculate_grounding_confidence(self, target_object: str, detection: Detection2D) -> float:
        """Calculate confidence in the grounding"""
        if detection.results:
            base_confidence = detection.results[0].hypothesis.score
            # Additional factors could be added here
            return base_confidence
        return 0.0

    def analyze_spatial_relationship(self, detection: Detection2D) -> Dict[str, Any]:
        """Analyze spatial relationship of detected object"""
        bbox = detection.bbox
        center_x = bbox.center.x
        center_y = bbox.center.y

        # Calculate relative position in image
        img_width, img_height = 640, 480  # Assuming standard resolution

        # Normalize coordinates
        norm_x = center_x / img_width
        norm_y = center_y / img_height

        # Determine spatial relationship
        position = []
        if norm_x < 0.33:
            position.append('left')
        elif norm_x > 0.67:
            position.append('right')
        else:
            position.append('center')

        if norm_y < 0.33:
            position.append('top')
        elif norm_y > 0.67:
            position.append('bottom')
        else:
            position.append('middle')

        return {
            'position': position,
            'normalized_coordinates': [norm_x, norm_y],
            'bbox': [bbox.center.x, bbox.center.y, bbox.size_x, bbox.size_y]
        }

    def create_grounding_visualization(self, target_object: str, detection: Detection2D) -> np.ndarray:
        """Create visualization showing grounding result"""
        if self.current_image is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        viz_image = self.current_image.copy()

        # Draw bounding box
        bbox = detection.bbox
        x = int(bbox.center.x - bbox.size_x / 2)
        y = int(bbox.center.y - bbox.size_y / 2)
        w = int(bbox.size_x)
        h = int(bbox.size_y)

        cv2.rectangle(viz_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Add label
        label = f"{target_object}"
        cv2.putText(viz_image, label, (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return viz_image

    def detection_to_dict(self, detection: Detection2D) -> Dict[str, Any]:
        """Convert detection to dictionary for JSON serialization"""
        result = {
            'bbox': {
                'center_x': detection.bbox.center.x,
                'center_y': detection.bbox.center.y,
                'size_x': detection.bbox.size_x,
                'size_y': detection.bbox.size_y
            },
            'class': detection.results[0].hypothesis.class_id if detection.results else 'unknown',
            'confidence': detection.results[0].hypothesis.score if detection.results else 0.0
        }
        return result

def main(args=None):
    rclpy.init(args=args)
    node = VisionLanguageGroundingNode()

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

## Action-Language Integration

### Natural Language Command to Action Mapping

```python
# File: multimodal_integration/action_language_mapping.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
import json
import openai
from typing import Dict, Any, List, Optional, Tuple
import re

class ActionLanguageMapperNode(Node):
    def __init__(self):
        super().__init__('action_language_mapper_node')

        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key='your-api-key')

        # Subscribers
        self.command_sub = self.create_subscription(
            String, '/natural/command', self.command_callback, 10)
        self.context_sub = self.create_subscription(
            String, '/multimodal/context', self.context_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)

        # Publishers
        self.action_pub = self.create_publisher(
            String, '/robot/action_sequence', 10)
        self.feedback_pub = self.create_publisher(
            String, '/voice/feedback', 10)

        # State
        self.context = {}
        self.scan_data = None

        self.get_logger().info('Action-Language Mapper Node Started')

    def context_callback(self, msg):
        """Update context"""
        try:
            self.context = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in context message')

    def scan_callback(self, msg):
        """Update laser scan data"""
        self.scan_data = msg

    def command_callback(self, msg):
        """Process natural language command and map to actions"""
        command = msg.data
        self.get_logger().info(f'Processing command: {command}')

        # Map command to action sequence
        action_sequence = self.map_command_to_actions(command)

        if action_sequence:
            # Publish action sequence
            action_msg = String()
            action_msg.data = json.dumps(action_sequence)
            self.action_pub.publish(action_msg)

            # Provide feedback
            feedback = f"Understood command: {command}. Executing {len(action_sequence)} actions."
            self.provide_feedback(feedback)
        else:
            feedback = f"Could not understand command: {command}"
            self.provide_feedback(feedback)

    def map_command_to_actions(self, command: str) -> List[Dict[str, Any]]:
        """Map natural language command to sequence of actions using LLM"""
        context_info = self.get_context_info()

        prompt = f"""
        Convert this natural language command to a sequence of robot actions:
        Command: "{command}"

        Context:
        {context_info}

        Provide a JSON list of actions with the following format:
        {{
            "action": "action_type",
            "parameters": {{"param1": "value1", "param2": "value2"}},
            "description": "Brief description of what this action does"
        }}

        Available action types:
        - navigation: move to a location
        - manipulation: pick/place objects
        - perception: look for/detect objects
        - communication: speak/communicate
        - wait: pause execution

        Consider safety, environment constraints, and robot capabilities.
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )

            content = response.choices[0].message.content

            # Extract JSON from response
            if '```json' in content:
                start = content.find('```json') + 7
                end = content.find('```', start)
                json_content = content[start:end].strip()
            elif '```' in content:
                start = content.find('```') + 3
                end = content.find('```', start)
                json_content = content[start:end].strip()
            else:
                json_content = content

            return json.loads(json_content)

        except Exception as e:
            self.get_logger().error(f'LLM mapping error: {e}')
            # Fallback to rule-based mapping
            return self.rule_based_mapping(command)

    def rule_based_mapping(self, command: str) -> List[Dict[str, Any]]:
        """Fallback rule-based command mapping"""
        command_lower = command.lower()

        # Navigation commands
        if any(word in command_lower for word in ['go to', 'move to', 'navigate to', 'go', 'move']):
            return self.parse_navigation_command(command)

        # Manipulation commands
        elif any(word in command_lower for word in ['pick up', 'grasp', 'take', 'pick']):
            return self.parse_manipulation_command(command)

        # Perception commands
        elif any(word in command_lower for word in ['find', 'look for', 'see', 'detect', 'where']):
            return self.parse_perception_command(command)

        # Simple movement commands
        elif any(word in command_lower for word in ['forward', 'backward', 'left', 'right', 'turn']):
            return self.parse_movement_command(command)

        else:
            return []

    def parse_navigation_command(self, command: str) -> List[Dict[str, Any]]:
        """Parse navigation command"""
        # Extract location using regex
        location_patterns = [
            r'to the (\w+)',  # "go to the kitchen"
            r'to (\w+)',     # "go to kitchen"
            r'(\w+)',        # "kitchen" (if context suggests movement)
        ]

        location = None
        for pattern in location_patterns:
            match = re.search(pattern, command.lower())
            if match:
                location = match.group(1)
                break

        if location:
            return [{
                'action': 'navigation',
                'parameters': {'target_location': location},
                'description': f'Navigate to {location}'
            }]
        else:
            return [{
                'action': 'navigation',
                'parameters': {'target_location': 'unknown'},
                'description': 'Navigate to specified location'
            }]

    def parse_manipulation_command(self, command: str) -> List[Dict[str, Any]]:
        """Parse manipulation command"""
        # Extract object
        object_patterns = [
            r'pick up the (\w+)',
            r'pick up (\w+)',
            r'grasp the (\w+)',
            r'take the (\w+)',
        ]

        obj = None
        for pattern in object_patterns:
            match = re.search(pattern, command.lower())
            if match:
                obj = match.group(1)
                break

        return [{
            'action': 'manipulation',
            'parameters': {'action_type': 'pick', 'object': obj or 'unknown'},
            'description': f'Pick up {obj or "object"}'
        }]

    def parse_perception_command(self, command: str) -> List[Dict[str, Any]]:
        """Parse perception command"""
        # Extract object to find
        object_patterns = [
            r'find the (\w+)',
            r'look for the (\w+)',
            r'where is the (\w+)',
            r'find (\w+)',
        ]

        obj = None
        for pattern in object_patterns:
            match = re.search(pattern, command.lower())
            if match:
                obj = match.group(1)
                break

        return [{
            'action': 'perception',
            'parameters': {'task': 'detection', 'target_object': obj or 'unknown'},
            'description': f'Look for {obj or "object"}'
        }]

    def parse_movement_command(self, command: str) -> List[Dict[str, Any]]:
        """Parse simple movement command"""
        command_lower = command.lower()

        if 'forward' in command_lower:
            return [{
                'action': 'movement',
                'parameters': {'direction': 'forward', 'distance': 1.0},
                'description': 'Move forward 1 meter'
            }]
        elif 'backward' in command_lower:
            return [{
                'action': 'movement',
                'parameters': {'direction': 'backward', 'distance': 1.0},
                'description': 'Move backward 1 meter'
            }]
        elif 'left' in command_lower:
            return [{
                'action': 'movement',
                'parameters': {'direction': 'left', 'angle': 90},
                'description': 'Turn left 90 degrees'
            }]
        elif 'right' in command_lower:
            return [{
                'action': 'movement',
                'parameters': {'direction': 'right', 'angle': 90},
                'description': 'Turn right 90 degrees'
            }]
        else:
            return []

    def get_context_info(self) -> str:
        """Get relevant context information"""
        context_str = f"""
        Robot State:
        - Current position: {self.context.get('robot_position', 'unknown')}
        - Battery level: {self.context.get('battery_level', 'unknown')}%
        - Available capabilities: {self.context.get('capabilities', [])}

        Environment:
        - Detected objects: {self.context.get('detected_objects', [])}
        - Navigation zones: {self.context.get('navigation_zones', {})}
        - Obstacles: {len(self.scan_data.ranges) if self.scan_data else 0} range readings
        """
        return context_str

    def provide_feedback(self, feedback: str):
        """Provide feedback to user"""
        feedback_msg = String()
        feedback_msg.data = feedback
        self.feedback_pub.publish(feedback_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ActionLanguageMapperNode()

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

## Multimodal Decision Making

### Cross-Modal Reasoning Engine

```python
# File: multimodal_integration/reasoning_engine.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import PoseStamped
import json
import openai
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import threading
import time

class MultimodalReasoningNode(Node):
    def __init__(self):
        super().__init__('multimodal_reasoning_node')

        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key='your-api-key')

        # Subscribers
        self.vision_sub = self.create_subscription(
            Image, '/camera/image_raw', self.vision_callback, 10)
        self.language_sub = self.create_subscription(
            String, '/user/request', self.language_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.pose_sub = self.create_subscription(
            PoseStamped, '/current_pose', self.pose_callback, 10)

        # Publishers
        self.decision_pub = self.create_publisher(
            String, '/multimodal/decision', 10)
        self.plan_pub = self.create_publisher(
            String, '/multimodal/action_plan', 10)

        # State
        self.multimodal_state = {
            'vision_data': None,
            'language_request': None,
            'scan_data': None,
            'robot_pose': None,
            'timestamp': time.time()
        }

        # Lock for thread-safe state updates
        self.state_lock = threading.Lock()

        # Timer for periodic reasoning
        self.reasoning_timer = self.create_timer(2.0, self.periodic_reasoning)

        self.get_logger().info('Multimodal Reasoning Node Started')

    def vision_callback(self, msg):
        """Update vision data"""
        with self.state_lock:
            self.multimodal_state['vision_data'] = {
                'timestamp': time.time(),
                'encoding': msg.encoding,
                'height': msg.height,
                'width': msg.width
            }
            self.multimodal_state['timestamp'] = time.time()

    def language_callback(self, msg):
        """Update language request"""
        with self.state_lock:
            self.multimodal_state['language_request'] = msg.data
            self.multimodal_state['timestamp'] = time.time()

    def scan_callback(self, msg):
        """Update scan data"""
        with self.state_lock:
            self.multimodal_state['scan_data'] = {
                'ranges_count': len(msg.ranges),
                'min_range': min(msg.ranges) if msg.ranges else float('inf'),
                'max_range': max(msg.ranges) if msg.ranges else 0,
                'timestamp': time.time()
            }
            self.multimodal_state['timestamp'] = time.time()

    def pose_callback(self, msg):
        """Update robot pose"""
        with self.state_lock:
            self.multimodal_state['robot_pose'] = {
                'x': msg.pose.position.x,
                'y': msg.pose.position.y,
                'z': msg.pose.position.z,
                'timestamp': time.time()
            }
            self.multimodal_state['timestamp'] = time.time()

    def periodic_reasoning(self):
        """Perform periodic multimodal reasoning"""
        with self.state_lock:
            current_state = self.multimodal_state.copy()

        # Only reason if we have recent data
        if time.time() - current_state['timestamp'] < 5.0:  # 5 seconds
            decision = self.perform_multimodal_reasoning(current_state)

            if decision:
                # Publish decision
                decision_msg = String()
                decision_msg.data = json.dumps(decision)
                self.decision_pub.publish(decision_msg)

                # Generate action plan
                action_plan = self.generate_action_plan(decision, current_state)
                if action_plan:
                    plan_msg = String()
                    plan_msg.data = json.dumps(action_plan)
                    self.plan_pub.publish(plan_msg)

    def perform_multimodal_reasoning(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Perform multimodal reasoning using LLM"""
        # Check if we have a language request to respond to
        if not state.get('language_request'):
            return None

        # Build reasoning prompt
        prompt = f"""
        Perform multimodal reasoning based on the following information:

        Language Request: "{state.get('language_request', 'No request')}"
        Robot Position: {state.get('robot_pose', 'Unknown')}
        Vision Data: {state.get('vision_data', 'No vision data')}
        Scan Data: {state.get('scan_data', 'No scan data')}
        Current Time: {time.time()}

        Analyze the situation by combining visual, spatial, and linguistic information to:
        1. Understand what the user wants
        2. Assess the current situation
        3. Determine if the request is feasible
        4. Identify potential challenges or constraints

        Provide your reasoning and a decision in JSON format:
        {{
            "reasoning": "step-by-step analysis combining all modalities",
            "feasibility": true/false,
            "confidence": 0.0-1.0,
            "action_needed": "type of action required",
            "constraints": ["list of constraints"],
            "safety_considerations": ["list of safety factors"]
        }}
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )

            content = response.choices[0].message.content

            # Extract JSON
            if '```json' in content:
                start = content.find('```json') + 7
                end = content.find('```', start)
                json_content = content[start:end].strip()
            elif '```' in content:
                start = content.find('```') + 3
                end = content.find('```', start)
                json_content = content[start:end].strip()
            else:
                json_content = content

            return json.loads(json_content)

        except Exception as e:
            self.get_logger().error(f'Multimodal reasoning error: {e}')
            return None

    def generate_action_plan(self, decision: Dict[str, Any], state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate action plan based on decision"""
        if not decision.get('feasibility', False):
            return {
                'plan': [],
                'reason': 'Request not feasible',
                'decision': decision
            }

        # Build planning prompt
        prompt = f"""
        Based on this decision and current state, generate an action plan:

        Decision: {json.dumps(decision, indent=2)}
        Current State: {json.dumps(state, indent=2)}

        Create a detailed action plan as JSON:
        {{
            "plan": [
                {{
                    "step": 1,
                    "action": "action_type",
                    "parameters": {{"param1": "value1"}},
                    "reason": "why this action is needed",
                    "expected_outcome": "what should happen",
                    "safety_check": "what to verify"
                }}
            ],
            "estimated_time": "time in seconds",
            "success_criteria": "how to verify completion",
            "fallback_actions": ["list of alternatives if primary fails"]
        }}
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )

            content = response.choices[0].message.content

            # Extract JSON
            if '```json' in content:
                start = content.find('```json') + 7
                end = content.find('```', start)
                json_content = content[start:end].strip()
            elif '```' in content:
                start = content.find('```') + 3
                end = content.find('```', start)
                json_content = content[start:end].strip()
            else:
                json_content = content

            plan_data = json.loads(json_content)
            plan_data['decision'] = decision  # Include original decision

            return plan_data

        except Exception as e:
            self.get_logger().error(f'Action planning error: {e}')
            return None

    def validate_multimodal_integration(self, decision: Dict[str, Any]) -> bool:
        """Validate that the decision properly integrates multiple modalities"""
        required_elements = ['reasoning', 'feasibility', 'confidence']

        for element in required_elements:
            if element not in decision:
                return False

        # Check that reasoning mentions multiple modalities
        reasoning = decision.get('reasoning', '').lower()
        modalities_mentioned = 0

        if any(modality in reasoning for modality in ['vision', 'visual', 'see', 'image', 'camera']):
            modalities_mentioned += 1
        if any(modality in reasoning for modality in ['language', 'text', 'command', 'understand']):
            modalities_mentioned += 1
        if any(modality in reasoning for modality in ['position', 'location', 'pose', 'spatial', 'navigation']):
            modalities_mentioned += 1
        if any(modality in reasoning for modality in ['scan', 'obstacle', 'laser', 'distance', 'safety']):
            modalities_mentioned += 1

        # Should mention at least 2 modalities
        return modalities_mentioned >= 2

def main(args=None):
    rclpy.init(args=args)
    node = MultimodalReasoningNode()

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

## Real-Time Multimodal Fusion

### Synchronized Processing Pipeline

```python
# File: multimodal_integration/synchronization.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from typing import Dict, Any, Optional
import time
from collections import deque
import threading

class MultimodalSynchronizerNode(Node):
    def __init__(self):
        super().__init__('multimodal_synchronizer_node')

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.pose_sub = self.create_subscription(
            PoseStamped, '/current_pose', self.pose_callback, 10)
        self.command_sub = self.create_subscription(
            String, '/user/command', self.command_callback, 10)

        # Publishers
        self.fused_output_pub = self.create_publisher(
            String, '/multimodal/synchronized_data', 10)

        # Buffers for synchronization
        self.image_buffer = deque(maxlen=10)
        self.scan_buffer = deque(maxlen=10)
        self.pose_buffer = deque(maxlen=10)
        self.command_buffer = deque(maxlen=5)

        # Timestamp tolerance for synchronization (in seconds)
        self.sync_tolerance = 0.1

        # Lock for thread safety
        self.buffer_lock = threading.Lock()

        # Timer for periodic fusion
        self.fusion_timer = self.create_timer(0.2, self.perform_fusion)

        self.get_logger().info('Multimodal Synchronizer Node Started')

    def image_callback(self, msg):
        """Add image to buffer"""
        with self.buffer_lock:
            self.image_buffer.append({
                'data': msg,
                'timestamp': self.get_timestamp_from_msg(msg)
            })

    def scan_callback(self, msg):
        """Add scan to buffer"""
        with self.buffer_lock:
            self.scan_buffer.append({
                'data': msg,
                'timestamp': self.get_timestamp_from_msg(msg)
            })

    def pose_callback(self, msg):
        """Add pose to buffer"""
        with self.buffer_lock:
            self.pose_buffer.append({
                'data': msg,
                'timestamp': self.get_timestamp_from_msg(msg)
            })

    def command_callback(self, msg):
        """Add command to buffer"""
        with self.buffer_lock:
            self.command_buffer.append({
                'data': msg,
                'timestamp': time.time()  # Commands use system time
            })

    def perform_fusion(self):
        """Perform multimodal fusion with temporal synchronization"""
        with self.buffer_lock:
            # Find temporally aligned data
            aligned_data = self.find_aligned_data()

        if aligned_data:
            # Perform fusion
            fused_result = self.fuse_data(aligned_data)

            if fused_result:
                # Publish fused result
                fused_msg = String()
                fused_msg.data = fused_result
                self.fused_output_pub.publish(fused_msg)

    def find_aligned_data(self) -> Optional[Dict[str, Any]]:
        """Find data that is temporally aligned"""
        current_time = time.time()

        # Find latest data of each type within tolerance
        aligned = {}

        # Find latest image
        for item in reversed(self.image_buffer):
            if current_time - item['timestamp'] <= self.sync_tolerance:
                aligned['image'] = item['data']
                break

        # Find latest scan
        for item in reversed(self.scan_buffer):
            if current_time - item['timestamp'] <= self.sync_tolerance:
                aligned['scan'] = item['data']
                break

        # Find latest pose
        for item in reversed(self.pose_buffer):
            if current_time - item['timestamp'] <= self.sync_tolerance:
                aligned['pose'] = item['data']
                break

        # Use the most recent command (if any)
        if self.command_buffer:
            aligned['command'] = self.command_buffer[-1]['data']

        # Must have at least image and scan for meaningful fusion
        if 'image' in aligned and 'scan' in aligned:
            return aligned

        return None

    def fuse_data(self, aligned_data: Dict[str, Any]) -> Optional[str]:
        """Fuse synchronized multimodal data"""
        try:
            fusion_result = {
                'timestamp': time.time(),
                'fused_data': {
                    'image_info': {
                        'encoding': aligned_data['image'].encoding,
                        'dimensions': [aligned_data['image'].width, aligned_data['image'].height],
                        'timestamp': self.get_timestamp_from_msg(aligned_data['image'])
                    },
                    'scan_info': {
                        'ranges_count': len(aligned_data['scan'].ranges),
                        'range_min': aligned_data['scan'].range_min,
                        'range_max': aligned_data['scan'].range_max,
                        'timestamp': self.get_timestamp_from_msg(aligned_data['scan'])
                    },
                    'pose_info': {
                        'position': [
                            aligned_data['pose'].pose.position.x,
                            aligned_data['pose'].pose.position.y,
                            aligned_data['pose'].pose.position.z
                        ],
                        'orientation': [
                            aligned_data['pose'].pose.orientation.x,
                            aligned_data['pose'].pose.orientation.y,
                            aligned_data['pose'].pose.orientation.z,
                            aligned_data['pose'].pose.orientation.w
                        ],
                        'timestamp': self.get_timestamp_from_msg(aligned_data['pose'])
                    }
                },
                'has_command': 'command' in aligned_data
            }

            return json.dumps(fusion_result)

        except Exception as e:
            self.get_logger().error(f'Fusion error: {e}')
            return None

    def get_timestamp_from_msg(self, msg) -> float:
        """Extract timestamp from ROS message"""
        if hasattr(msg.header, 'stamp'):
            return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) / 1e9
        else:
            return time.time()  # Fallback to current time

import json

def main(args=None):
    rclpy.init(args=args)
    node = MultimodalSynchronizerNode()

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

## Best Practices for Multimodal Integration

### 1. Temporal Synchronization
- Align data from different modalities based on timestamps
- Use appropriate buffer sizes to handle latency differences
- Implement interpolation for high-frequency modalities

### 2. Uncertainty Management
- Track uncertainty in each modality
- Use probabilistic fusion methods
- Implement confidence-based decision making

### 3. Computational Efficiency
- Use appropriate processing frequencies for each modality
- Implement early fusion vs. late fusion strategies
- Optimize for real-time performance requirements

### 4. Robustness
- Handle missing modality data gracefully
- Implement fallback strategies
- Validate cross-modal consistency

## Chapter Summary

Multimodal integration enables Physical AI systems to process and combine information from multiple sensory modalities simultaneously. The VLA framework provides a unified approach to integrating vision, language, and action capabilities. Successful multimodal systems require careful attention to temporal synchronization, cross-modal grounding, uncertainty management, and real-time performance optimization. The integration of these modalities enables robots to understand complex, natural commands and respond appropriately to their environment.

## Exercises

1. Implement a multimodal system that can respond to commands like "Bring me the red cup on the table."
2. Create a vision-language grounding system that can identify and manipulate specific objects.
3. Build a cross-modal reasoning system that combines perception and language for decision making.

## Next Steps

In the next chapter, we'll work on the CAPSTONE project that integrates all the VLA concepts learned in Module 4 into a complete Physical AI system.