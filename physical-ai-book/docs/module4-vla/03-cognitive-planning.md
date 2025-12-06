---
sidebar_position: 3
---

# Cognitive Planning with LLMs

## Chapter Objectives

By the end of this chapter, you will be able to:
- Design cognitive architectures that integrate LLMs with robotic planning
- Implement hierarchical task planning using LLM reasoning
- Create context-aware planning systems for Physical AI
- Develop multi-step reasoning capabilities for complex tasks
- Build robust planning systems that handle uncertainty and failures

## Cognitive Planning Overview

### What is Cognitive Planning?

Cognitive planning in Physical AI refers to the high-level reasoning process that enables robots to understand complex tasks, break them down into manageable steps, and execute them in a context-aware manner. When enhanced with Large Language Models (LLMs), cognitive planning gains the ability to:

- Understand natural language task descriptions
- Apply common sense reasoning
- Handle ambiguous or incomplete instructions
- Adapt plans based on changing circumstances
- Learn from experience and improve over time

### Cognitive Planning vs Traditional Planning

Traditional robotic planning focuses on:
- Low-level path planning and motion planning
- Deterministic state transitions
- Predefined action sequences
- Reactive behavior to environmental changes

Cognitive planning with LLMs adds:
- High-level task understanding
- Commonsense reasoning about the world
- Natural language interaction
- Adaptive and flexible behavior
- Learning from experience

## Cognitive Architecture

### Hierarchical Planning Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                    Task Level (LLM)                             │
│  - High-level task decomposition                                │
│  - Goal reasoning                                               │
│  - Context understanding                                        │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                 Action Level (Planner)                          │
│  - Action sequence generation                                   │
│  - Constraint satisfaction                                      │
│  - Resource allocation                                          │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                Execution Level (Robot)                          │
│  - Low-level motion control                                     │
│  - Sensor feedback processing                                   │
│  - Real-time adaptation                                         │
└─────────────────────────────────────────────────────────────────┘
```

### Cognitive Planning Components

1. **Goal Parser**: Interprets high-level goals from natural language
2. **World Model**: Maintains understanding of the environment and robot state
3. **Task Decomposer**: Breaks complex tasks into subtasks
4. **Action Planner**: Generates executable action sequences
5. **Context Manager**: Tracks relevant context and constraints
6. **Learning Module**: Updates knowledge based on experience

## LLM-Enhanced Task Decomposition

### Hierarchical Task Networks (HTN) with LLMs

```python
# File: cognitive_planning/htn_planner.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
import json
import openai
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

class TaskType(Enum):
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    PERCEPTION = "perception"
    COMPOSITE = "composite"

@dataclass
class Task:
    id: str
    type: TaskType
    description: str
    parameters: Dict[str, Any]
    dependencies: List[str]
    status: str = "pending"  # pending, executing, completed, failed

class HTNPlannerNode(Node):
    def __init__(self):
        super().__init__('htn_planner_node')

        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key='your-api-key')

        # Subscribers
        self.task_sub = self.create_subscription(
            String, '/cognitive/task', self.task_callback, 10)

        # Publishers
        self.subtask_pub = self.create_publisher(
            String, '/cognitive/subtasks', 10)
        self.action_pub = self.create_publisher(
            String, '/robot/action', 10)

        # State
        self.current_tasks = {}
        self.world_model = WorldModel()

        self.get_logger().info('HTN Planner Node Started')

    def task_callback(self, msg):
        """Process high-level task request"""
        try:
            task_request = json.loads(msg.data)
            task_description = task_request['description']
            task_context = task_request.get('context', {})

            # Use LLM to decompose task
            subtasks = self.decompose_task(task_description, task_context)

            # Publish subtasks
            subtasks_msg = String()
            subtasks_msg.data = json.dumps({
                'original_task': task_description,
                'subtasks': [self.task_to_dict(task) for task in subtasks],
                'context': task_context
            })
            self.subtask_pub.publish(subtasks_msg)

            # Execute subtasks
            self.execute_subtasks(subtasks)

        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in task request')

    def decompose_task(self, task_description: str, context: Dict[str, Any]) -> List[Task]:
        """Use LLM to decompose high-level task into subtasks"""
        world_state = self.world_model.get_state()

        prompt = f"""
        Decompose the following task into a sequence of executable subtasks:
        Task: "{task_description}"

        Current world state: {world_state}
        Context: {context}

        Provide a JSON list of subtasks with the following structure:
        {{
            "id": "unique_id",
            "type": "navigation|manipulation|perception|composite",
            "description": "detailed description",
            "parameters": {{"key": "value"}},
            "dependencies": ["task_id_1", "task_id_2"]  // tasks that must complete first
        }}

        Consider:
        1. Physical constraints of the robot
        2. Environmental constraints
        3. Safety requirements
        4. Logical dependencies between tasks
        5. Available robot capabilities
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )

            subtasks_data = json.loads(response.choices[0].message.content)

            subtasks = []
            for task_data in subtasks_data:
                task = Task(
                    id=task_data['id'],
                    type=TaskType(task_data['type']),
                    description=task_data['description'],
                    parameters=task_data.get('parameters', {}),
                    dependencies=task_data.get('dependencies', [])
                )
                subtasks.append(task)

            return subtasks

        except Exception as e:
            self.get_logger().error(f'Task decomposition error: {e}')
            # Return fallback subtasks
            return self.create_fallback_subtasks(task_description)

    def create_fallback_subtasks(self, task_description: str) -> List[Task]:
        """Create fallback subtasks if LLM decomposition fails"""
        # Simple fallback based on keywords
        if 'navigate' in task_description.lower() or 'go to' in task_description.lower():
            return [Task(
                id='nav_1',
                type=TaskType.NAVIGATION,
                description='Navigate to specified location',
                parameters={'target_location': 'unknown'},
                dependencies=[]
            )]
        elif 'pick' in task_description.lower() or 'grasp' in task_description.lower():
            return [Task(
                id='manip_1',
                type=TaskType.MANIPULATION,
                description='Manipulate object',
                parameters={'action': 'pick', 'object': 'unknown'},
                dependencies=[]
            )]
        else:
            return [Task(
                id='default_1',
                type=TaskType.PERCEPTION,
                description='Perceive environment',
                parameters={},
                dependencies=[]
            )]

    def execute_subtasks(self, subtasks: List[Task]):
        """Execute subtasks in dependency order"""
        # Sort tasks by dependencies
        sorted_tasks = self.topological_sort(subtasks)

        for task in sorted_tasks:
            self.execute_task(task)

    def topological_sort(self, tasks: List[Task]) -> List[Task]:
        """Sort tasks in dependency order using topological sort"""
        # Create adjacency list
        graph = {task.id: [] for task in tasks}
        for task in tasks:
            for dep_id in task.dependencies:
                if dep_id in graph:
                    graph[dep_id].append(task.id)

        # Perform topological sort
        visited = set()
        result = []

        def dfs(task_id):
            if task_id in visited:
                return
            visited.add(task_id)

            for neighbor in graph[task_id]:
                dfs(neighbor)

            # Find task object and add to result
            for task in tasks:
                if task.id == task_id:
                    result.append(task)
                    break

        for task in tasks:
            if task.id not in visited:
                dfs(task.id)

        return result

    def execute_task(self, task: Task):
        """Execute a single task"""
        self.get_logger().info(f'Executing task: {task.description}')

        if task.type == TaskType.NAVIGATION:
            self.execute_navigation_task(task)
        elif task.type == TaskType.MANIPULATION:
            self.execute_manipulation_task(task)
        elif task.type == TaskType.PERCEPTION:
            self.execute_perception_task(task)
        else:
            self.get_logger().warn(f'Unknown task type: {task.type}')

        task.status = 'completed'

    def execute_navigation_task(self, task: Task):
        """Execute navigation task"""
        target = task.parameters.get('target_location', 'unknown')
        self.get_logger().info(f'Navigating to {target}')

        # Publish navigation goal
        goal_msg = PoseStamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = 'map'
        # Set target coordinates based on task parameters
        goal_msg.pose.position.x = task.parameters.get('x', 0.0)
        goal_msg.pose.position.y = task.parameters.get('y', 0.0)
        goal_msg.pose.position.z = 0.0
        goal_msg.pose.orientation.w = 1.0

        # In real implementation, this would publish to navigation stack
        pass

    def execute_manipulation_task(self, task: Task):
        """Execute manipulation task"""
        action = task.parameters.get('action', 'unknown')
        obj = task.parameters.get('object', 'unknown')
        self.get_logger().info(f'Performing {action} on {obj}')

        # In real implementation, this would call manipulation services
        pass

    def execute_perception_task(self, task: Task):
        """Execute perception task"""
        perception_type = task.parameters.get('type', 'object_detection')
        self.get_logger().info(f'Performing {perception_type}')

        # In real implementation, this would call perception services
        pass

    def task_to_dict(self, task: Task) -> Dict[str, Any]:
        """Convert task to dictionary for JSON serialization"""
        return {
            'id': task.id,
            'type': task.type.value,
            'description': task.description,
            'parameters': task.parameters,
            'dependencies': task.dependencies,
            'status': task.status
        }

class WorldModel:
    """Maintains current state of the world for planning"""
    def __init__(self):
        self.robot_pose = None
        self.objects = {}
        self.environment = {}
        self.robot_capabilities = {
            'navigation': True,
            'manipulation': False,  # Default to no manipulation
            'perception': True
        }

    def get_state(self) -> Dict[str, Any]:
        """Get current world state as dictionary"""
        return {
            'robot_pose': self.robot_pose,
            'objects': self.objects,
            'environment': self.environment,
            'robot_capabilities': self.robot_capabilities
        }

    def update_robot_pose(self, pose):
        """Update robot position in world model"""
        self.robot_pose = pose

    def update_object(self, object_id: str, properties: Dict[str, Any]):
        """Update object information in world model"""
        self.objects[object_id] = properties

def main(args=None):
    rclpy.init(args=args)
    node = HTNPlannerNode()

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

## Context-Aware Planning

### Dynamic Context Management

```python
# File: cognitive_planning/context_manager.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
import json
from typing import Dict, Any, List
from datetime import datetime
import threading

class ContextManagerNode(Node):
    def __init__(self):
        super().__init__('context_manager_node')

        # Subscribers
        self.task_sub = self.create_subscription(
            String, '/cognitive/task', self.task_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.pose_sub = self.create_subscription(
            PoseStamped, '/current_pose', self.pose_callback, 10)
        self.perception_sub = self.create_subscription(
            String, '/perception/objects', self.perception_callback, 10)

        # Publishers
        self.context_pub = self.create_publisher(
            String, '/cognitive/context', 10)

        # Context storage
        self.context = {
            'timestamp': datetime.now().isoformat(),
            'robot_state': {
                'position': [0, 0, 0],
                'orientation': [0, 0, 0, 1],
                'battery_level': 100,
                'status': 'idle'
            },
            'environment': {
                'obstacles': [],
                'free_space': [],
                'navigation_zones': {}
            },
            'objects': {},
            'constraints': {},
            'history': []
        }

        # Thread lock for context updates
        self.context_lock = threading.Lock()

        # Timer for context updates
        self.context_timer = self.create_timer(1.0, self.publish_context)

        self.get_logger().info('Context Manager Node Started')

    def task_callback(self, msg):
        """Update context with new task information"""
        try:
            task_data = json.loads(msg.data)
            with self.context_lock:
                self.context['history'].append({
                    'type': 'task',
                    'data': task_data,
                    'timestamp': datetime.now().isoformat()
                })
                self.context['timestamp'] = datetime.now().isoformat()
        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in task message')

    def scan_callback(self, msg):
        """Update context with obstacle information"""
        with self.context_lock:
            # Process laser scan to identify obstacles
            obstacles = []
            for i, range_val in enumerate(msg.ranges):
                if msg.range_min <= range_val <= msg.range_max:
                    angle = msg.angle_min + i * msg.angle_increment
                    x = range_val * math.cos(angle)
                    y = range_val * math.sin(angle)
                    obstacles.append({'x': x, 'y': y, 'distance': range_val})

            self.context['environment']['obstacles'] = obstacles
            self.context['timestamp'] = datetime.now().isoformat()

    def pose_callback(self, msg):
        """Update context with robot position"""
        with self.context_lock:
            self.context['robot_state']['position'] = [
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z
            ]
            self.context['robot_state']['orientation'] = [
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w
            ]
            self.context['timestamp'] = datetime.now().isoformat()

    def perception_callback(self, msg):
        """Update context with object detection results"""
        try:
            objects_data = json.loads(msg.data)
            with self.context_lock:
                self.context['objects'] = objects_data
                self.context['timestamp'] = datetime.now().isoformat()
        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in perception message')

    def publish_context(self):
        """Publish current context"""
        with self.context_lock:
            context_msg = String()
            context_msg.data = json.dumps(self.context)
            self.context_pub.publish(context_msg)

    def get_context_prompt(self) -> str:
        """Generate context prompt for LLM"""
        with self.context_lock:
            context_str = f"""
            Current Context:
            - Robot Position: {self.context['robot_state']['position']}
            - Robot Status: {self.context['robot_state']['status']}
            - Battery Level: {self.context['robot_state']['battery_level']}%
            - Obstacles Detected: {len(self.context['environment']['obstacles'])}
            - Objects in Environment: {list(self.context['objects'].keys())}
            - Navigation Constraints: {self.context['constraints']}
            - Recent Actions: {[h['type'] for h in self.context['history'][-5:]]}
            """
            return context_str

    def add_constraint(self, constraint_type: str, constraint_value: Any):
        """Add constraint to context"""
        with self.context_lock:
            if constraint_type not in self.context['constraints']:
                self.context['constraints'][constraint_type] = []
            self.context['constraints'][constraint_type].append(constraint_value)
            self.context['timestamp'] = datetime.now().isoformat()

import math

def main(args=None):
    rclpy.init(args=args)
    node = ContextManagerNode()

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

## Multi-Step Reasoning with LLMs

### Chain-of-Thought Planning

```python
# File: cognitive_planning/reasoning_engine.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import openai
from typing import Dict, Any, List, Tuple

class ReasoningEngineNode(Node):
    def __init__(self):
        super().__init__('reasoning_engine_node')

        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key='your-api-key')

        # Subscribers
        self.task_sub = self.create_subscription(
            String, '/cognitive/high_level_task', self.task_callback, 10)
        self.context_sub = self.create_subscription(
            String, '/cognitive/context', self.context_callback, 10)

        # Publishers
        self.plan_pub = self.create_publisher(
            String, '/cognitive/reasoned_plan', 10)

        # State
        self.current_context = {}

        self.get_logger().info('Reasoning Engine Node Started')

    def context_callback(self, msg):
        """Update current context"""
        try:
            self.current_context = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in context message')

    def task_callback(self, msg):
        """Process high-level task with multi-step reasoning"""
        try:
            task_request = json.loads(msg.data)
            task_description = task_request['task']
            goal = task_request['goal']

            # Perform chain-of-thought reasoning
            reasoned_plan = self.reason_about_task(task_description, goal)

            # Publish reasoned plan
            plan_msg = String()
            plan_msg.data = json.dumps(reasoned_plan)
            self.plan_pub.publish(plan_msg)

        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in task request')

    def reason_about_task(self, task: str, goal: str) -> Dict[str, Any]:
        """Use LLM for multi-step reasoning about task execution"""
        context = self.current_context

        # Chain-of-thought prompting
        cot_prompt = f"""
        Task: {task}
        Goal: {goal}

        Current Context:
        {json.dumps(context, indent=2)}

        Let's think step by step to create a plan:

        1. What is the current situation?
        2. What are the requirements to achieve the goal?
        3. What obstacles or constraints exist?
        4. What sequence of actions would achieve the goal?
        5. What are potential failure points and how to handle them?

        Provide your reasoning and then a structured plan in JSON format:
        {{
            "reasoning": "step-by-step reasoning process",
            "plan": [
                {{
                    "step": 1,
                    "action": "action_description",
                    "reason": "why this action is needed",
                    "expected_outcome": "what should happen",
                    "safety_check": "what to verify before/after"
                }}
            ],
            "constraints": ["list of constraints to consider"],
            "fallbacks": ["list of alternative approaches if primary plan fails"]
        }}
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",  # Using GPT-4 for better reasoning
                messages=[{"role": "user", "content": cot_prompt}],
                temperature=0.3
            )

            content = response.choices[0].message.content

            # Extract JSON from response if wrapped in code blocks
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
            self.get_logger().error(f'Reasoning error: {e}')
            # Return fallback plan
            return {
                "reasoning": "Error in reasoning, using fallback",
                "plan": [{"step": 1, "action": "perceive_environment", "reason": "gather information", "expected_outcome": "environment understanding", "safety_check": "obstacle detection"}],
                "constraints": ["safety", "navigation", "manipulation"],
                "fallbacks": ["simple_navigation", "request_help"]
            }

    def validate_plan(self, plan: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate the generated plan"""
        # Check if plan has required structure
        required_keys = ['plan', 'reasoning']
        for key in required_keys:
            if key not in plan:
                return False, f"Missing required key: {key}"

        # Check if plan has steps
        if not isinstance(plan['plan'], list) or len(plan['plan']) == 0:
            return False, "Plan must contain at least one step"

        # Validate each step
        for i, step in enumerate(plan['plan']):
            required_step_keys = ['action', 'reason']
            for key in required_step_keys:
                if key not in step:
                    return False, f"Step {i} missing required key: {key}"

        return True, "Plan is valid"

def main(args=None):
    rclpy.init(args=args)
    node = ReasoningEngineNode()

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

## Learning and Adaptation

### Experience-Based Planning Improvement

```python
# File: cognitive_planning/learning_module.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_msgs.msg import Bool
import json
import pickle
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Experience:
    task: str
    plan: List[Dict[str, Any]]
    outcome: str  # success, partial_success, failure
    feedback: str
    timestamp: str
    execution_time: float

class LearningModuleNode(Node):
    def __init__(self):
        super().__init__('learning_module_node')

        # Subscribers
        self.plan_sub = self.create_subscription(
            String, '/cognitive/reasoned_plan', self.plan_callback, 10)
        self.outcome_sub = self.create_subscription(
            String, '/cognitive/execution_outcome', self.outcome_callback, 10)
        self.feedback_sub = self.create_subscription(
            String, '/user/feedback', self.feedback_callback, 10)

        # Publishers
        self.adapted_plan_pub = self.create_publisher(
            String, '/cognitive/adapted_plan', 10)

        # State
        self.experiences: List[Experience] = []
        self.current_plan = None
        self.current_task = None
        self.start_time = None

        # Load previous experiences
        self.load_experiences()

        # Timer for periodic learning updates
        self.learning_timer = self.create_timer(30.0, self.update_knowledge)

        self.get_logger().info('Learning Module Node Started')

    def plan_callback(self, msg):
        """Store plan for learning"""
        try:
            plan_data = json.loads(msg.data)
            self.current_plan = plan_data
            self.current_task = plan_data.get('task', 'unknown')
            self.start_time = datetime.now()
        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in plan message')

    def outcome_callback(self, msg):
        """Process execution outcome for learning"""
        try:
            outcome_data = json.loads(msg.data)
            outcome = outcome_data['outcome']
            task = outcome_data.get('task', self.current_task)

            if self.start_time:
                execution_time = (datetime.now() - self.start_time).total_seconds()
            else:
                execution_time = 0

            experience = Experience(
                task=task,
                plan=self.current_plan,
                outcome=outcome,
                feedback=outcome_data.get('feedback', ''),
                timestamp=datetime.now().isoformat(),
                execution_time=execution_time
            )

            self.experiences.append(experience)
            self.current_plan = None
            self.current_task = None
            self.start_time = None

            # Save experience
            self.save_experiences()

            self.get_logger().info(f'Learned from experience: {outcome} for task {task}')

        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in outcome message')

    def feedback_callback(self, msg):
        """Process user feedback for learning"""
        feedback = msg.data
        if self.current_plan:
            # Add feedback to current experience when it completes
            pass

    def update_knowledge(self):
        """Periodically update knowledge based on experiences"""
        if len(self.experiences) < 10:  # Need sufficient experiences
            return

        # Analyze patterns in experiences
        patterns = self.analyze_patterns()

        # Update planning strategies based on patterns
        self.update_planning_strategies(patterns)

        self.get_logger().info(f'Updated knowledge from {len(self.experiences)} experiences')

    def analyze_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in experiences"""
        patterns = {
            'task_success_rates': {},
            'common_failure_modes': [],
            'effective_strategies': [],
            'ineffective_strategies': []
        }

        # Calculate success rates by task type
        task_outcomes = {}
        for exp in self.experiences:
            task = exp.task
            if task not in task_outcomes:
                task_outcomes[task] = {'success': 0, 'total': 0}

            task_outcomes[task]['total'] += 1
            if exp.outcome == 'success':
                task_outcomes[task]['success'] += 1

        for task, outcomes in task_outcomes.items():
            success_rate = outcomes['success'] / outcomes['total']
            patterns['task_success_rates'][task] = success_rate

        # Identify common failure modes
        failures = [exp for exp in self.experiences if exp.outcome == 'failure']
        for failure in failures:
            # Analyze what went wrong
            patterns['common_failure_modes'].append({
                'task': failure.task,
                'plan_structure': [step['action'] for step in failure.plan.get('plan', [])],
                'feedback': failure.feedback
            })

        return patterns

    def update_planning_strategies(self, patterns: Dict[str, Any]):
        """Update planning strategies based on learned patterns"""
        # This would update the planning process based on learned patterns
        # For example, if certain task types have low success rates,
        # modify the planning approach for those tasks

        # Example: If navigation tasks have low success, add more safety checks
        if patterns['task_success_rates'].get('navigation', 1.0) < 0.7:
            self.get_logger().info('Low navigation success rate detected, updating strategies')

    def adapt_plan(self, original_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt plan based on learned experiences"""
        adapted_plan = original_plan.copy()

        # Apply learned improvements
        if 'plan' in adapted_plan:
            for i, step in enumerate(adapted_plan['plan']):
                # Apply learned modifications based on task type and context
                adapted_plan['plan'][i] = self.adapt_step(step)

        return adapted_plan

    def adapt_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt individual step based on learning"""
        # Example: Add extra safety checks for actions that previously failed
        adapted_step = step.copy()

        # Check if this type of action has failed before
        action_type = step.get('action', '')
        if self.has_failed_action(action_type):
            # Add safety verification
            if 'safety_check' not in adapted_step:
                adapted_step['safety_check'] = []
            adapted_step['safety_check'].append('environment_verification')

        return adapted_step

    def has_failed_action(self, action_type: str) -> bool:
        """Check if action type has failed in past experiences"""
        for exp in self.experiences:
            if exp.outcome == 'failure':
                for step in exp.plan.get('plan', []):
                    if step.get('action') == action_type:
                        return True
        return False

    def save_experiences(self):
        """Save experiences to persistent storage"""
        try:
            with open('/tmp/robot_experiences.pkl', 'wb') as f:
                pickle.dump(self.experiences, f)
        except Exception as e:
            self.get_logger().error(f'Failed to save experiences: {e}')

    def load_experiences(self):
        """Load experiences from persistent storage"""
        try:
            if os.path.exists('/tmp/robot_experiences.pkl'):
                with open('/tmp/robot_experiences.pkl', 'rb') as f:
                    self.experiences = pickle.load(f)
                self.get_logger().info(f'Loaded {len(self.experiences)} experiences')
        except Exception as e:
            self.get_logger().info(f'No previous experiences found: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = LearningModuleNode()

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

## Failure Handling and Recovery

### Robust Planning with Fallbacks

```python
# File: cognitive_planning/failure_recovery.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_msgs.msg import Bool
import json
from typing import Dict, Any, List, Optional
from enum import Enum
import time

class RecoveryStrategy(Enum):
    RETRY = "retry"
    SIMPLIFY = "simplify"
    ALTERNATIVE = "alternative"
    ESCALATE = "escalate"

class FailureRecoveryNode(Node):
    def __init__(self):
        super().__init__('failure_recovery_node')

        # Subscribers
        self.failure_sub = self.create_subscription(
            String, '/cognitive/failure', self.failure_callback, 10)
        self.status_sub = self.create_subscription(
            String, '/robot/status', self.status_callback, 10)

        # Publishers
        self.recovery_plan_pub = self.create_publisher(
            String, '/cognitive/recovery_plan', 10)
        self.emergency_stop_pub = self.create_publisher(
            Bool, '/emergency_stop', 10)

        # State
        self.failure_history = []
        self.current_recovery_level = 0
        self.max_recovery_attempts = 3

        self.get_logger().info('Failure Recovery Node Started')

    def failure_callback(self, msg):
        """Handle failure and generate recovery plan"""
        try:
            failure_data = json.loads(msg.data)
            failure_type = failure_data['type']
            failure_context = failure_data.get('context', {})
            original_task = failure_data.get('original_task', 'unknown')

            self.get_logger().warn(f'Failure detected: {failure_type}')

            # Record failure
            self.failure_history.append({
                'type': failure_type,
                'context': failure_context,
                'timestamp': time.time(),
                'original_task': original_task
            })

            # Generate recovery plan
            recovery_plan = self.generate_recovery_plan(
                failure_type, failure_context, original_task
            )

            # Publish recovery plan
            recovery_msg = String()
            recovery_msg.data = json.dumps(recovery_plan)
            self.recovery_plan_pub.publish(recovery_msg)

        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in failure message')

    def generate_recovery_plan(self, failure_type: str, context: Dict[str, Any], original_task: str) -> Dict[str, Any]:
        """Generate recovery plan based on failure type"""
        recovery_strategy = self.select_recovery_strategy(failure_type, context)

        if recovery_strategy == RecoveryStrategy.RETRY:
            return self.create_retry_plan(failure_type, context, original_task)
        elif recovery_strategy == RecoveryStrategy.SIMPLIFY:
            return self.create_simplification_plan(failure_type, context, original_task)
        elif recovery_strategy == RecoveryStrategy.ALTERNATIVE:
            return self.create_alternative_plan(failure_type, context, original_task)
        elif recovery_strategy == RecoveryStrategy.ESCALATE:
            return self.create_escalation_plan(failure_type, context, original_task)
        else:
            return self.create_default_recovery_plan(failure_type, context, original_task)

    def select_recovery_strategy(self, failure_type: str, context: Dict[str, Any]) -> RecoveryStrategy:
        """Select appropriate recovery strategy based on failure type and context"""
        # Count recent failures of same type
        recent_failures = [
            f for f in self.failure_history
            if f['type'] == failure_type and time.time() - f['timestamp'] < 300  # 5 minutes
        ]

        if len(recent_failures) >= self.max_recovery_attempts:
            # Too many failures of same type, escalate
            return RecoveryStrategy.ESCALATE

        # Select strategy based on failure type
        strategy_map = {
            'navigation_failure': RecoveryStrategy.ALTERNATIVE,
            'manipulation_failure': RecoveryStrategy.SIMPLIFY,
            'perception_failure': RecoveryStrategy.RETRY,
            'communication_failure': RecoveryStrategy.RETRY,
            'safety_violation': RecoveryStrategy.ESCALATE
        }

        return strategy_map.get(failure_type, RecoveryStrategy.RETRY)

    def create_retry_plan(self, failure_type: str, context: Dict[str, Any], original_task: str) -> Dict[str, Any]:
        """Create plan to retry the failed action"""
        return {
            'strategy': 'retry',
            'action': 'retry_original',
            'parameters': context,
            'max_attempts': 3,
            'delay': 2.0,  # 2 second delay before retry
            'original_task': original_task,
            'recovery_step': self.current_recovery_level
        }

    def create_simplification_plan(self, failure_type: str, context: Dict[str, Any], original_task: str) -> Dict[str, Any]:
        """Create plan to simplify the task"""
        return {
            'strategy': 'simplify',
            'action': 'simplified_task',
            'parameters': {
                'original_task': original_task,
                'simplified_goal': self.simplify_task(original_task)
            },
            'original_task': original_task,
            'recovery_step': self.current_recovery_level
        }

    def create_alternative_plan(self, failure_type: str, context: Dict[str, Any], original_task: str) -> Dict[str, Any]:
        """Create plan with alternative approach"""
        return {
            'strategy': 'alternative',
            'action': 'alternative_approach',
            'parameters': {
                'original_task': original_task,
                'alternative_methods': self.get_alternative_methods(original_task)
            },
            'original_task': original_task,
            'recovery_step': self.current_recovery_level
        }

    def create_escalation_plan(self, failure_type: str, context: Dict[str, Any], original_task: str) -> Dict[str, Any]:
        """Create plan to escalate to human operator"""
        # Trigger emergency stop if safety-related
        if 'safety' in failure_type.lower():
            emergency_msg = Bool()
            emergency_msg.data = True
            self.emergency_stop_pub.publish(emergency_msg)

        return {
            'strategy': 'escalate',
            'action': 'request_human_assistance',
            'parameters': {
                'failure_type': failure_type,
                'context': context,
                'original_task': original_task
            },
            'original_task': original_task,
            'recovery_step': self.current_recovery_level
        }

    def create_default_recovery_plan(self, failure_type: str, context: Dict[str, Any], original_task: str) -> Dict[str, Any]:
        """Create default recovery plan"""
        return {
            'strategy': 'default',
            'action': 'safe_return',
            'parameters': {
                'return_location': 'home_base',
                'safety_check': True
            },
            'original_task': original_task,
            'recovery_step': self.current_recovery_level
        }

    def simplify_task(self, task: str) -> str:
        """Simplify a complex task"""
        # Example simplifications
        if 'complex navigation' in task.lower():
            return task.replace('complex navigation', 'simple navigation')
        elif 'precise manipulation' in task.lower():
            return task.replace('precise manipulation', 'basic manipulation')
        else:
            return f"basic version of {task}"

    def get_alternative_methods(self, task: str) -> List[str]:
        """Get alternative methods for a task"""
        alternatives = {
            'navigation': ['alternative_path', 'different_approach', 'wait_and_retry'],
            'manipulation': ['different_grip', 'alternative_approach', 'tool_assistance'],
            'perception': ['different_sensor', 'change_viewpoint', 'illumination_change']
        }

        for key, methods in alternatives.items():
            if key in task.lower():
                return methods

        return ['general_alternative']

    def status_callback(self, msg):
        """Monitor robot status for proactive recovery"""
        try:
            status = json.loads(msg.data)
            robot_state = status.get('state', 'unknown')

            # Check for states that might indicate potential failures
            if robot_state in ['stuck', 'error', 'low_battery']:
                self.get_logger().info(f'Potential issue detected: {robot_state}')
                # Could trigger proactive recovery measures

        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in status message')

def main(args=None):
    rclpy.init(args=args)
    node = FailureRecoveryNode()

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

## Integration with Robot Control

### Planning-Execution Bridge

```python
# File: cognitive_planning/execution_bridge.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist, PoseStamped
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from typing import Dict, Any, Optional
import time
import threading

class ExecutionBridgeNode(Node):
    def __init__(self):
        super().__init__('execution_bridge_node')

        # Subscribers
        self.plan_sub = self.create_subscription(
            String, '/cognitive/reasoned_plan', self.plan_callback, 10)
        self.recovery_plan_sub = self.create_subscription(
            String, '/cognitive/recovery_plan', self.recovery_plan_callback, 10)

        # Publishers
        self.status_pub = self.create_publisher(
            String, '/cognitive/execution_status', 10)
        self.outcome_pub = self.create_publisher(
            String, '/cognitive/execution_outcome', 10)

        # Action clients
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # State
        self.current_plan = None
        self.plan_thread = None
        self.execution_active = False

        self.get_logger().info('Execution Bridge Node Started')

    def plan_callback(self, msg):
        """Execute a new plan"""
        try:
            plan = json.loads(msg.data)
            self.execute_plan(plan)
        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in plan message')

    def recovery_plan_callback(self, msg):
        """Execute a recovery plan"""
        try:
            recovery_plan = json.loads(msg.data)
            self.execute_recovery_plan(recovery_plan)
        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in recovery plan message')

    def execute_plan(self, plan: Dict[str, Any]):
        """Execute a cognitive plan"""
        if self.execution_active:
            self.get_logger().warn('Plan execution already active, skipping')
            return

        self.execution_active = True
        self.current_plan = plan

        # Execute in separate thread to avoid blocking
        self.plan_thread = threading.Thread(
            target=self.execute_plan_thread,
            args=(plan,)
        )
        self.plan_thread.start()

    def execute_plan_thread(self, plan: Dict[str, Any]):
        """Execute plan in separate thread"""
        start_time = time.time()
        success_count = 0
        total_steps = len(plan.get('plan', []))

        for i, step in enumerate(plan.get('plan', [])):
            self.publish_status(f'Executing step {i+1}/{total_steps}: {step.get("action", "unknown")}')

            try:
                step_success = self.execute_step(step)
                if step_success:
                    success_count += 1
                    self.get_logger().info(f'Step {i+1} completed successfully')
                else:
                    self.get_logger().warn(f'Step {i+1} failed')
                    # Could trigger recovery here
                    break

            except Exception as e:
                self.get_logger().error(f'Step {i+1} execution error: {e}')
                break

        execution_time = time.time() - start_time
        overall_success = success_count == total_steps

        # Publish outcome
        outcome = {
            'task': plan.get('task', 'unknown'),
            'outcome': 'success' if overall_success else 'partial_success' if success_count > 0 else 'failure',
            'success_steps': success_count,
            'total_steps': total_steps,
            'execution_time': execution_time,
            'timestamp': time.time()
        }

        outcome_msg = String()
        outcome_msg.data = json.dumps(outcome)
        self.outcome_pub.publish(outcome_msg)

        self.execution_active = False

    def execute_step(self, step: Dict[str, Any]) -> bool:
        """Execute a single step of the plan"""
        action = step.get('action', 'unknown')

        if action == 'navigate_to_pose':
            return self.execute_navigation_step(step)
        elif action == 'move_robot':
            return self.execute_move_step(step)
        elif action == 'perceive_environment':
            return self.execute_perception_step(step)
        elif action == 'wait':
            return self.execute_wait_step(step)
        else:
            self.get_logger().warn(f'Unknown action: {action}')
            return False

    def execute_navigation_step(self, step: Dict[str, Any]) -> bool:
        """Execute navigation step"""
        target = step.get('parameters', {}).get('target', {})

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.pose.position.x = target.get('x', 0.0)
        goal_msg.pose.pose.position.y = target.get('y', 0.0)
        goal_msg.pose.pose.position.z = target.get('z', 0.0)
        goal_msg.pose.pose.orientation.w = 1.0

        # Wait for action server
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Navigation action server not available')
            return False

        # Send goal
        future = self.nav_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Navigation goal rejected')
            return False

        # Get result
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result().result
        status = result_future.result().status

        return status == GoalStatus.STATUS_SUCCEEDED

    def execute_move_step(self, step: Dict[str, Any]) -> bool:
        """Execute simple movement step"""
        cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        cmd = Twist()
        cmd.linear.x = step.get('parameters', {}).get('linear_velocity', 0.0)
        cmd.angular.z = step.get('parameters', {}).get('angular_velocity', 0.0)

        # Execute for specified duration
        duration = step.get('parameters', {}).get('duration', 1.0)

        start_time = time.time()
        while time.time() - start_time < duration:
            cmd_pub.publish(cmd)
            time.sleep(0.1)

        # Stop
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        cmd_pub.publish(cmd)

        return True

    def execute_perception_step(self, step: Dict[str, Any]) -> bool:
        """Execute perception step"""
        # In real implementation, this would call perception services
        # For now, simulate perception by publishing a message
        self.get_logger().info('Performing perception task')
        return True

    def execute_wait_step(self, step: Dict[str, Any]) -> bool:
        """Execute wait step"""
        duration = step.get('parameters', {}).get('duration', 1.0)
        time.sleep(duration)
        return True

    def execute_recovery_plan(self, recovery_plan: Dict[str, Any]):
        """Execute a recovery plan"""
        strategy = recovery_plan.get('strategy', 'unknown')
        self.get_logger().info(f'Executing recovery strategy: {strategy}')

        if strategy == 'retry':
            self.execute_retry(recovery_plan)
        elif strategy == 'simplify':
            self.execute_simplification(recovery_plan)
        elif strategy == 'alternative':
            self.execute_alternative(recovery_plan)
        elif strategy == 'escalate':
            self.execute_escalation(recovery_plan)

    def execute_retry(self, recovery_plan: Dict[str, Any]):
        """Execute retry strategy"""
        # Implementation would retry the failed action
        pass

    def execute_simplification(self, recovery_plan: Dict[str, Any]):
        """Execute simplification strategy"""
        # Implementation would execute simplified version
        pass

    def execute_alternative(self, recovery_plan: Dict[str, Any]):
        """Execute alternative strategy"""
        # Implementation would try alternative approach
        pass

    def execute_escalation(self, recovery_plan: Dict[str, Any]):
        """Execute escalation strategy"""
        # Implementation would request human assistance
        pass

    def publish_status(self, status: str):
        """Publish execution status"""
        status_msg = String()
        status_msg.data = json.dumps({
            'status': status,
            'timestamp': time.time()
        })
        self.status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ExecutionBridgeNode()

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

## Best Practices for Cognitive Planning

### 1. Layered Architecture
- Separate high-level reasoning from low-level execution
- Use appropriate abstraction levels for different planning components
- Implement clear interfaces between components

### 2. Context Management
- Maintain comprehensive world state
- Update context in real-time as environment changes
- Use context for adaptive planning

### 3. Failure Handling
- Design robust failure detection mechanisms
- Implement multiple recovery strategies
- Learn from failures to improve future performance

### 4. Performance Optimization
- Cache frequently used plans and patterns
- Use appropriate planning horizons
- Balance planning quality with execution speed

## Chapter Summary

Cognitive planning with LLMs enables robots to perform high-level reasoning and task decomposition that goes beyond traditional planning approaches. By integrating LLMs with contextual awareness, learning capabilities, and robust failure handling, Physical AI systems can achieve more flexible and adaptive behavior. The key components include hierarchical task decomposition, context-aware reasoning, learning from experience, and robust execution with recovery mechanisms.

## Exercises

1. Implement a cognitive planning system that can decompose complex household tasks.
2. Add learning capabilities to improve planning based on execution outcomes.
3. Create a context-aware planning system that adapts to environmental changes.

## Next Steps

In the next chapter, we'll explore multimodal integration, learning how to combine vision, language, and action in Physical AI systems for more sophisticated capabilities.