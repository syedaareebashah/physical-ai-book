# Data Model: Physical AI & Humanoid Robotics Docusaurus Book

**Feature**: Physical AI & Humanoid Robotics Docusaurus Book
**Created**: 2025-12-06
**Author**: Claude Code

## Core Entities

### Physical AI
**Description**: A paradigm that integrates artificial intelligence with physical systems, enabling robots to interact with and understand the real world through sensors, actuators, and embodied intelligence.

**Attributes**:
- id: string (unique identifier)
- name: string (e.g., "Computer Vision AI", "Navigation AI")
- description: string
- concepts: string[] (related concepts like "perception", "control", "learning")
- applications: string[] (robotics domains where applied)
- methodologies: string[] (approaches like "deep learning", "classical control")

**Relationships**:
- connects to → Robotics Systems
- connects to → AI Algorithms
- connects to → Sensors
- connects to → Actuators

### Humanoid Robot
**Description**: A robot designed to mimic the human body, typically with a torso, head, two arms, and two legs, enabling human-like interaction with environments.

**Attributes**:
- id: string (unique identifier)
- name: string (e.g., "Atlas", "Pepper", "Nao")
- description: string
- structure: string (description of physical structure)
- joints: Joint[] (list of joint definitions)
- links: Link[] (list of link definitions)
- kinematics: string (forward/inverse kinematics approach)

**Relationships**:
- connects to → URDF Models
- connects to → Simulation Environments
- connects to → Control Systems
- connects to → AI Behaviors

### Joint
**Description**: A connection between two links in a robot that allows relative motion.

**Attributes**:
- id: string (unique identifier)
- name: string
- type: string (revolute, prismatic, continuous, etc.)
- parent_link: string
- child_link: string
- limits: JointLimits (position, velocity, effort)

### Link
**Description**: A rigid component of a robot that connects joints.

**Attributes**:
- id: string (unique identifier)
- name: string
- visual_mesh: string (path to visual mesh file)
- collision_mesh: string (path to collision mesh file)
- inertial_properties: Inertial (mass, center of mass, inertia tensor)

### JointLimits
**Description**: Defines the operational limits of a joint.

**Attributes**:
- lower: number (lower position limit)
- upper: number (upper position limit)
- velocity: number (maximum velocity)
- effort: number (maximum effort/torque)

### Inertial
**Description**: Physical properties of a link for simulation.

**Attributes**:
- mass: number
- center_of_mass: Vector3 (x, y, z)
- inertia_tensor: Matrix3x3 (3x3 inertia matrix)

### Vector3
**Description**: 3D vector representation.

**Attributes**:
- x: number
- y: number
- z: number

### Matrix3x3
**Description**: 3x3 matrix representation.

**Attributes**:
- xx: number, xy: number, xz: number
- yx: number, yy: number, yz: number
- zx: number, zy: number, zz: number

### ROS 2 (Robot Operating System 2)
**Description**: A flexible framework for writing robotic software that provides hardware abstraction, device drivers, libraries, and message-passing capabilities.

**Attributes**:
- id: string (unique identifier)
- name: string
- description: string
- nodes: ROSNode[] (list of nodes)
- topics: Topic[] (list of topics)
- services: Service[] (list of services)
- actions: Action[] (list of actions)
- parameters: Parameter[] (list of parameters)

**Relationships**:
- connects to → rclpy (Python client library)
- connects to → Publishers
- connects to → Subscribers

### ROSNode
**Description**: A computational process within the ROS 2 graph that performs specific tasks and communicates with other nodes.

**Attributes**:
- id: string (unique identifier)
- name: string
- namespace: string
- publishers: Publisher[]
- subscribers: Subscriber[]
- services: ServiceServer[]
- clients: ServiceClient[]
- actions: ActionServer[] | ActionClient[]

### Topic
**Description**: An asynchronous communication channel for one-to-many data streaming between nodes.

**Attributes**:
- id: string (unique identifier)
- name: string
- message_type: string
- qos_profile: QoSProfile

### Publisher
**Description**: A component that publishes messages to a topic.

**Attributes**:
- id: string (unique identifier)
- topic: string
- message_type: string
- qos_profile: QoSProfile

### Subscriber
**Description**: A component that subscribes to messages from a topic.

**Attributes**:
- id: string (unique identifier)
- topic: string
- message_type: string
- callback_function: string
- qos_profile: QoSProfile

### QoSProfile
**Description**: Quality of Service settings for ROS 2 communication.

**Attributes**:
- reliability: string (reliable, best_effort)
- durability: string (volatile, transient_local)
- history: string (keep_last, keep_all)
- depth: number

### Service
**Description**: A synchronous communication mechanism for request-response interactions between nodes.

**Attributes**:
- id: string (unique identifier)
- name: string
- request_type: string
- response_type: string

### Simulation Environment
**Description**: A virtual representation of the physical world used to test and validate robotic systems before deployment to real hardware.

**Attributes**:
- id: string (unique identifier)
- name: string (e.g., "Gazebo", "Unity", "Isaac Sim")
- description: string
- physics_engine: string
- supported_sensors: string[]
- environments: SimulationWorld[]
- robot_models: URDFModel[]

**Relationships**:
- connects to → Gazebo
- connects to → Unity
- connects to → Isaac Sim
- connects to → Robot Models

### SimulationWorld
**Description**: A specific environment within a simulation platform.

**Attributes**:
- id: string (unique identifier)
- name: string
- description: string
- physics_properties: PhysicsProperties
- lighting: LightingProperties

### PhysicsProperties
**Description**: Physical properties of a simulation environment.

**Attributes**:
- gravity: Vector3
- time_step: number
- solver_type: string

### LightingProperties
**Description**: Lighting configuration for a simulation environment.

**Attributes**:
- ambient_light: Color
- directional_lights: DirectionalLight[]
- point_lights: PointLight[]

### Color
**Description**: RGB color representation.

**Attributes**:
- r: number (0-1)
- g: number (0-1)
- b: number (0-1)
- a: number (0-1)

### Vision-Language-Action (VLA) System
**Description**: An integrated system that combines visual perception, language understanding, and physical action to enable complex robotic behaviors.

**Attributes**:
- id: string (unique identifier)
- name: string
- description: string
- perception_modules: PerceptionModule[]
- language_modules: LanguageModule[]
- action_modules: ActionModule[]
- integration_approach: string

**Relationships**:
- connects to → LLMs (Large Language Models)
- connects to → Computer Vision Systems
- connects to → Control Systems
- connects to → Multimodal Processing

### PerceptionModule
**Description**: Component responsible for visual perception in VLA systems.

**Attributes**:
- id: string (unique identifier)
- name: string
- input_type: string (camera, lidar, etc.)
- output_type: string (features, objects, etc.)
- model_type: string (CNN, transformer, etc.)

### LanguageModule
**Description**: Component responsible for language understanding in VLA systems.

**Attributes**:
- id: string (unique identifier)
- name: string
- input_type: string (text, speech)
- output_type: string (intent, command, etc.)
- model_type: string (LLM, NLP model, etc.)

### ActionModule
**Description**: Component responsible for physical action in VLA systems.

**Attributes**:
- id: string (unique identifier)
- name: string
- input_type: string (command, plan)
- output_type: string (motor commands, etc.)
- execution_type: string (motion, manipulation, etc.)

### Learning Module
**Description**: A structured educational unit containing theory, implementation, exercises, and assessments focused on specific robotics concepts.

**Attributes**:
- id: string (unique identifier)
- title: string
- description: string
- objectives: string[] (learning objectives)
- content_path: string (path to MDX content)
- exercises: Exercise[]
- assessments: Assessment[]
- prerequisites: string[]
- estimated_time: number (in hours)
- difficulty_level: string (beginner, intermediate, advanced)

**Relationships**:
- connects to → Curriculum
- connects to → Progression Path
- connects to → Prerequisites

### Exercise
**Description**: A practical task designed to reinforce learning objectives.

**Attributes**:
- id: string (unique identifier)
- title: string
- description: string
- type: string (coding, conceptual, simulation, etc.)
- estimated_time: number (in minutes)
- difficulty_level: string (beginner, intermediate, advanced)
- solution_path: string (path to solution if applicable)

### Assessment
**Description**: A formal evaluation to validate student learning.

**Attributes**:
- id: string (unique identifier)
- title: string
- description: string
- passing_score: number (e.g., 80 for 80%)
- questions: Question[]
- time_limit: number (in minutes, 0 if untimed)

### Question
**Description**: A single item in an assessment.

**Attributes**:
- id: string (unique identifier)
- text: string
- type: string (multiple_choice, true_false, short_answer, etc.)
- options: string[] (for multiple choice)
- correct_answer: string | number
- points: number

### URDFModel
**Description**: A robot model defined in Unified Robot Description Format.

**Attributes**:
- id: string (unique identifier)
- name: string
- file_path: string (path to URDF file)
- links: Link[]
- joints: Joint[]
- materials: Material[]
- gazebo_extensions: GazeboExtension[]

### Material
**Description**: Visual material properties for URDF models.

**Attributes**:
- id: string (unique identifier)
- name: string
- color: Color
- texture_path: string (optional)

### GazeboExtension
**Description**: Gazebo-specific extensions for URDF models.

**Attributes**:
- id: string (unique identifier)
- plugin_type: string
- parameters: object (key-value pairs)
- gazebo_reference: string