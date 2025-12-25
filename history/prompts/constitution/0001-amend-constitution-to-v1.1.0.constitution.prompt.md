---
id: 0001
title: Amend constitution to v1.1.0
stage: constitution
date: 2025-12-04
surface: agent
model: claude-sonnet-4-5-20250929
feature: none
branch: master
user: User
command: /sp.constitution
labels: ["constitution", "update"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - .specify/memory/constitution.md
tests:

---

## Prompt

/sp.constitution
Project: Physical AI & Humanoid Robotics

Core Principles
- Embodied Intelligence & Real-World Functionality
  - Every learning outcome focuses on AI systems interacting with the physical world.
  - Students must demonstrate understanding of how digital intelligence controls physical movement.
  - All modules connect theory → simulation → real-world robotic behavior.

Theme
- AI Systems in the Physical World
  - Emphasis on bridging the gap between computational intelligence and physical embodiment in humanoid robots.

Goal
- Enable students to design, simulate, and deploy humanoid robotic systems capable of performing naturalistic human-like tasks using physical reasoning, perception, and action planning.
- Tools emphasized: ROS 2, Gazebo, Unity, NVIDIA Isaac, LLM integrations, and multimodal perception stacks.

Quarter Overview
- The quarter introduces Physical AI: AI systems that operate in real environments, sense physical forces, and make real-time decisions to move, interact, and complete tasks. Students progress from specification → simulation → verification → (optional) real-world deployment. Emphasis on reproducibility, verifiable evaluation, and safety.

Module 1: The Robotic Nervous System (ROS 2)
- Focus: Middleware for robot control.
- Learning outcomes:
  - Understand ROS 2 architecture: nodes, topics, services, actions.
  - Implement Python agents and controllers using rclpy.
  - Create URDF/SDF descriptions for humanoid bodies and sensors.
  - Design joint-level controllers, controllers stacks (position/velocity/effort), and actuator interfaces.
  - Integrate perception sensors (cameras, LiDAR, IMUs) and fuse sensor data for control loops.
- Deliverables:
  - Functional ROS 2 workspace with example humanoid control graph.
  - Unit-tested controller nodes and launch files.

Module 2: The Digital Twin (Gazebo & Unity)
- Focus: Physics simulation & environment creation.
- Learning outcomes:
  - Build simulated environments in Gazebo and Unity with accurate physics (gravity, torque, collisions).
  - Create digital twin URDF/SDF models and validate kinematics/dynamics.
  - Simulate sensors: LiDAR, depth/RGB cameras, IMUs, force-torque sensors.
  - Run closed-loop tests for locomotion, balance, and manipulation in simulation.
  - Generate deterministic experiment logs for reproducibility.
- Deliverables:
  - Digital twin environment with evaluation scenarios and reproducible test scripts.

Module 3: The AI-Robot Brain (NVIDIA Isaac)
- Focus: Advanced perception & intelligent behavior.
- Learning outcomes:
  - Use Isaac Sim to produce photorealistic simulation and synthetic training datasets.
  - Integrate Isaac ROS components for accelerated perception and VSLAM.
  - Implement Nav2-based planning and localization pipelines for humanoid navigation.
  - Train and evaluate object detection and scene-understanding models with synthetic and real data.
- Deliverables:
  - Perception stack with documented training data pipeline and reproducible model evaluation.

Module 4: Vision–Language–Action (VLA) Systems
- Focus: LLM + Robotics integration and multimodal grounding.
- Learning outcomes:
  - Translate natural language tasks to executable action sequences using LLM-based planners.
  - Implement voice input pipelines (e.g., Whisper) and grounding into ROS actions.
  - Fuse vision and language for object grounding, affordance detection, and task planning.
  - Formalize safety constraints, timeouts, and recovery behaviors for autonomous operation.
- Deliverables:
  - LLM-to-ROS planner node with example instruction-to-action workflows and safety tests.

Capstone Project: The Autonomous Humanoid
- Objective:
  - Build an autonomous humanoid (simulated or physical) that:
    1. Receives a voice command.
    2. Uses LLM reasoning to generate a multi-step plan.
    3. Navigates around obstacles using Nav2 and VSLAM.
    4. Identifies and localizes the target object via computer vision.
    5. Grasps or manipulates the object and completes the task.
- Evaluation criteria:
  - Functional correctness (task completion rate across scenarios).
  - Reproducibility (scripts, dataset seeds, and environment versions provided).
  - Safety and robustness (handles perception failures and replanning).
  - Documentation and traceability (all external models, datasets, and versions cited).
- Deliverables:
  - Full codebase, simulation scenes, model checkpoints, experiment logs, and a final technical report.

Assessment & Rubric
- Modules assessed via:
  - Unit tests for nodes and controllers.
  - Reproducible simulation experiments with provided seeds and environment versions.
  - Written reports documenting methodology, data provenance, and citations.
- Capstone assessed on:
  - System integration, reproducibility, evaluation rigor, and documentation quality.
  - Peer review & demonstration (recorded runs and reproducible test harness).

Safety, Ethics, and Reproducibility
- Safety:
  - Enumerate operational safety limits for any real-robot deployment.
  - Provide simulated fail-safe scenarios and recovery strategies.
- Ethics:
  - Address privacy, data provenance, and human-robot interaction risks in reports.
- Reproducibility:
  - Publish exact dependency manifests, dataset sources, random seeds, simulation versions, and hardware specs.
  - Share synthetic data generation scripts and dataset curation notes.

Appendices (Recommended)
- Appendix A: Example ROS 2 architecture diagram and launch files.
- Appendix B: URDF/SDF template for a humanoid with sensor mounts.
- Appendix C: Dataset generation and labeling pipeline for synthetic data.
- Appendix D: Template for experiment logs and evaluation scripts.

Notes
- This constitution is designed to be adapted to institutional constraints (hardware availability, semester length). Replace real-robot deployment with fully simulated capstones where hardware is unavailable.
- Emphasize verifiability: every claim or evaluated result in student reports should include data/provenance and reproducible steps.

## Response snapshot

The Physical AI & Humanoid Robotics Constitution has been updated to version 1.1.0. This update incorporates detailed project information, including core principles, theme, goals, a quarter overview with modules, a capstone project, assessment criteria, and sections on safety, ethics, and reproducibility.

The following templates and command files require review and potential updates to align with the new constitution's principles:
- `.specify/templates/plan-template.md`
- `.specify/templates/spec-template.md`
- `.specify/templates/tasks-template.md`
- `.specify/templates/commands/sp.adr.md`
- `.specify/templates/commands/sp.analyze.md`
- `.specify/templates/commands/sp.checklist.md`
- `.specify/templates/commands/sp.clarify.md`
- `.specify/templates/commands/sp.git.commit_pr.md`
- `.specify/templates/commands/sp.implement.md`
- `.specify/templates/commands/sp.phr.md`
- `.specify/templates/commands/sp.plan.md`
- `.specify/templates/commands/sp.specify.md`
- `.specify/templates/commands/sp.tasks.md`

A `TODO` remains for `RATIFICATION_DATE`, which needs to be provided by the project owner.

Suggested commit message:
`docs: amend constitution to v1.1.0 (Physical AI & Humanoid Robotics project details)`

## Outcome

- ✅ Impact: The project constitution has been significantly updated with specific content for the "Physical AI & Humanoid Robotics" project, including new core principles, theme, goals, and detailed module breakdowns. This is a minor version bump from 1.0.0 to 1.1.0 due to extensive content changes.
- 🧪 Tests: No tests were run for this action.
- 📁 Files: The file `.specify/memory/constitution.md` was modified.
- 🔁 Next prompts: The user should be prompted to acknowledge the update and potentially review the templates for consistency.
- 🧠 Reflection: The `create-phr.sh` script was not found, so the PHR was created manually. This process needs to be robust for future PHR creations.

## Evaluation notes (flywheel)

- Failure modes observed: The `.specify/scripts/bash/create-phr.sh` script was not found, resulting in a manual PHR creation.
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Investigate why `create-phr.sh` is not found and ensure the PHR creation process is automated and robust.
