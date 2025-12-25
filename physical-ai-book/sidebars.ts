import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Introduction',
      items: [
        'intro',
        'prerequisites',
        'setup',
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Module 1: The Robotic Nervous System (ROS 2)',
      items: [
        'module1-ros2/introduction',
        'module1-ros2/core-concepts',
        'module1-ros2/first-node',
        'module1-ros2/python-rclpy',
        'module1-ros2/urdf',
        'module1-ros2/project',
        'module1-ros2/assessment',
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Module 2: The Digital Twin (Gazebo & Unity)',
      items: [
        'module2-simulation/introduction',
        'module2-simulation/gazebo-fundamentals',
        'module2-simulation/simulating-sensors',
        'module2-simulation/unity-rendering',
        'module2-simulation/building-environments',
        'module2-simulation/project',
        'module2-simulation/assessment',
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Module 3: The AI-Robot Brain (NVIDIA Isaac)',
      items: [
        'module3-isaac/introduction',
        'module3-isaac/isaac-sim',
        'module3-isaac/visual-slam',
        'module3-isaac/navigation-stack',
        'module3-isaac/perception-pipeline',
        'module3-isaac/project',
        'module3-isaac/assessment',
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action (VLA)',
      items: [
        'module4-vla/llms-robotics',
        'module4-vla/voice-action-pipeline',
        'module4-vla/cognitive-planning',
        'module4-vla/multimodal-integration',
        'module4-vla/capstone',
        'module4-vla/advanced-topics',
        'module4-vla/final-assessment',
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Appendices',
      items: [
        'appendices/installation-guides',
        'appendices/ros2-command-reference',
        'appendices/python-libraries-reference',
        'appendices/resources-links',
        'appendices/glossary',
      ],
      collapsed: true,
    },
  ],
};

export default sidebars;
