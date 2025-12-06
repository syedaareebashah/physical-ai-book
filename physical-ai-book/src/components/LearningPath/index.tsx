import React from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

type ModuleItem = {
  id: string;
  title: string;
  description: string;
  duration: string;
  level: string;
  completed?: boolean;
};

const ModuleList: ModuleItem[] = [
  {
    id: 'module1',
    title: 'Module 1: The Robotic Nervous System (ROS 2)',
    description: 'Learn core ROS 2 concepts, create your first nodes, and integrate with Python using rclpy',
    duration: '3-4 weeks',
    level: 'Beginner',
    completed: false,
  },
  {
    id: 'module2',
    title: 'Module 2: The Digital Twin (Gazebo & Unity)',
    description: 'Master robot simulation with Gazebo and Unity, creating digital twins for testing',
    duration: '3-4 weeks',
    level: 'Intermediate',
    completed: false,
  },
  {
    id: 'module3',
    title: 'Module 3: The AI-Robot Brain (NVIDIA Isaac)',
    description: 'Build intelligent robot brains using NVIDIA Isaac for perception and navigation',
    duration: '3-4 weeks',
    level: 'Advanced',
    completed: false,
  },
  {
    id: 'module4',
    title: 'Module 4: Vision-Language-Action (VLA)',
    description: 'Combine LLMs with robotics for cognitive planning and multimodal integration',
    duration: '3-4 weeks',
    level: 'Advanced',
    completed: false,
  },
];

function ModuleCard({ id, title, description, duration, level, completed }: ModuleItem) {
  return (
    <div className={clsx('card', styles.moduleCard, completed ? styles.completed : '')}>
      <div className="card__header">
        <h3>{title}</h3>
        <span className={clsx('badge badge--secondary', styles.levelBadge, `badge--${level.toLowerCase()}`)}>
          {level}
        </span>
      </div>
      <div className="card__body">
        <p>{description}</p>
      </div>
      <div className="card__footer">
        <div className={styles.moduleFooter}>
          <span className="badge badge--info">{duration}</span>
          <button className="button button--primary button--sm">
            {completed ? 'Review' : 'Start Learning'}
          </button>
        </div>
      </div>
    </div>
  );
}

export default function LearningPath(): JSX.Element {
  return (
    <section className={styles.learningPath}>
      <div className="container padding-vert--xl">
        <div className="row">
          <div className="col col--12">
            <h2 className={styles.sectionTitle}>Your Learning Path</h2>
            <p className={styles.sectionSubtitle}>
              Progress through our structured curriculum designed to take you from beginner to expert in Physical AI and Humanoid Robotics
            </p>

            <div className={styles.modulesGrid}>
              {ModuleList.map((module) => (
                <div key={module.id} className="col col--12 col--md-6">
                  <ModuleCard {...module} />
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}