import React from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  Svg?: React.ComponentType<React.ComponentProps<'svg'>>;
  description: React.ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Physical Intelligence',
    description: (
      <>
        Learn how to bridge the gap between digital intelligence and the physical world through embodied AI systems.
      </>
    ),
  },
  {
    title: 'ROS 2 Integration',
    description: (
      <>
        Master the Robot Operating System 2 for creating distributed robotic applications with Python and rclpy.
      </>
    ),
  },
  {
    title: 'Simulation First',
    description: (
      <>
        Develop and test robotic systems in Gazebo and Unity before deploying to real hardware with NVIDIA Isaac.
      </>
    ),
  },
];

function Feature({title, description}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): React.JSX.Element {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
