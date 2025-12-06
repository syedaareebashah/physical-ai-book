import React, { JSX } from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import LearningPath from '@site/src/components/LearningPath';
import ComparisonTable from '@site/src/components/ComparisonTable';

import Heading from '@theme/Heading';
import styles from './index.module.css';
import { Brain, Cpu, Workflow, Zap, Code, Globe } from 'lucide-react';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <div className={styles.heroContent}>
          <div className={styles.heroText}>
            <Heading as="h1" className="hero__title">
              {siteConfig.title}
            </Heading>
            <p className="hero__subtitle">{siteConfig.tagline}</p>
            <p className={styles.subtitle}>
              AI Systems in the Physical World | Embodied Intelligence | Bridging Digital Brains with Physical Bodies
            </p>
            <div className={styles.buttons}>
              <Link
                className="button button--secondary button--lg"
                to="/docs/intro">
                Start Learning Physical AI - 5min ⏱️
              </Link>
              <Link
                className="button button--primary button--lg"
                to="/docs/module1-ros2/introduction">
                Begin with ROS 2 - 10min ⏱️
              </Link>
            </div>
          </div>
           <div className={styles.heroImage}>
            <img src="https://images.unsplash.com/photo-1581092580497-c3a42146b952?q=80&w=1887&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D" alt="Humanoid Robot" />
          </div>
        </div>
      </div>
    </header>
  );
}

export default function Home(): JSX.Element {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Physical AI & Humanoid Robotics`}
      description="Comprehensive curriculum for Physical AI - AI systems that function in reality and comprehend physical laws">
      <HomepageHeader />
      <main>
        <section className={styles.featuresSection}>
          <div className="container">
            <div className="row">
              <div className="col col--4">
                <div className={clsx(styles.featureCard, styles.cardHover)}>
                  <div className={styles.featureIcon}>
                    <img src="https://images.unsplash.com/photo-1547127796-0b-6f85e32b62ad?q=80&w=1887&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D" alt="Robotic Nervous System" />
                  </div>
                  <div className={styles.featureCardContent}>
                    <h3>Robotic Nervous System</h3>
                    <p>Learn ROS 2 fundamentals, nodes, topics, and services for controlling humanoid robots with rclpy.</p>
                  </div>
                </div>
              </div>
              <div className="col col--4">
                <div className={clsx(styles.featureCard, styles.cardHover)}>
                  <div className={styles.featureIcon}>
                    <img src="https://images.unsplash.com/photo-1611606063065-ee7946f0b34a?q=80&w=1887&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D" alt="Digital Twin" />
                  </div>
                  <div className={styles.featureCardContent}>
                    <h3>Digital Twin</h3>
                    <p>Master physics simulation in Gazebo and high-fidelity rendering in Unity for realistic environments.</p>
                  </div>
                </div>
              </div>
              <div className="col col--4">
                <div className={clsx(styles.featureCard, styles.cardHover)}>
                  <div className={styles.featureIcon}>
                    <img src="https://images.unsplash.com/photo-1527430222751-9149f7253b73?q=80&w=1887&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D" alt="AI-Robot Brain" />
                  </div>
                  <div className={styles.featureCardContent}>
                    <h3>AI-Robot Brain</h3>
                    <p>Explore NVIDIA Isaac for advanced perception, VSLAM, and accelerated navigation systems.</p>
                  </div>
                </div>
              </div>
            </div>
            <div className="row" style={{marginTop: '2rem'}}>
              <div className="col col--4 col--offset-2">
                <div className={clsx(styles.featureCard, styles.cardHover)}>
                  <div className={styles.featureIcon}>
                    <img src="https://images.unsplash.com/photo-1554224155-169544351720?q=80&w=1740&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D" alt="Vision-Language-Action" />
                  </div>
                  <div className={styles.featureCardContent}>
                    <h3>Vision-Language-Action</h3>
                    <p>Converge LLMs with robotics: Voice-to-Action using OpenAI Whisper and multimodal AI.</p>
                  </div>
                </div>
              </div>
              <div className="col col--4">
                <div className={clsx(styles.featureCard, styles.cardHover)}>
                  <div className={styles.featureIcon}>
                    <img src="https://images.unsplash.com/photo-1571383323975-92337171b39a?q=80&w=1887&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D" alt="Physical AI Focus" />
                  </div>
                  <div className={styles.featureCardContent}>
                    <h3>Physical AI Focus</h3>
                    <p>Bridging the gap between digital intelligence and physical reality with embodied systems.</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>
        <HomepageFeatures />
        <LearningPath />
        <ComparisonTable />
        <section className={styles.ctaSection}>
          <div className="container text--center padding-vert--xl">
            <Heading as="h2">Ready to Build Intelligent Robots?</Heading>
            <p className="padding-horiz--md">
              Join the revolution where AI meets the physical world. Start your journey in Physical AI today.
            </p>
            <Link
              className="button button--primary button--lg"
              to="/docs/intro">
              Begin Your Physical AI Journey
            </Link>
          </div>
        </section>
      </main>
    </Layout>
  );
}
