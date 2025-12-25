import React, { JSX } from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import LearningPath from '@site/src/components/LearningPath';
import ComparisonTable from '@site/src/components/ComparisonTable';

import Heading from '@theme/Heading';
import { Brain, Cpu, Workflow, Zap, Code, Globe } from 'lucide-react';
import ChatKit from '@site/src/components/ChatKit';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className="hero">
      <div className="container">
        <div className="row">
          <div className="col col--6">
            <div className="hero__content">
              <Heading as="h1" className="hero__title font-display text-uppercase">
                {siteConfig.title}
              </Heading>
              <p className="hero__subtitle">{siteConfig.tagline}</p>
              <p className="hero__subtitle">
                AI Systems in the Physical World | Embodied Intelligence | Bridging Digital Brains with Physical Bodies
              </p>
              <div className="button-group" style={{marginTop: '2rem'}}>
                <Link
                  className="button button--secondary button--lg"
                  to="/docs/intro">
                  Start Learning Physical AI
                </Link>
                <Link
                  className="button button--primary button--lg"
                  to="/docs/module1-ros2/introduction">
                  Begin with ROS 2
                </Link>
              </div>
            </div>
          </div>
          <div className="col col--6">
            <div className="hero__image-container">
              <img
                src={require('@site/static/img/hero2.png').default}
                alt="Humanoid Robot"
                className="hero__image"
                style={{width: '100%', height: 'auto'}}
              />
            </div>
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
          {/* Editorial Feature Cards Section */}
          <section className="container padding-vert--lg">
            <div className="row">
              <div className="col col--4">
                <div className="card hover-lift">
                  <div className="card__header">
                    <h3 className="font-display text-uppercase">Robotic Nervous System</h3>
                  </div>
                  <div className="card__body">
                    <p>Learn ROS 2 fundamentals, nodes, topics, and services for controlling humanoid robots with rclpy.</p>
                  </div>
                </div>
              </div>
              <div className="col col--4">
                <div className="card hover-lift">
                  <div className="card__header">
                    <h3 className="font-display text-uppercase">Digital Twin</h3>
                  </div>
                  <div className="card__body">
                    <p>Master physics simulation in Gazebo and high-fidelity rendering in Unity for realistic environments.</p>
                  </div>
                </div>
              </div>
              <div className="col col--4">
                <div className="card hover-lift">
                  <div className="card__header">
                    <h3 className="font-display text-uppercase">AI-Robot Brain</h3>
                  </div>
                  <div className="card__body">
                    <p>Explore NVIDIA Isaac for advanced perception, VSLAM, and accelerated navigation systems.</p>
                  </div>
                </div>
              </div>
            </div>
            <div className="row" style={{marginTop: '2rem'}}>
              <div className="col col--4 col--offset-2">
                <div className="card hover-lift">
                  <div className="card__header">
                    <h3 className="font-display text-uppercase">Vision-Language-Action</h3>
                  </div>
                  <div className="card__body">
                    <p>Converge LLMs with robotics: Voice-to-Action using OpenAI Whisper and multimodal AI.</p>
                  </div>
                </div>
              </div>
              <div className="col col--4">
                <div className="card hover-lift">
                  <div className="card__header">
                    <h3 className="font-display text-uppercase">Physical AI Focus</h3>
                  </div>
                  <div className="card__body">
                    <p>Bridging the gap between digital intelligence and physical reality with embodied systems.</p>
                  </div>
                </div>
              </div>
            </div>
          </section>

          <HomepageFeatures />
          <LearningPath />
          <ComparisonTable />

          {/* Editorial Highlight Box for AI Assistant */}
          <section className="container padding-vert--lg">
            <div className="editorial-highlight-box">
              <div className="row">
                <div className="col col--12">
                  <Heading as="h2" className="text-center font-display text-uppercase">Ask Our AI Assistant</Heading>
                  <p className="text-center">
                    Have questions about the content? Use our AI assistant powered by the book content to get answers.
                  </p>
                  <div className="text--center" style={{maxWidth: '900px', margin: '2rem auto', height: '500px', border: '1px solid var(--editorial-border)', borderRadius: '8px', overflow: 'hidden'}}>
                    <ChatKit endpoint="http://localhost:8000/api" sessionId="default-session" />
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* Editorial Call-to-Action */}
          <section className="container padding-vert--lg">
            <div className="text--center padding-vert--xl bg-card" style={{borderRadius: '8px', padding: '3rem 2rem'}}>
              <Heading as="h2" className="font-display accent-red">Ready to Build Intelligent Robots?</Heading>
              <p style={{fontSize: '1.2rem', maxWidth: '600px', margin: '1.5rem auto', color: 'var(--editorial-text-secondary)'}}>
                Join the revolution where AI meets the physical world. Start your journey in Physical AI today.
              </p>
              <div className="button-group" style={{marginTop: '1.5rem'}}>
                <Link
                  className="button button--primary button--lg"
                  to="/docs/intro">
                  Begin Your Physical AI Journey
                </Link>
              </div>
            </div>
          </section>
      </main>
    </Layout>
  );
}
