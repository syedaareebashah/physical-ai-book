import React from 'react';
import Layout from '@theme/Layout';
import CyberpunkRobot from '@site/src/components/CyberpunkRobot/CyberpunkRobot';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';

function CyberpunkDemo() {
  const {siteConfig} = useDocusaurusContext();

  return (
    <Layout
      title={`Cyberpunk Demo`}
      description="Interactive Cyberpunk 3D Robot Demo for Physical AI & Humanoid Robotics">
      <main className="container margin-vert--lg">
        <div className="row">
          <div className="col col--12">
            <div className="text--center padding-vert--md">
              <h1 className="cyber-text cyber-text-animated">Cyberpunk Robot Showcase</h1>
              <p className="hero__subtitle">Interactive 3D Robot Effects for Physical AI</p>
            </div>

            <div className="margin-vert--lg">
              <h2 className="cyber-text">Different Robot Sizes</h2>
              <div className="row" style={{ justifyContent: 'center', gap: '2rem', margin: '2rem 0' }}>
                <div className="col col--2">
                  <CyberpunkRobot size="small" animation={true} color="primary" />
                  <p className="text--center">Small Robot</p>
                </div>
                <div className="col col--2">
                  <CyberpunkRobot size="medium" animation={true} color="primary" />
                  <p className="text--center">Medium Robot</p>
                </div>
                <div className="col col--2">
                  <CyberpunkRobot size="large" animation={true} color="primary" />
                  <p className="text--center">Large Robot</p>
                </div>
              </div>
            </div>

            <div className="margin-vert--lg">
              <h2 className="cyber-text">Different Robot Colors</h2>
              <div className="row" style={{ justifyContent: 'center', gap: '2rem', margin: '2rem 0' }}>
                <div className="col col--2">
                  <CyberpunkRobot size="medium" animation={true} color="primary" />
                  <p className="text--center">Primary Color</p>
                </div>
                <div className="col col--2">
                  <CyberpunkRobot size="medium" animation={true} color="secondary" />
                  <p className="text--center">Secondary Color</p>
                </div>
                <div className="col col--2">
                  <CyberpunkRobot size="medium" animation={true} color="accent" />
                  <p className="text--center">Accent Color</p>
                </div>
              </div>
            </div>

            <div className="margin-vert--lg">
              <h2 className="cyber-text">Interactive Area</h2>
              <div
                className="cyber-grid-container"
                style={{
                  minHeight: '400px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  background: 'var(--cyberpunk-dark-bg)',
                  border: 'var(--cyberpunk-border)',
                  borderRadius: '8px',
                  margin: '2rem 0',
                  position: 'relative',
                  overflow: 'hidden'
                }}
              >
                <div style={{ position: 'relative', zIndex: 2 }}>
                  <CyberpunkRobot size="large" animation={true} color="primary" />
                </div>

                {/* Floating cyberpunk elements */}
                {[...Array(10)].map((_, i) => (
                  <div
                    key={i}
                    className="floating-element"
                    style={{
                      position: 'absolute',
                      top: `${Math.random() * 100}%`,
                      left: `${Math.random() * 100}%`,
                      animationDelay: `${Math.random() * 5}s`,
                      animationDuration: `${10 + Math.random() * 20}s`
                    }}
                  ></div>
                ))}
              </div>
            </div>

            <div className="text--center padding-vert--lg">
              <Link
                className="button button--primary button--lg cyberpunk-button"
                to="/chat">
                Try AI Assistant
              </Link>

              <Link
                className="button button--secondary button--lg cyberpunk-button"
                to="/">
                Back to Home
              </Link>
            </div>
          </div>
        </div>
      </main>
    </Layout>
  );
}

export default CyberpunkDemo;