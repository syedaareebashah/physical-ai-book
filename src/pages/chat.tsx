import React from 'react';
import Layout from '@theme/Layout';
import ChatKit from '@site/src/components/ChatKit';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';

function ChatPage() {
  const {siteConfig} = useDocusaurusContext();

  // Determine API endpoint based on environment
  const apiEndpoint = process.env.NODE_ENV === 'production'
    ? 'https://your-backend-url/api'  // Replace with actual production URL
    : 'http://localhost:8000/api';

  return (
    <Layout
      title={`AI Assistant`}
      description="Interactive AI assistant for Physical AI & Humanoid Robotics">
      <main className="container margin-vert--lg">
        <div className="row">
          <div className="col col--12">
            <div className="text--center padding-vert--md">
              <h1>Physical AI Assistant</h1>
              <p className="hero__subtitle">Ask me anything about Physical AI & Humanoid Robotics</p>
            </div>

            <div style={{
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
              minHeight: '600px',
              maxWidth: '800px',
              margin: '0 auto'
            }}>
              <ChatKit
                endpoint={apiEndpoint}
                sessionId={null} // Will be generated automatically
              />
            </div>
          </div>
        </div>
      </main>
    </Layout>
  );
}

export default ChatPage;