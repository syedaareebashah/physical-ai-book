import React, { useState, useEffect } from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';
import ChatKit from '@site/src/components/ChatKit';

// Function to create and manage the chatbot widget
const ChatbotWidget = () => {
  const [isVisible, setIsVisible] = useState(true);
  const [isMinimized, setIsMinimized] = useState(false);
  const [hasUnreadMessage, setHasUnreadMessage] = useState(false);

  // Determine API endpoint based on environment
  const apiEndpoint = process.env.NODE_ENV === 'production'
    ? 'https://areebashah-rag-chatbot.hf.space'  // Your deployed backend URL
    : 'http://localhost:8000'; // Local development backend URL

  // Track if there are new messages to show unread indicator
  useEffect(() => {
    const handleNewMessage = () => {
      if (!isMinimized) {
        setHasUnreadMessage(false);
      }
    };

    // In a real implementation, you would listen to ChatKit's message events
    // For now, we'll just set unread to false when widget is opened
    if (!isMinimized) {
      setHasUnreadMessage(false);
    }
  }, [isMinimized]);

  if (!isVisible) {
    return (
      <button
        id="chatbot-open-btn"
        onClick={() => {
          setIsVisible(true);
          setIsMinimized(false);
        }}
        style={{
          position: 'fixed',
          bottom: '20px',
          right: '20px',
          zIndex: '10000',
          width: '65px',
          height: '65px',
          borderRadius: '50%',
          background: 'linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%)',
          color: 'white',
          border: 'none',
          fontSize: '24px',
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          boxShadow: '0 8px 20px rgba(79, 70, 229, 0.4)',
          transition: 'all 0.3s ease',
        }}
        onMouseEnter={(e) => {
          e.target.style.transform = 'scale(1.05)';
          e.target.style.boxShadow = '0 10px 25px rgba(79, 70, 229, 0.5)';
        }}
        onMouseLeave={(e) => {
          e.target.style.transform = 'scale(1)';
          e.target.style.boxShadow = '0 8px 20px rgba(79, 70, 229, 0.4)';
        }}
      >
        🤖
      </button>
    );
  }

  return (
    <div
      id="chatkit-widget"
      style={{
        position: 'fixed',
        bottom: '20px',
        right: '20px',
        width: isMinimized ? '250px' : '350px',
        height: isMinimized ? '60px' : '500px',
        border: '1px solid var(--ifm-color-emphasis-300)',
        borderRadius: '8px',
        boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
        backgroundColor: 'var(--ifm-background-color)',
        zIndex: '10000',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
      }}
    >
      {isMinimized ? (
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            padding: '12px 16px',
            background: 'linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%)',
            color: 'white',
            cursor: 'pointer',
            borderRadius: '8px 8px 0 0',
          }}
          onClick={() => setIsMinimized(false)}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{fontSize: '18px'}}>🤖</span>
            <span>AI Assistant</span>
            {hasUnreadMessage && (
              <span
                style={{
                  width: '10px',
                  height: '10px',
                  borderRadius: '50%',
                  backgroundColor: '#fbbf24',
                  marginLeft: '8px',
                }}
              />
            )}
          </div>
          <button
            onClick={(e) => {
              e.stopPropagation();
              setIsVisible(false);
            }}
            style={{
              background: 'rgba(255, 255, 255, 0.2)',
              border: 'none',
              color: 'white',
              fontSize: '16px',
              cursor: 'pointer',
              padding: '4px',
              width: '24px',
              height: '24px',
              borderRadius: '50%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            ×
          </button>
        </div>
      ) : (
        <>
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              padding: '14px 16px',
              background: 'linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%)',
              color: 'white',
              borderRadius: '8px 8px 0 0',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{fontSize: '20px'}}>🤖</span>
              <div>
                <div style={{fontWeight: '600', fontSize: '1rem'}}>AI Assistant</div>
                <div style={{fontSize: '0.8rem', opacity: 0.85}}>Ask me about Physical AI</div>
              </div>
              {hasUnreadMessage && (
                <span
                  style={{
                    width: '10px',
                    height: '10px',
                    borderRadius: '50%',
                    backgroundColor: '#fbbf24',
                    marginLeft: '8px',
                  }}
                />
              )}
            </div>
            <div style={{ display: 'flex', gap: '10px' }}>
              <button
                onClick={() => setIsMinimized(true)}
                style={{
                  background: 'rgba(255, 255, 255, 0.2)',
                  border: 'none',
                  color: 'white',
                  fontSize: '16px',
                  cursor: 'pointer',
                  padding: '4px',
                  width: '28px',
                  height: '28px',
                  borderRadius: '50%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                −
              </button>
              <button
                onClick={() => setIsVisible(false)}
                style={{
                  background: 'rgba(255, 255, 255, 0.2)',
                  border: 'none',
                  color: 'white',
                  fontSize: '16px',
                  cursor: 'pointer',
                  padding: '4px',
                  width: '28px',
                  height: '28px',
                  borderRadius: '50%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                ×
              </button>
            </div>
          </div>
          <div
            style={{
              flex: 1,
              overflow: 'hidden',
            }}
          >
            <ChatKit
              endpoint={apiEndpoint}
              sessionId={null}
            />
          </div>
        </>
      )}
    </div>
  );
};

// Docusaurus Root component
function Root({ children }) {
  return (
    <>
      {children}
      <BrowserOnly>
        {() => <ChatbotWidget />}
      </BrowserOnly>
    </>
  );
}

export default Root;