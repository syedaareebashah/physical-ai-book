import React from 'react';
import MessagesList from './MessagesList';
import InputArea from './InputArea';

const ChatWindow = ({
  messages,
  inputValue,
  setInputValue,
  isLoading,
  selectedText,
  clearSelectedText,
  sendMessage,
  handleKeyPress,
  messagesEndRef
}) => {
  return (
    <div className="chatkit-container">
      <div className="chatkit-header">
        <div className="chatkit-header-icon">🤖</div>
        <div>
          <h3>AI Assistant</h3>
          <p>Ask me anything about Physical AI & Humanoid Robotics</p>
        </div>
      </div>

      {/* Display selected text if available */}
      {selectedText && (
        <div className="chatkit-selected-text-container">
          <div className="chatkit-selected-text-header">
            <span className="chatkit-selected-text-label">Selected Text Context</span>
            <button
              className="chatkit-clear-selection-btn"
              onClick={clearSelectedText}
              title="Clear selected text context"
            >
              ×
            </button>
          </div>
          <div className="chatkit-selected-text-content">
            "{selectedText.substring(0, 200)}{selectedText.length > 200 ? '...' : ''}"
          </div>
        </div>
      )}

      <MessagesList
        messages={messages}
        messagesEndRef={messagesEndRef}
      />

      <InputArea
        inputValue={inputValue}
        setInputValue={setInputValue}
        isLoading={isLoading}
        selectedText={selectedText}
        sendMessage={sendMessage}
        handleKeyPress={handleKeyPress}
      />
    </div>
  );
};

export default ChatWindow;