import React from 'react';

const InputArea = ({
  inputValue,
  setInputValue,
  isLoading,
  selectedText,
  sendMessage,
  handleKeyPress
}) => {
  return (
    <div className="chatkit-input-area">
      <div className="chatkit-input-container">
        <textarea
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder={selectedText
            ? "Ask a question about the selected text..."
            : "Ask a question about Physical AI & Humanoid Robotics..."}
          className="chatkit-input"
          rows="1"
          disabled={isLoading}
        />
        <button
          onClick={sendMessage}
          disabled={isLoading || !inputValue.trim()}
          className="chatkit-send-button"
        >
          {isLoading ? (
            <span className="chatkit-loading-spinner">●●●</span>
          ) : (
            <>
              <span>Send</span>
              <span>→</span>
            </>
          )}
        </button>
      </div>
      <div className="chatkit-input-hint">
        {selectedText
          ? "Press Enter to send (using selected text as context)"
          : "Press Enter to send, Shift+Enter for new line"}
      </div>
    </div>
  );
};

export default InputArea;