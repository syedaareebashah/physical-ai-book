import React from 'react';

const MessagesList = ({ messages, messagesEndRef }) => {
  return (
    <div className="chatkit-messages">
      {messages.map((message) => (
        <div
          key={message.id}
          className={`chatkit-message chatkit-message-${message.role}`}
        >
          <div className="chatkit-message-content">
            <div className="chatkit-message-text">
              {message.content}
            </div>
            {message.sources && message.sources.length > 0 && (
              <div className="chatkit-sources">
                <details className="chatkit-sources-details">
                  <summary>Sources</summary>
                  <ul>
                    {message.sources.map((source, index) => (
                      <li key={index} className="chatkit-source-item">
                        <strong>{source.filename}</strong>
                        {source.page && <span>, Page {source.page}</span>}
                        {source.content && (
                          <p className="chatkit-source-excerpt">
                            "{source.content.substring(0, 100)}..."
                          </p>
                        )}
                      </li>
                    ))}
                  </ul>
                </details>
              </div>
            )}
            {/* Display SpeckItPlus specification information if available */}
            {message.spec_info && (
              <div className="chatkit-spec-info">
                <div className="chatkit-spec-compliance">
                  Spec Compliance: {message.spec_info.spec_compliance || 'unknown'}
                </div>
                {message.spec_info.confidence !== undefined && (
                  <div className="chatkit-spec-confidence">
                    Confidence: {(message.spec_info.confidence * 100).toFixed(1)}%
                  </div>
                )}
                {message.spec_info.analysis && (
                  <div className="chatkit-spec-analysis">
                    Analysis: {message.spec_info.analysis}
                  </div>
                )}
                {message.spec_info.spec_matches && message.spec_info.spec_matches.length > 0 && (
                  <div className="chatkit-spec-matches">
                    Matches: {message.spec_info.spec_matches.join(', ')}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      ))}
      <div ref={messagesEndRef} />
    </div>
  );
};

export default MessagesList;