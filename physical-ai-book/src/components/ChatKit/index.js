import React, { useState, useEffect, useRef } from 'react';
import ChatWindow from './ChatWindow';
import './ChatKit.css';

const ChatKit = ({ endpoint, sessionId: propSessionId }) => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState(propSessionId || null);
  const [selectedText, setSelectedText] = useState(''); // Track selected text from the book
  const messagesEndRef = useRef(null);

  // Scroll to bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Load conversation history if sessionId exists
  useEffect(() => {
    if (sessionId) {
      loadConversationHistory();
    }
  }, [sessionId]);

  // Listen to text selection globally
  useEffect(() => {
    const handleSelectionChange = () => {
      const selection = window.getSelection();
      const text = selection.toString().trim();

      if (text && text.length > 10) { // Only set if meaningful text is selected
        setSelectedText(text);
      }
    };

    document.addEventListener('selectionchange', handleSelectionChange);
    return () => {
      document.removeEventListener('selectionchange', handleSelectionChange);
    };
  }, []);

  const loadConversationHistory = async () => {
    try {
      const response = await fetch(`${endpoint}/chat/history/${sessionId}`);
      if (response.ok) {
        const data = await response.json();
        setMessages(data.messages.map(msg => ({
          id: msg.id,
          role: msg.role,
          content: msg.content,
          timestamp: new Date(msg.timestamp),
          sources: msg.sources
        })));
      }
    } catch (error) {
      console.error('Error loading conversation history:', error);
    }
  };

  const sendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: inputValue,
      timestamp: new Date()
    };

    // Add user message to UI immediately
    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    setInputValue('');
    setIsLoading(true);

    try {
      // Use the selected text endpoint if there's selected text
      let response;
      if (selectedText) {
        // Send selected_text as a query parameter to the endpoint
        const url = new URL(`${endpoint}/chat/with-selected-text`);
        url.searchParams.append('selected_text', selectedText);

        response = await fetch(url, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            message: inputValue,
            session_id: sessionId
          })
        });
      } else {
        response = await fetch(`${endpoint}/chat`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            message: inputValue,
            session_id: sessionId
          }),
        });
      }

      if (response.ok) {
        const data = await response.json();

        // Update session ID if it's new
        if (!sessionId) {
          setSessionId(data.session_id);
        }

        const assistantMessage = {
          id: `assistant-${Date.now()}`,
          role: 'assistant',
          content: data.message,
          timestamp: new Date(),
          sources: data.sources
        };

        setMessages(prev => [...prev, assistantMessage]);
      } else {
        throw new Error('Failed to get response');
      }
    } catch (error) {
      console.error('Error sending message:', error);

      // Add error message to UI
      const errorMessage = {
        id: `error-${Date.now()}`,
        role: 'assistant',
        content: 'Sorry, I encountered an error processing your request. Please try again.',
        timestamp: new Date(),
        sources: []
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  // Function to clear the selected text
  const clearSelectedText = () => {
    setSelectedText('');
  };

  return (
    <ChatWindow
      messages={messages}
      inputValue={inputValue}
      setInputValue={setInputValue}
      isLoading={isLoading}
      selectedText={selectedText}
      clearSelectedText={clearSelectedText}
      sendMessage={sendMessage}
      handleKeyPress={handleKeyPress}
      messagesEndRef={messagesEndRef}
    />
  );
};

export default ChatKit;