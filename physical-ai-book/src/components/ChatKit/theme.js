// Theme configuration for ChatKit component
// Matches the ai-native.panaversity.org aesthetic

const chatKitTheme = {
  // Colors
  primaryColor: '#2563eb', // blue-600
  primaryColorDark: '#1d4ed8', // blue-700
  backgroundColor: 'var(--ifm-background-color)',
  textColor: 'var(--ifm-font-color-base)',
  userMessageBg: 'var(--ifm-color-primary)',
  userMessageText: 'white',
  assistantMessageBg: 'var(--ifm-color-emphasis-100)',
  assistantMessageText: 'var(--ifm-font-color-base)',

  // Typography
  fontFamily: 'var(--ifm-font-family-base)',
  fontSize: '1rem',
  lineHeight: '1.5',

  // Spacing
  padding: '1rem',
  messageSpacing: '1rem',
  borderRadius: '18px',
  inputHeight: '44px',

  // Shadows
  boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
  messageShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',

  // Animation
  fadeInDuration: '0.3s',
  easeOut: 'ease-out',

  // Responsive
  mobileHeight: '400px',
  mobileMaxWidth: '90%',

  // Accessibility
  focusOutline: '0 0 0 2px rgba(0, 123, 255, 0.25)',

  // Dark mode support
  dark: {
    backgroundColor: 'var(--ifm-background-color)',
    assistantMessageBg: 'var(--ifm-color-emphasis-200)',
    textColor: 'var(--ifm-font-color-base)',
  }
};

export default chatKitTheme;