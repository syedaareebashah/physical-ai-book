import React, { useState } from 'react';
import { useAuth } from '../../services/user-context';

interface TranslateButtonProps {
  chapterId: string;
  content: string;
  onTranslatedContent: (translatedContent: string) => void;
}

const TranslateButton: React.FC<TranslateButtonProps> = ({
  chapterId,
  content,
  onTranslatedContent
}) => {
  const { user, token, isAuthenticated } = useAuth();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleTranslate = async () => {
    if (!isAuthenticated) {
      setError('You must be logged in to translate content');
      return;
    }

    if (!token) {
      setError('Authentication token missing');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/translation/urdu', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
        body: JSON.stringify({
          chapterId,
          content,
          preserveFormatting: true,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Failed to translate content');
      }

      const data = await response.json();
      onTranslatedContent(data.urduContent);
    } catch (err) {
      console.error('Translation error:', err);
      setError(err instanceof Error ? err.message : 'An error occurred during translation');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mt-4">
      <button
        onClick={handleTranslate}
        disabled={loading}
        className="bg-green-600 hover:bg-green-700 text-white font-semibold py-2 px-4 rounded-md disabled:opacity-50 transition-colors focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2"
      >
        {loading ? (
          <span className="flex items-center">
            <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            Translating...
          </span>
        ) : (
          'Translate to Urdu'
        )}
      </button>

      {error && (
        <div className="mt-2 p-2 bg-red-100 text-red-700 rounded-md text-sm">
          {error}
        </div>
      )}
    </div>
  );
};

export default TranslateButton;