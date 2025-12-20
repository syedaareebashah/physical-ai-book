import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import { useAuth } from '../../services/user-context';
import PersonalizeButton from '../../components/chapters/PersonalizeButton';
import TranslateButton from '../../components/chapters/TranslateButton';

interface Chapter {
  id: string;
  title: string;
  content: string;
  urduContent?: string;
}

const ChapterPage: React.FC = () => {
  const router = useRouter();
  const { id } = router.query;
  const { isAuthenticated } = useAuth();
  const [chapter, setChapter] = useState<Chapter | null>(null);
  const [displayContent, setDisplayContent] = useState<string>('');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchChapter = async () => {
      if (!id) return;

      try {
        setIsLoading(true);
        // In a real implementation, we would fetch from an API
        // For now, we'll simulate with sample data
        const mockChapter: Chapter = {
          id: id as string,
          title: 'Introduction to JavaScript',
          content: `JavaScript is a versatile programming language that is primarily used for web development. It allows you to create dynamic and interactive web content.

# Basic Concepts
- Variables: Use 'let', 'const', or 'var' to declare variables
- Functions: Reusable blocks of code that perform specific tasks
- Objects: Collections of key-value pairs

# Example Code
Here's a simple function that greets a user:

function greetUser(name) {
  return "Hello, " + name + "!";
}

console.log(greetUser("Alice")); // Output: Hello, Alice!`,
        };

        setChapter(mockChapter);
        setDisplayContent(mockChapter.content);
      } catch (err) {
        setError('Failed to load chapter');
        console.error(err);
      } finally {
        setIsLoading(false);
      }
    };

    fetchChapter();
  }, [id]);

  const handlePersonalizedContent = (personalizedContent: string) => {
    setDisplayContent(personalizedContent);
  };

  const handleTranslatedContent = (translatedContent: string) => {
    setDisplayContent(translatedContent);
  };

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <p>Loading chapter...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <p className="text-red-500">{error}</p>
      </div>
    );
  }

  if (!chapter) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <p>Chapter not found</p>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-3xl font-bold text-gray-900 mb-6">{chapter.title}</h1>

      <div className="bg-white shadow overflow-hidden sm:rounded-lg">
        <div className="px-4 py-5 sm:px-6">
          <div className={displayContent.includes('یہ متن اردو میں ترجمہ شدہ ہے') ? 'urdu-text' : 'prose max-w-none'}>
            <pre className={displayContent.includes('یہ متن اردو میں ترجمہ شدہ ہے') ? 'whitespace-pre-wrap text-gray-700 text-right' : 'whitespace-pre-wrap text-gray-700'}>
              {displayContent}
            </pre>
          </div>

          <div className="mt-6 flex flex-wrap gap-3">
            {isAuthenticated && (
              <>
                <PersonalizeButton
                  chapterId={chapter.id}
                  content={chapter.content}
                  onPersonalizedContent={handlePersonalizedContent}
                />
                <TranslateButton
                  chapterId={chapter.id}
                  content={chapter.content}
                  onTranslatedContent={handleTranslatedContent}
                />
              </>
            )}

            {!isAuthenticated && (
              <p className="text-gray-600">
                Log in to access personalization and translation features
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChapterPage;