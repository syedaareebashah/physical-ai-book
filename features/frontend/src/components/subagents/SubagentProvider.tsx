import React, { createContext, useContext, useState, useEffect } from 'react';
import { useAuth } from '../../services/user-context';

interface Subagent {
  id: string;
  name: string;
  type: 'personalization' | 'translation' | 'content-summary' | 'other';
  description?: string;
}

interface SubagentContextType {
  subagents: Subagent[];
  loading: boolean;
  error: string | null;
  executeSubagent: (subagentId: string, input: any) => Promise<any>;
  refreshSubagents: () => Promise<void>;
}

const SubagentContext = createContext<SubagentContextType | undefined>(undefined);

export const useSubagents = () => {
  const context = useContext(SubagentContext);
  if (!context) {
    throw new Error('useSubagents must be used within a SubagentProvider');
  }
  return context;
};

interface SubagentProviderProps {
  children: React.ReactNode;
}

export const SubagentProvider: React.FC<SubagentProviderProps> = ({ children }) => {
  const { token } = useAuth();
  const [subagents, setSubagents] = useState<Subagent[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchSubagents = async () => {
    if (!token) return;

    try {
      setLoading(true);
      setError(null);

      // In a real implementation, we would fetch from an API
      // For now, we'll return mock subagents
      const mockSubagents: Subagent[] = [
        {
          id: 'personalization-agent',
          name: 'Personalization Agent',
          type: 'personalization',
          description: 'Adapts content based on user background'
        },
        {
          id: 'translation-agent',
          name: 'Translation Agent',
          type: 'translation',
          description: 'Translates content to Urdu'
        },
        {
          id: 'summary-agent',
          name: 'Summary Agent',
          type: 'content-summary',
          description: 'Generates content summaries'
        }
      ];

      setSubagents(mockSubagents);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch subagents');
      console.error('Subagent fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (token) {
      fetchSubagents();
    } else {
      setSubagents([]);
      setLoading(false);
    }
  }, [token]);

  const executeSubagent = async (subagentId: string, input: any) => {
    if (!token) {
      throw new Error('Authentication required');
    }

    try {
      // In a real implementation, we would call the subagent API
      // For now, we'll simulate the call
      return new Promise((resolve) => {
        setTimeout(() => {
          resolve({
            result: `Result from ${subagentId}`,
            metadata: { subagentId, processingTime: 100 },
          });
        }, 100);
      });
    } catch (err) {
      console.error('Subagent execution error:', err);
      throw err;
    }
  };

  const refreshSubagents = async () => {
    await fetchSubagents();
  };

  const value = {
    subagents,
    loading,
    error,
    executeSubagent,
    refreshSubagents,
  };

  return (
    <SubagentContext.Provider value={value}>
      {children}
    </SubagentContext.Provider>
  );
};