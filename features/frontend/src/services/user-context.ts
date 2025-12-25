import React, { createContext, useContext, useReducer, useEffect } from 'react';

interface User {
  id: string;
  email: string;
  softwareExperience?: string;
  programmingLanguages?: string[];
  devExperienceYears?: number;
  hardwareSpecs?: {
    deviceType?: string;
    os?: string;
    cpu?: string;
    gpu?: string;
    ram?: number;
  };
  developmentFocus?: string[];
  createdAt: string;
  updatedAt: string;
}

interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  loading: boolean;
}

interface AuthAction {
  type: string;
  payload?: any;
}

const initialState: AuthState = {
  user: null,
  token: null,
  isAuthenticated: false,
  loading: true,
};

// Define action types
const SET_USER = 'SET_USER';
const SET_TOKEN = 'SET_TOKEN';
const CLEAR_AUTH = 'CLEAR_AUTH';
const SET_LOADING = 'SET_LOADING';

const authReducer = (state: AuthState, action: AuthAction): AuthState => {
  switch (action.type) {
    case SET_USER:
      return {
        ...state,
        user: action.payload,
        isAuthenticated: !!action.payload,
        loading: false,
      };
    case SET_TOKEN:
      return {
        ...state,
        token: action.payload,
      };
    case CLEAR_AUTH:
      return {
        ...initialState,
        loading: false,
      };
    case SET_LOADING:
      return {
        ...state,
        loading: action.payload,
      };
    default:
      return state;
  }
};

interface AuthContextType {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  loading: boolean;
  login: (email: string, password: string) => Promise<boolean>;
  signup: (userData: any) => Promise<boolean>;
  logout: () => void;
  fetchUserProfile: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

interface AuthProviderProps {
  children: React.ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [state, dispatch] = useReducer(authReducer, initialState);

  // Check for existing session on initial load
  useEffect(() => {
    const token = localStorage.getItem('authToken');
    if (token) {
      dispatch({ type: SET_TOKEN, payload: token });
      fetchUserProfile();
    } else {
      dispatch({ type: SET_LOADING, payload: false });
    }
  }, []);

  const login = async (email: string, password: string): Promise<boolean> => {
    try {
      dispatch({ type: SET_LOADING, payload: true });

      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email, password }),
      });

      const data = await response.json();

      if (response.ok) {
        // Store token in localStorage
        const token = data.token || 'dummy-token'; // In a real implementation, the token would come from the response
        localStorage.setItem('authToken', token);
        dispatch({ type: SET_TOKEN, payload: token });

        // Fetch user profile
        await fetchUserProfile();

        return true;
      } else {
        throw new Error(data.message || 'Login failed');
      }
    } catch (error) {
      console.error('Login error:', error);
      return false;
    } finally {
      dispatch({ type: SET_LOADING, payload: false });
    }
  };

  const signup = async (userData: any): Promise<boolean> => {
    try {
      dispatch({ type: SET_LOADING, payload: true });

      const response = await fetch('/api/auth/signup', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(userData),
      });

      const data = await response.json();

      if (response.ok) {
        // For signup, we might want to automatically log in or redirect to login
        // For this implementation, we'll return success
        return true;
      } else {
        throw new Error(data.message || 'Signup failed');
      }
    } catch (error) {
      console.error('Signup error:', error);
      return false;
    } finally {
      dispatch({ type: SET_LOADING, payload: false });
    }
  };

  const logout = () => {
    localStorage.removeItem('authToken');
    dispatch({ type: CLEAR_AUTH });
  };

  const fetchUserProfile = async () => {
    try {
      const token = localStorage.getItem('authToken');
      if (!token) {
        dispatch({ type: SET_LOADING, payload: false });
        return;
      }

      const response = await fetch('/api/auth/me', {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const userData = await response.json();
        dispatch({ type: SET_USER, payload: userData });
      } else {
        // If token is invalid, clear auth state
        localStorage.removeItem('authToken');
        dispatch({ type: CLEAR_AUTH });
      }
    } catch (error) {
      console.error('Error fetching user profile:', error);
      localStorage.removeItem('authToken');
      dispatch({ type: CLEAR_AUTH });
    }
  };

  const value = {
    user: state.user,
    token: state.token,
    isAuthenticated: state.isAuthenticated,
    loading: state.loading,
    login,
    signup,
    logout,
    fetchUserProfile,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};