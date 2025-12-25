import { Request, Response, NextFunction } from 'express';
import { auth } from '../lib/auth';
import { Logger } from './logger';

export interface AuthRequest extends Request {
  user?: {
    id: string;
    email: string;
    // Add other user properties as needed
  };
}

export const authGuard = async (req: AuthRequest, res: Response, next: NextFunction): Promise<void> => {
  try {
    // Extract token from Authorization header
    const authHeader = req.headers.authorization;
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      res.status(401).json({
        success: false,
        message: 'Access token required'
      });
      return;
    }

    const token = authHeader.split(' ')[1];

    // Verify the token using Better Auth
    const session = await auth.$getSessionByToken(token);

    if (!session || !session.user) {
      res.status(401).json({
        success: false,
        message: 'Invalid or expired token'
      });
      return;
    }

    // Add user info to request object
    req.user = {
      id: session.user.id,
      email: session.user.email,
      // Add other user properties as needed
    };

    next();
  } catch (error) {
    Logger.error('Authentication error', { error });
    res.status(401).json({
      success: false,
      message: 'Authentication failed'
    });
  }
};