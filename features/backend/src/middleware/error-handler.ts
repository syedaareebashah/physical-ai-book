import { Request, Response, NextFunction } from 'express';
import { Logger } from './logger';

export interface ApiError extends Error {
  statusCode?: number;
  isOperational?: boolean;
}

export class ErrorHandler {
  static handle(err: ApiError, req: Request, res: Response, next: NextFunction): void {
    const statusCode = err.statusCode || 500;
    const message = err.isOperational ? err.message : 'Internal server error';

    Logger.error(`${req.method} ${req.path} - ${statusCode}: ${message}`, {
      error: err.stack,
      url: req.url,
      method: req.method,
      ip: req.ip,
      userAgent: req.get('User-Agent'),
    });

    res.status(statusCode).json({
      success: false,
      message,
      ...(process.env.NODE_ENV === 'development' && { stack: err.stack }),
    });
  }
}

export const notFoundHandler = (req: Request, res: Response, next: NextFunction): void => {
  const error: ApiError = new Error(`Route not found: ${req.originalUrl}`);
  error.statusCode = 404;
  next(error);
};