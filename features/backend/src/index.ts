import express from 'express';
import cors from 'cors';
import { config } from './config';
import apiRouter from './api';
import { ErrorHandler, notFoundHandler } from './middleware/error-handler';
import { requestLogger, Logger } from './middleware/logger';
import { securityHelmet, limiter, securityHeaders } from './middleware/security';
import { auth } from './lib/auth';

const app = express();

// Initialize logger
Logger.init();

// Security middleware
app.use(securityHelmet);
app.use(securityHeaders);

// Rate limiting
app.use(limiter);

// CORS
app.use(cors({
  origin: config.cors.origin,
  credentials: true,
}));

// Request logging
app.use(requestLogger);

// Body parsing middleware
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// Add the auth middleware from Better Auth
app.use(auth);

// API routes
app.use('/api', apiRouter);

// Health check endpoint
app.get('/health', (req, res) => {
  res.status(200).json({
    status: 'OK',
    timestamp: new Date().toISOString(),
    version: '1.0.0'
  });
});

// Catch-all for undefined routes
app.use(notFoundHandler);

// Error handling middleware (should be last)
app.use(ErrorHandler.handle);

// Start the server
const server = app.listen(config.port, () => {
  Logger.info(`Server running on port ${config.port} in ${config.nodeEnv} mode`);
  console.log(`🚀 Server ready at http://localhost:${config.port}`);
  console.log(`📊 Health check at http://localhost:${config.port}/health`);
});

// Graceful shutdown
process.on('SIGTERM', () => {
  Logger.info('SIGTERM received, shutting down gracefully');
  server.close(() => {
    Logger.info('Process terminated');
  });
});

process.on('SIGINT', () => {
  Logger.info('SIGINT received, shutting down gracefully');
  server.close(() => {
    Logger.info('Process terminated');
  });
});

export default app;