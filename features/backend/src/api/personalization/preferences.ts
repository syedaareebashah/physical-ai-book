import { Request, Response } from 'express';
import { PrismaClient } from '@prisma/client';
import { PersonalizationService } from '../../services/personalization';
import { Logger } from '../../middleware/logger';
import { AuthRequest } from '../../middleware/auth-guard';

export const preferencesHandler = async (req: AuthRequest, res: Response): Promise<void> => {
  try {
    const prisma = new PrismaClient();
    const personalizationService = new PersonalizationService(
      prisma,
      // We'll pass a minimal subagent service for now - in a real implementation
      // we'd initialize it properly
      new (class {
        async initialize() {}
        async executeSubagent() { return { result: '', processingTime: 0 }; }
      })() as any
    );

    const userId = req.user?.id;
    if (!userId) {
      res.status(401).json({
        success: false,
        message: 'User not authenticated'
      });
      return;
    }

    if (req.method === 'GET') {
      // Get user preferences
      const preferences = await personalizationService.getPersonalizationPreferences(userId);
      res.status(200).json(preferences);

      Logger.info(`Personalization preferences retrieved for user: ${userId}`);
    } else if (req.method === 'PUT') {
      // Update user preferences
      const preferences = await personalizationService.updatePersonalizationPreferences(userId, req.body.preferences);
      res.status(200).json(preferences);

      Logger.info(`Personalization preferences updated for user: ${userId}`);
    } else {
      res.status(405).json({
        success: false,
        message: 'Method not allowed'
      });
    }
  } catch (error) {
    Logger.error('Personalization preferences error', { error });
    res.status(500).json({
      success: false,
      message: 'Internal server error'
    });
  } finally {
    await new PrismaClient().$disconnect();
  }
};