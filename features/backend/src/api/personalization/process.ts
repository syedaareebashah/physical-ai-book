import { Request, Response } from 'express';
import { PrismaClient } from '@prisma/client';
import { PersonalizationService } from '../../services/personalization';
import { SubagentService } from '../../services/subagents';
import { Logger } from '../../middleware/logger';
import { AuthRequest } from '../../middleware/auth-guard';

export const processHandler = async (req: AuthRequest, res: Response): Promise<void> => {
  try {
    const prisma = new PrismaClient();

    // Initialize subagent service
    const subagentService = new SubagentService(prisma);
    await subagentService.initialize();

    // Initialize personalization service
    const personalizationService = new PersonalizationService(prisma, subagentService);

    const { chapterId, content, userId } = req.body;

    // Validate required fields
    if (!chapterId || !content) {
      res.status(400).json({
        success: false,
        message: 'Chapter ID and content are required'
      });
      return;
    }

    // Use the authenticated user's ID if not provided in the request
    const targetUserId = userId || req.user?.id;
    if (!targetUserId) {
      res.status(401).json({
        success: false,
        message: 'User ID is required for personalization'
      });
      return;
    }

    // Perform personalization
    const result = await personalizationService.personalizeContent({
      userId: targetUserId,
      chapterId,
      content
    });

    res.status(200).json(result);

    Logger.info(`Content personalized for user: ${targetUserId}, chapter: ${chapterId}`);
  } catch (error) {
    Logger.error('Personalization process error', { error });
    res.status(500).json({
      success: false,
      message: 'Internal server error during personalization'
    });
  } finally {
    await new PrismaClient().$disconnect();
  }
};