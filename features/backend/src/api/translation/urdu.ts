import { Request, Response } from 'express';
import { PrismaClient } from '@prisma/client';
import { TranslationService } from '../../services/translation';
import { SubagentService } from '../../services/subagents';
import { Logger } from '../../middleware/logger';
import { AuthRequest } from '../../middleware/auth-guard';

export const urduHandler = async (req: AuthRequest, res: Response): Promise<void> => {
  try {
    const prisma = new PrismaClient();

    // Initialize subagent service
    const subagentService = new SubagentService(prisma);
    await subagentService.initialize();

    // Initialize translation service
    const translationService = new TranslationService(prisma, subagentService);

    const { chapterId, content, preserveFormatting } = req.body;

    // Validate required fields
    if (!chapterId || !content) {
      res.status(400).json({
        success: false,
        message: 'Chapter ID and content are required'
      });
      return;
    }

    // Check if user is authenticated
    if (!req.user) {
      res.status(401).json({
        success: false,
        message: 'User not authenticated'
      });
      return;
    }

    // Perform translation to Urdu
    const result = await translationService.translateToUrdu({
      chapterId,
      content,
      preserveFormatting: preserveFormatting !== false // Default to true if not specified
    });

    res.status(200).json(result);

    Logger.info(`Content translated to Urdu for user: ${req.user.id}, chapter: ${chapterId}`);
  } catch (error) {
    Logger.error('Translation to Urdu error', { error });
    res.status(500).json({
      success: false,
      message: 'Internal server error during translation'
    });
  } finally {
    await new PrismaClient().$disconnect();
  }
};