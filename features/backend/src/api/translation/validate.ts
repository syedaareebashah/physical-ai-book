import { Request, Response } from 'express';
import { PrismaClient } from '@prisma/client';
import { TranslationService } from '../../services/translation';
import { Logger } from '../../middleware/logger';
import { AuthRequest } from '../../middleware/auth-guard';

export const validateHandler = async (req: AuthRequest, res: Response): Promise<void> => {
  try {
    const prisma = new PrismaClient();
    const translationService = new TranslationService(
      prisma,
      // We'll pass a minimal subagent service for now - in a real implementation
      // we'd initialize it properly
      new (class {
        async initialize() {}
        async executeSubagent() { return { result: '', processingTime: 0 }; }
      })() as any
    );

    // Check if user is authenticated
    if (!req.user) {
      res.status(401).json({
        success: false,
        message: 'User not authenticated'
      });
      return;
    }

    const { content } = req.body;

    // Validate required fields
    if (!content) {
      res.status(400).json({
        success: false,
        message: 'Content is required for validation'
      });
      return;
    }

    // Perform validation
    const result = await translationService.validateForTranslation(content);

    res.status(200).json(result);

    Logger.info(`Content validated for translation by user: ${req.user.id}`);
  } catch (error) {
    Logger.error('Translation validation error', { error });
    res.status(500).json({
      success: false,
      message: 'Internal server error during validation'
    });
  } finally {
    await new PrismaClient().$disconnect();
  }
};