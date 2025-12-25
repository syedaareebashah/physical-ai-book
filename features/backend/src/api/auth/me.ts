import { Request, Response } from 'express';
import { PrismaClient } from '@prisma/client';
import { UserModel } from '../../models/user';
import { Logger } from '../../middleware/logger';
import { AuthRequest } from '../../middleware/auth-guard';

export const meHandler = async (req: AuthRequest, res: Response): Promise<void> => {
  try {
    const prisma = new PrismaClient();
    const userModel = new UserModel(prisma);

    // Get user ID from the authenticated request (added by auth middleware)
    const userId = req.user?.id;
    if (!userId) {
      res.status(401).json({
        success: false,
        message: 'User not authenticated'
      });
      return;
    }

    // Get user details
    const user = await userModel.findById(userId);
    if (!user) {
      res.status(404).json({
        success: false,
        message: 'User not found'
      });
      return;
    }

    // Get user background information
    const userBackground = await userModel.getBackground(userId);

    res.status(200).json({
      id: user.id,
      email: user.email,
      softwareExperience: userBackground?.softwareExperience,
      programmingLanguages: userBackground?.programmingLanguages || [],
      devExperienceYears: userBackground?.devExperienceYears,
      hardwareSpecs: userBackground?.hardwareSpecs,
      developmentFocus: userBackground?.developmentFocus || [],
      createdAt: user.createdAt,
      updatedAt: user.updatedAt
    });

    Logger.info(`User profile accessed: ${user.email}`);
  } catch (error) {
    Logger.error('Get user profile error', { error });
    res.status(500).json({
      success: false,
      message: 'Internal server error'
    });
  } finally {
    await new PrismaClient().$disconnect();
  }
};