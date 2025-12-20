import { Request, Response } from 'express';
import { PrismaClient } from '@prisma/client';
import { UserModel, CreateUserInput } from '../../models/user';
import { Logger } from '../../middleware/logger';
import bcrypt from 'bcrypt';

const SALT_ROUNDS = 10;

const hashPassword = async (password: string): Promise<string> => {
  return await bcrypt.hash(password, SALT_ROUNDS);
};

export const signupHandler = async (req: Request, res: Response): Promise<void> => {
  try {
    const prisma = new PrismaClient();
    const userModel = new UserModel(prisma);

    const { email, password, ...background } = req.body;

    // Validate required fields
    if (!email || !password) {
      res.status(400).json({
        success: false,
        message: 'Email and password are required'
      });
      return;
    }

    // Check if user already exists
    const existingUser = await userModel.findByEmail(email);
    if (existingUser) {
      res.status(409).json({
        success: false,
        message: 'User with this email already exists'
      });
      return;
    }

    // Hash the password
    const hashedPassword = await hashPassword(password);

    // Create user with background information
    const userData: CreateUserInput = {
      email,
      password: hashedPassword,
      background: {
        softwareExperience: background.softwareExperience,
        programmingLanguages: Array.isArray(background.programmingLanguages)
          ? background.programmingLanguages
          : [],
        devExperienceYears: background.devExperienceYears ? Number(background.devExperienceYears) : undefined,
        hardwareSpecs: background.hardwareSpecs || undefined,
        developmentFocus: Array.isArray(background.developmentFocus)
          ? background.developmentFocus
          : [],
      }
    };

    const user = await userModel.create(userData);

    // In a real implementation, we would create an authentication session
    // For now, we'll return a simple success response
    res.status(201).json({
      success: true,
      user: {
        id: user.id,
        email: user.email,
        createdAt: user.createdAt
      },
      message: 'User created successfully'
    });

    Logger.info(`New user registered: ${email}`);
  } catch (error) {
    Logger.error('Signup error', { error });
    res.status(500).json({
      success: false,
      message: 'Internal server error'
    });
  } finally {
    await new PrismaClient().$disconnect();
  }
};