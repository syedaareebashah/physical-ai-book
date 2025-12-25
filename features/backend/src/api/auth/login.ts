import { Request, Response } from 'express';
import { PrismaClient } from '@prisma/client';
import { UserModel } from '../../models/user';
import { Logger } from '../../middleware/logger';
import bcrypt from 'bcrypt';

export const loginHandler = async (req: Request, res: Response): Promise<void> => {
  try {
    const prisma = new PrismaClient();
    const userModel = new UserModel(prisma);

    const { email, password } = req.body;

    // Validate required fields
    if (!email || !password) {
      res.status(400).json({
        success: false,
        message: 'Email and password are required'
      });
      return;
    }

    // Find user by email
    const user = await userModel.findByEmail(email);
    if (!user) {
      res.status(401).json({
        success: false,
        message: 'Invalid credentials'
      });
      return;
    }

    // Compare passwords (in a real implementation)
    // Note: In a real implementation, we would compare the hashed password
    const passwordValid = await bcrypt.compare(password, user.password);
    if (!passwordValid) {
      res.status(401).json({
        success: false,
        message: 'Invalid credentials'
      });
      return;
    }

    // In a real implementation, we would create an authentication session
    // For now, we'll return a simple success response with user info
    res.status(200).json({
      success: true,
      user: {
        id: user.id,
        email: user.email
      },
      message: 'Login successful'
    });

    Logger.info(`User logged in: ${email}`);
  } catch (error) {
    Logger.error('Login error', { error });
    res.status(500).json({
      success: false,
      message: 'Internal server error'
    });
  } finally {
    await new PrismaClient().$disconnect();
  }
};