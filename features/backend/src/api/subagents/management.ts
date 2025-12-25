import { Request, Response } from 'express';
import { PrismaClient } from '@prisma/client';
import { SubagentService } from '../../services/subagents';
import { Logger } from '../../middleware/logger';
import { AuthRequest } from '../../middleware/auth-guard';

// Handler for registering a new subagent
export const registerHandler = async (req: AuthRequest, res: Response): Promise<void> => {
  try {
    const prisma = new PrismaClient();
    const subagentService = new SubagentService(prisma);

    const { id, name, type, description, parameters } = req.body;

    // Validate required fields
    if (!id || !name || !type) {
      res.status(400).json({
        success: false,
        message: 'ID, name, and type are required for subagent registration'
      });
      return;
    }

    // Check if user is authenticated and has admin privileges (simplified check)
    if (!req.user) {
      res.status(401).json({
        success: false,
        message: 'User not authenticated'
      });
      return;
    }

    // Register the subagent
    await subagentService.registerSubagent({
      id,
      name,
      type,
      description,
      parameters
    });

    res.status(201).json({
      success: true,
      message: 'Subagent registered successfully',
      subagent: { id, name, type, description, parameters }
    });

    Logger.info(`Subagent registered by user: ${req.user.id}`, { id, name, type });
  } catch (error) {
    Logger.error('Subagent registration error', { error });
    res.status(500).json({
      success: false,
      message: 'Internal server error during subagent registration'
    });
  } finally {
    await new PrismaClient().$disconnect();
  }
};

// Handler for listing all subagents
export const listHandler = async (req: AuthRequest, res: Response): Promise<void> => {
  try {
    const prisma = new PrismaClient();
    const subagentService = new SubagentService(prisma);
    await subagentService.initialize();

    // Check if user is authenticated
    if (!req.user) {
      res.status(401).json({
        success: false,
        message: 'User not authenticated'
      });
      return;
    }

    const subagents = subagentService.getAllSubagents();

    res.status(200).json({
      success: true,
      subagents
    });

    Logger.info(`Subagents listed for user: ${req.user.id}`, { count: subagents.length });
  } catch (error) {
    Logger.error('Subagent listing error', { error });
    res.status(500).json({
      success: false,
      message: 'Internal server error during subagent listing'
    });
  } finally {
    await new PrismaClient().$disconnect();
  }
};

// Handler for updating a subagent
export const updateHandler = async (req: AuthRequest, res: Response): Promise<void> => {
  try {
    const prisma = new PrismaClient();
    const subagentService = new SubagentService(prisma);

    const { id } = req.params;
    const { name, type, description, parameters } = req.body;

    // Check if user is authenticated
    if (!req.user) {
      res.status(401).json({
        success: false,
        message: 'User not authenticated'
      });
      return;
    }

    // Update the subagent
    await subagentService.updateSubagent(id, {
      name,
      type,
      description,
      parameters
    });

    res.status(200).json({
      success: true,
      message: 'Subagent updated successfully',
      subagent: { id, name, type, description, parameters }
    });

    Logger.info(`Subagent updated by user: ${req.user.id}`, { id });
  } catch (error) {
    Logger.error('Subagent update error', { error });
    res.status(500).json({
      success: false,
      message: 'Internal server error during subagent update'
    });
  } finally {
    await new PrismaClient().$disconnect();
  }
};

// Handler for deleting a subagent
export const deleteHandler = async (req: AuthRequest, res: Response): Promise<void> => {
  try {
    const prisma = new PrismaClient();
    const subagentService = new SubagentService(prisma);

    const { id } = req.params;

    // Check if user is authenticated
    if (!req.user) {
      res.status(401).json({
        success: false,
        message: 'User not authenticated'
      });
      return;
    }

    // Delete the subagent
    await subagentService.deleteSubagent(id);

    res.status(200).json({
      success: true,
      message: 'Subagent deleted successfully',
      subagentId: id
    });

    Logger.info(`Subagent deleted by user: ${req.user.id}`, { id });
  } catch (error) {
    Logger.error('Subagent deletion error', { error });
    res.status(500).json({
      success: false,
      message: 'Internal server error during subagent deletion'
    });
  } finally {
    await new PrismaClient().$disconnect();
  }
};