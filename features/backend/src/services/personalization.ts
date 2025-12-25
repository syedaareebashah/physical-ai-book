import { PrismaClient } from '@prisma/client';
import { ChapterModel } from '../models/chapter';
import { UserModel, UserBackground } from '../models/user';
import { SubagentService, SubagentOutput } from './subagents';
import { Logger } from '../middleware/logger';

export interface PersonalizationInput {
  userId: string;
  chapterId: string;
  content: string;
}

export interface PersonalizationOutput {
  chapterId: string;
  originalContent: string;
  personalizedContent: string;
  personalizationMetadata: {
    userExperienceLevel: string;
    appliedChanges: string[];
    processingTime: number;
  };
}

export class PersonalizationService {
  private prisma: PrismaClient;
  private chapterModel: ChapterModel;
  private userModel: UserModel;
  private subagentService: SubagentService;

  constructor(prisma: PrismaClient, subagentService: SubagentService) {
    this.prisma = prisma;
    this.chapterModel = new ChapterModel(prisma);
    this.userModel = new UserModel(prisma);
    this.subagentService = subagentService;
  }

  async personalizeContent(input: PersonalizationInput): Promise<PersonalizationOutput> {
    const start = Date.now();

    try {
      // Get user background
      const userBackground = await this.userModel.getBackground(input.userId);
      if (!userBackground) {
        throw new Error(`User background not found for user: ${input.userId}`);
      }

      // Check if personalized content is already cached
      const cachedPersonalizedContent = await this.chapterModel.getPersonalizedContent(
        input.chapterId,
        userBackground
      );

      if (cachedPersonalizedContent) {
        Logger.info(`Using cached personalized content for user: ${input.userId}, chapter: ${input.chapterId}`);
        return {
          chapterId: input.chapterId,
          originalContent: input.content,
          personalizedContent: cachedPersonalizedContent,
          personalizationMetadata: {
            userExperienceLevel: userBackground.softwareExperience || 'unknown',
            appliedChanges: ['retrieved from cache'],
            processingTime: Date.now() - start,
          },
        };
      }

      // Use Claude Subagent to personalize content
      const subagentInput = {
        userId: input.userId,
        content: input.content,
        context: {
          userBackground,
          personalizationLevel: 'code-examples-and-terminology',
        },
      };

      // Execute the personalization subagent
      const subagentOutput: SubagentOutput = await this.subagentService.executeSubagent(
        'personalization-agent',
        subagentInput
      );

      const personalizedContent = subagentOutput.result;

      // Save the personalized content to cache
      await this.chapterModel.savePersonalizedContent(
        input.chapterId,
        userBackground,
        personalizedContent
      );

      const processingTime = Date.now() - start;

      Logger.info(`Content personalized for user: ${input.userId}, chapter: ${input.chapterId}`, {
        processingTime,
        userExperienceLevel: userBackground.softwareExperience,
      });

      return {
        chapterId: input.chapterId,
        originalContent: input.content,
        personalizedContent,
        personalizationMetadata: {
          userExperienceLevel: userBackground.softwareExperience || 'unknown',
          appliedChanges: [
            'Adjusted technical terminology',
            'Added relevant code examples',
            'Simplified complex concepts based on experience level',
          ],
          processingTime,
        },
      };
    } catch (error) {
      const processingTime = Date.now() - start;
      Logger.error(`Personalization failed for user: ${input.userId}, chapter: ${input.chapterId}`, { error });

      // Return original content if personalization fails
      return {
        chapterId: input.chapterId,
        originalContent: input.content,
        personalizedContent: input.content,
        personalizationMetadata: {
          userExperienceLevel: 'unknown',
          appliedChanges: ['personalization failed - returned original content'],
          processingTime,
        },
      };
    }
  }

  async getPersonalizationPreferences(userId: string): Promise<any> {
    // In a real implementation, this would fetch user's personalization preferences
    // For now, we'll return default preferences based on user background
    const userBackground = await this.userModel.getBackground(userId);

    return {
      userId,
      preferences: {
        defaultPersonalization: true,
        contentComplexity: userBackground?.softwareExperience || 'intermediate',
        preferredLanguages: ['en', 'ur'],
        lastPersonalizedChapters: [] // Would be populated with actual data in a real implementation
      }
    };
  }

  async updatePersonalizationPreferences(userId: string, preferences: any): Promise<any> {
    // In a real implementation, this would update user's personalization preferences in the database
    // For now, we'll just return the preferences
    return {
      success: true,
      preferences
    };
  }
}