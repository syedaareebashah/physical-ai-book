import { PrismaClient } from '@prisma/client';
import { SubagentService, SubagentInput, SubagentOutput } from './subagents';
import { Logger } from '../middleware/logger';

export interface AgentSkill {
  id: string;
  name: string;
  description: string;
  parameters: {
    [key: string]: {
      type: string;
      required: boolean;
      description: string;
    };
  };
  execute: (params: Record<string, any>) => Promise<any>;
}

export class AgentSkillsService {
  private prisma: PrismaClient;
  private subagentService: SubagentService;
  private skills: Map<string, AgentSkill>;

  constructor(prisma: PrismaClient, subagentService: SubagentService) {
    this.prisma = prisma;
    this.subagentService = subagentService;
    this.skills = new Map();

    // Register default skills
    this.registerDefaultSkills();
  }

  private registerDefaultSkills(): void {
    // Skill for content summarization
    this.registerSkill({
      id: 'content-summarizer',
      name: 'Content Summarizer',
      description: 'Summarizes content to a specified length',
      parameters: {
        content: {
          type: 'string',
          required: true,
          description: 'The content to summarize'
        },
        length: {
          type: 'number',
          required: false,
          description: 'Target length in sentences (default: 3)'
        }
      },
      execute: async (params: Record<string, any>) => {
        const { content, length = 3 } = params;
        const subagentInput: SubagentInput = {
          content,
          context: {
            task: 'summarization',
            targetLength: length
          }
        };

        // In a real implementation, we'd have a summary subagent
        // For now, we'll return a simple summary
        return {
          summary: `This is a summary of the content: ${content.substring(0, 100)}...`
        };
      }
    });

    // Skill for content translation
    this.registerSkill({
      id: 'content-translator',
      name: 'Content Translator',
      description: 'Translates content to a target language',
      parameters: {
        content: {
          type: 'string',
          required: true,
          description: 'The content to translate'
        },
        targetLanguage: {
          type: 'string',
          required: true,
          description: 'Target language code (e.g., ur, es, fr)'
        }
      },
      execute: async (params: Record<string, any>) => {
        const { content, targetLanguage } = params;

        if (targetLanguage === 'ur') {
          // Use the translation subagent
          const subagentInput: SubagentInput = {
            content,
            context: {
              targetLanguage,
              preserveFormatting: true
            }
          };

          // In a real implementation, we'd execute the translation subagent
          // For now, we'll return a simple translation
          return {
            translatedContent: `یہ متن اردو میں ترجمہ شدہ ہے:\n${content}`
          };
        }

        // For other languages, return original content
        return { translatedContent: content };
      }
    });

    // Skill for content personalization
    this.registerSkill({
      id: 'content-personalizer',
      name: 'Content Personalizer',
      description: 'Personalizes content based on user profile',
      parameters: {
        content: {
          type: 'string',
          required: true,
          description: 'The content to personalize'
        },
        userId: {
          type: 'string',
          required: true,
          description: 'User ID to get personalization preferences'
        }
      },
      execute: async (params: Record<string, any>) => {
        const { content, userId } = params;

        // In a real implementation, we'd execute the personalization subagent
        // For now, we'll return the content with a personalization note
        return {
          personalizedContent: `Personalized content for user ${userId}:\n${content}`
        };
      }
    });
  }

  registerSkill(skill: AgentSkill): void {
    this.skills.set(skill.id, skill);
    Logger.info(`Registered agent skill: ${skill.name} (${skill.id})`);
  }

  getSkill(id: string): AgentSkill | undefined {
    return this.skills.get(id);
  }

  getAllSkills(): AgentSkill[] {
    return Array.from(this.skills.values());
  }

  async executeSkill(skillId: string, params: Record<string, any>): Promise<any> {
    const skill = this.skills.get(skillId);
    if (!skill) {
      throw new Error(`Skill with id ${skillId} not found`);
    }

    // Validate required parameters
    for (const [paramName, paramDef] of Object.entries(skill.parameters)) {
      if (paramDef.required && params[paramName] === undefined) {
        throw new Error(`Required parameter '${paramName}' is missing for skill '${skillId}'`);
      }
    }

    try {
      const result = await skill.execute(params);
      Logger.info(`Executed agent skill: ${skill.name} (${skillId})`);
      return result;
    } catch (error) {
      Logger.error(`Failed to execute agent skill: ${skill.name} (${skillId})`, { error });
      throw error;
    }
  }

  async registerSubagentAsSkill(subagentId: string): Promise<void> {
    // In a real implementation, this would register a subagent as a reusable skill
    // For now, we'll just log the action
    Logger.info(`Registered subagent as skill: ${subagentId}`);
  }
}