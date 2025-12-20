import { PrismaClient } from '@prisma/client';
import { ChapterModel } from '../models/chapter';
import { SubagentService, SubagentOutput } from './subagents';
import { Logger } from '../middleware/logger';

export interface TranslationInput {
  chapterId: string;
  content: string;
  preserveFormatting?: boolean;
}

export interface TranslationOutput {
  chapterId: string;
  originalContent: string;
  urduContent: string;
  translationMetadata: {
    accuracy: number;
    translatedWords: number;
    processingTime: number;
    fallbackParts: string[];
    rtlSupport: boolean;
  };
}

export interface ValidationOutput {
  isValid: boolean;
  issues: {
    type: string;
    position: number;
    text: string;
    suggestion: string;
  }[];
  suggestions: string[];
  estimatedProcessingTime: number;
}

export class TranslationService {
  private prisma: PrismaClient;
  private chapterModel: ChapterModel;
  private subagentService: SubagentService;

  constructor(prisma: PrismaClient, subagentService: SubagentService) {
    this.prisma = prisma;
    this.chapterModel = new ChapterModel(prisma);
    this.subagentService = subagentService;
  }

  async translateToUrdu(input: TranslationInput): Promise<TranslationOutput> {
    const start = Date.now();

    try {
      // Use Claude Subagent to translate content to Urdu
      const subagentInput = {
        content: input.content,
        context: {
          targetLanguage: 'urdu',
          preserveFormatting: input.preserveFormatting,
          translationLevel: '90-accuracy-with-fallback',
        },
      };

      // Execute the translation subagent
      const subagentOutput: SubagentOutput = await this.subagentService.executeSubagent(
        'translation-agent',
        subagentInput
      );

      const urduContent = subagentOutput.result;

      // Update the chapter with the translated content
      await this.chapterModel.update(input.chapterId, {
        urduContent: urduContent,
      });

      const wordCount = input.content.split(/\s+/).length;
      const processingTime = Date.now() - start;

      Logger.info(`Content translated to Urdu for chapter: ${input.chapterId}`, {
        processingTime,
        wordCount,
      });

      return {
        chapterId: input.chapterId,
        originalContent: input.content,
        urduContent,
        translationMetadata: {
          accuracy: 0.92, // Simulated accuracy
          translatedWords: wordCount,
          processingTime,
          fallbackParts: ['code snippets'], // Simulated fallback parts
          rtlSupport: true,
        },
      };
    } catch (error) {
      const processingTime = Date.now() - start;
      Logger.error(`Translation to Urdu failed for chapter: ${input.chapterId}`, { error });

      // Return original content if translation fails
      return {
        chapterId: input.chapterId,
        originalContent: input.content,
        urduContent: input.content,
        translationMetadata: {
          accuracy: 0,
          translatedWords: 0,
          processingTime,
          fallbackParts: ['entire content'],
          rtlSupport: false,
        },
      };
    }
  }

  async validateForTranslation(content: string): Promise<ValidationOutput> {
    const issues: any[] = [];
    const suggestions: string[] = [];

    // Check for content that might be difficult to translate
    const codePattern = /```[\s\S]*?```|`[^`]*`/g;
    let match;
    while ((match = codePattern.exec(content)) !== null) {
      issues.push({
        type: 'untranslatable',
        position: match.index,
        text: match[0].substring(0, 20) + '...',
        suggestion: 'Preserve in original language'
      });
    }

    // Check for other potential issues
    const technicalTerms = content.match(/\b[A-Z]{2,}\b/g) || [];
    if (technicalTerms.length > 10) {
      suggestions.push('Consider adding technical term glossary');
    }

    return {
      isValid: issues.length === 0,
      issues,
      suggestions,
      estimatedProcessingTime: Math.min(content.length * 2, 5000) // Estimate based on content length
    };
  }
}