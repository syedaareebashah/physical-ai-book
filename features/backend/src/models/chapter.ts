import { PrismaClient, Chapter as PrismaChapter } from '@prisma/client';
import { UserBackground } from './user';

export interface ChapterContent {
  id: string;
  title: string;
  content: string;
  personalizedContent?: Record<string, string>; // Cache of personalized versions keyed by user background profile
  urduContent?: string; // Urdu translation of content
  createdAt: Date;
  updatedAt: Date;
}

export interface CreateChapterInput {
  title: string;
  content: string;
}

export interface UpdateChapterInput {
  title?: string;
  content?: string;
  personalizedContent?: Record<string, string>;
  urduContent?: string;
}

export class ChapterModel {
  private prisma: PrismaClient;

  constructor(prisma: PrismaClient) {
    this.prisma = prisma;
  }

  async create(chapterData: CreateChapterInput): Promise<PrismaChapter> {
    const { title, content } = chapterData;

    return this.prisma.chapter.create({
      data: {
        title,
        content,
      },
    });
  }

  async findById(id: string): Promise<PrismaChapter | null> {
    return this.prisma.chapter.findUnique({
      where: { id },
    });
  }

  async findAll(): Promise<PrismaChapter[]> {
    return this.prisma.chapter.findMany({
      orderBy: { createdAt: 'asc' },
    });
  }

  async update(id: string, chapterData: UpdateChapterInput): Promise<PrismaChapter> {
    return this.prisma.chapter.update({
      where: { id },
      data: {
        title: chapterData.title,
        content: chapterData.content,
        personalizedContent: chapterData.personalizedContent,
        urduContent: chapterData.urduContent,
      },
    });
  }

  async delete(id: string): Promise<PrismaChapter> {
    return this.prisma.chapter.delete({
      where: { id },
    });
  }

  async getPersonalizedContent(chapterId: string, userBackground: UserBackground): Promise<string | null> {
    const chapter = await this.prisma.chapter.findUnique({
      where: { id: chapterId },
    });

    if (!chapter) {
      return null;
    }

    // Create a key based on user background to look up cached personalized content
    const backgroundKey = this.generateBackgroundKey(userBackground);

    if (chapter.personalizedContent && typeof chapter.personalizedContent === 'object') {
      const personalizedContent = chapter.personalizedContent as Record<string, string>;
      return personalizedContent[backgroundKey] || null;
    }

    return null;
  }

  async savePersonalizedContent(chapterId: string, userBackground: UserBackground, personalizedContent: string): Promise<void> {
    const chapter = await this.prisma.chapter.findUnique({
      where: { id: chapterId },
    });

    if (!chapter) {
      throw new Error(`Chapter with id ${chapterId} not found`);
    }

    // Create a key based on user background
    const backgroundKey = this.generateBackgroundKey(userBackground);

    // Update the personalizedContent field with the new personalized version
    const updatedPersonalizedContent = {
      ...(chapter.personalizedContent as Record<string, string> || {}),
      [backgroundKey]: personalizedContent,
    };

    await this.prisma.chapter.update({
      where: { id: chapterId },
      data: {
        personalizedContent: updatedPersonalizedContent,
      },
    });
  }

  private generateBackgroundKey(userBackground: UserBackground): string {
    // Create a consistent key based on user background properties
    return JSON.stringify({
      softwareExperience: userBackground.softwareExperience,
      programmingLanguages: userBackground.programmingLanguages?.sort(),
      devExperienceYears: userBackground.devExperienceYears,
      developmentFocus: userBackground.developmentFocus?.sort(),
    });
  }
}