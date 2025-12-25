import { PrismaClient, User as PrismaUser } from '@prisma/client';

export interface UserBackground {
  softwareExperience?: string; // beginner, intermediate, advanced
  programmingLanguages?: string[]; // Known programming languages
  devExperienceYears?: number; // Years of development experience
  hardwareSpecs?: {
    deviceType?: string; // laptop/desktop/tablet
    os?: string; // Operating system
    cpu?: string; // CPU specifications
    gpu?: string; // GPU specifications (if applicable)
    ram?: number; // RAM in GB
  };
  developmentFocus?: string[]; // Development areas of focus (web/mobile/AI/ML/etc.)
}

export interface CreateUserInput {
  email: string;
  password: string;
  background?: UserBackground;
}

export interface UpdateUserInput {
  background?: UserBackground;
}

export class UserModel {
  private prisma: PrismaClient;

  constructor(prisma: PrismaClient) {
    this.prisma = prisma;
  }

  async create(userData: CreateUserInput): Promise<PrismaUser> {
    const { email, password, background } = userData;

    return this.prisma.user.create({
      data: {
        email,
        password, // In a real implementation, this should be hashed
        softwareExperience: background?.softwareExperience,
        programmingLanguages: background?.programmingLanguages || [],
        devExperienceYears: background?.devExperienceYears,
        hardwareSpecs: background?.hardwareSpecs,
        developmentFocus: background?.developmentFocus || [],
      },
    });
  }

  async findByEmail(email: string): Promise<PrismaUser | null> {
    return this.prisma.user.findUnique({
      where: { email },
    });
  }

  async findById(id: string): Promise<PrismaUser | null> {
    return this.prisma.user.findUnique({
      where: { id },
    });
  }

  async updateBackground(userId: string, background: UserBackground): Promise<PrismaUser> {
    return this.prisma.user.update({
      where: { id: userId },
      data: {
        softwareExperience: background.softwareExperience,
        programmingLanguages: background.programmingLanguages,
        devExperienceYears: background.devExperienceYears,
        hardwareSpecs: background.hardwareSpecs,
        developmentFocus: background.developmentFocus,
      },
    });
  }

  async getBackground(userId: string): Promise<UserBackground | null> {
    const user = await this.prisma.user.findUnique({
      where: { id: userId },
      select: {
        softwareExperience: true,
        programmingLanguages: true,
        devExperienceYears: true,
        hardwareSpecs: true,
        developmentFocus: true,
      },
    });

    if (!user) {
      return null;
    }

    return {
      softwareExperience: user.softwareExperience,
      programmingLanguages: user.programmingLanguages,
      devExperienceYears: user.devExperienceYears,
      hardwareSpecs: user.hardwareSpecs as any, // Prisma returns JSON as any
      developmentFocus: user.developmentFocus,
    };
  }
}