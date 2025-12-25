import { PrismaClient, Subagent as PrismaSubagent } from '@prisma/client';

export interface SubagentConfig {
  id: string;
  name: string;
  type: 'personalization' | 'translation' | 'content-summary' | 'other';
  description?: string;
  parameters?: Record<string, any>;
  createdAt: Date;
  updatedAt: Date;
}

export interface CreateSubagentInput {
  id: string;
  name: string;
  type: 'personalization' | 'translation' | 'content-summary' | 'other';
  description?: string;
  parameters?: Record<string, any>;
}

export interface UpdateSubagentInput {
  name?: string;
  type?: 'personalization' | 'translation' | 'content-summary' | 'other';
  description?: string;
  parameters?: Record<string, any>;
}

export class SubagentModel {
  private prisma: PrismaClient;

  constructor(prisma: PrismaClient) {
    this.prisma = prisma;
  }

  async create(subagentData: CreateSubagentInput): Promise<PrismaSubagent> {
    const { id, name, type, description, parameters } = subagentData;

    return this.prisma.subagent.create({
      data: {
        id,
        name,
        type,
        description: description || null,
        parameters: parameters || null,
      },
    });
  }

  async findById(id: string): Promise<PrismaSubagent | null> {
    return this.prisma.subagent.findUnique({
      where: { id },
    });
  }

  async findAll(): Promise<PrismaSubagent[]> {
    return this.prisma.subagent.findMany({
      orderBy: { createdAt: 'asc' },
    });
  }

  async update(id: string, subagentData: UpdateSubagentInput): Promise<PrismaSubagent> {
    return this.prisma.subagent.update({
      where: { id },
      data: {
        name: subagentData.name,
        type: subagentData.type,
        description: subagentData.description || null,
        parameters: subagentData.parameters || null,
      },
    });
  }

  async delete(id: string): Promise<PrismaSubagent> {
    return this.prisma.subagent.delete({
      where: { id },
    });
  }

  async getByName(name: string): Promise<PrismaSubagent | null> {
    return this.prisma.subagent.findFirst({
      where: { name },
    });
  }
}