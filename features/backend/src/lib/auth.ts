import { betterAuth } from "better-auth";
import { prismaAdapter } from "@better-auth/adapter-prisma";
import { PrismaClient } from "@prisma/client";

const prisma = new PrismaClient();

export const auth = betterAuth({
  database: prismaAdapter(prisma, {
    provider: "postgresql",
  }),
  secret: process.env.BETTER_AUTH_SECRET || "your-secret-key-here",
  baseURL: process.env.BETTER_AUTH_URL || "http://localhost:3000",
  emailAndPassword: {
    enabled: true,
    requireEmailVerification: false,
  },
  user: {
    // Extend the user model to include background information
    additionalFields: {
      softwareExperience: {
        type: "string",
        required: false,
      },
      programmingLanguages: {
        type: "string",
        required: false,
        input: "json",
      },
      devExperienceYears: {
        type: "number",
        required: false,
      },
      hardwareSpecs: {
        type: "string",
        required: false,
        input: "json",
      },
      developmentFocus: {
        type: "string",
        required: false,
        input: "json",
      },
    },
  },
});