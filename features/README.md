# Hackathon Book Project with RAG Chatbot, Authentication, Personalization, and Translation Features

This project enhances an existing book platform with RAG chatbot by adding reusable intelligence, secure authentication with user background collection, content personalization, and Urdu translation features to maximize bonus points in the hackathon.

## Features

### 1. Authentication with Background Collection (50 bonus points)
- Secure signup/signin using Better Auth
- Collects user's software and hardware background during signup:
  - Programming experience level (beginner/intermediate/advanced)
  - Known programming languages
  - Years of development experience
  - Hardware specifications (device type, OS, CPU, GPU, RAM)
  - Development focus areas (web, mobile, AI/ML, etc.)

### 2. Content Personalization (50 bonus points)
- Personalization button at the start of each chapter for logged-in users
- Adapts content based on user's stored background information
- Adjusts technical terminology and examples based on experience level
- Caches personalized content for performance

### 3. Urdu Translation (50 bonus points)
- Translation button at the start of each chapter for logged-in users
- Converts content to Urdu with proper RTL (right-to-left) rendering
- Maintains formatting and code snippets during translation
- Handles untranslatable content with fallbacks

### 4. Reusable Intelligence (50 bonus points)
- Claude Code Subagents for modular functionality
- Reusable Agent Skills for common tasks
- Personalization and translation subagents
- Configurable subagent management API

## Tech Stack

- **Backend**: Node.js, Express, TypeScript
- **Frontend**: React, Next.js, TypeScript
- **Authentication**: Better Auth
- **Database**: PostgreSQL with Prisma ORM
- **AI Integration**: Claude Code Subagents
- **Styling**: Tailwind CSS

## Getting Started

See the [Quickstart Guide](specs/1-book-enhancement/quickstart.md) for detailed setup instructions.

## API Endpoints

### Authentication
- `POST /api/auth/signup` - User registration with background questions
- `POST /api/auth/login` - User login
- `GET /api/auth/me` - Get current user profile

### Personalization
- `POST /api/personalization/process` - Process content personalization
- `GET /api/personalization/preferences` - Get user preferences

### Translation
- `POST /api/translation/urdu` - Translate content to Urdu
- `POST /api/translation/validate` - Validate content for translation
- `GET /api/translation/status` - Check translation service status

### Subagents
- `POST /api/subagents/register` - Register a new subagent
- `GET /api/subagents` - List all subagents
- `PUT /api/subagents/:id` - Update a subagent
- `DELETE /api/subagents/:id` - Delete a subagent

## Project Structure

```
project-root/
├── backend/                 # Backend server and API
│   ├── src/
│   │   ├── models/          # Data models
│   │   ├── services/        # Business logic (subagents, personalization, translation)
│   │   ├── api/            # API routes
│   │   ├── middleware/     # Authentication guards, logging, security
│   │   ├── lib/            # Authentication library config
│   │   └── config/         # Environment config
│   ├── prisma/             # Database schema
│   └── tests/
├── frontend/               # Frontend application
│   ├── src/
│   │   ├── components/     # React components (auth, chapters, subagents)
│   │   ├── pages/         # Page components
│   │   ├── services/      # API clients, user context
│   │   └── styles/        # Global styles (with RTL support)
│   └── tests/
├── specs/                 # Feature specifications
└── .env.example          # Example environment variables
```

## Security Features

- Rate limiting for authentication endpoints
- Helmet security headers
- Password hashing with bcrypt
- Input sanitization
- Session management with Better Auth
- Encrypted data storage

## Performance Optimizations

- Content caching for personalized versions
- Efficient database queries with Prisma
- Asynchronous processing for AI operations
- Designed to maintain no more than 10% performance degradation for existing RAG chatbot

## Hackathon Submission

This project was created for a hackathon with a focus on earning maximum bonus points through:
- Reusable intelligence implementation using Claude Code Subagents
- Authentication with comprehensive background questions
- Content personalization based on user profile
- Urdu translation functionality
- Clean integration with existing RAG chatbot