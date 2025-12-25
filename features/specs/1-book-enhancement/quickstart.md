# Quickstart Guide: Hackathon Book Project

## Setup Instructions

### Prerequisites
- Node.js (v18 or higher)
- npm or yarn package manager
- Git version control
- Speckitplus framework installed
- PostgreSQL database (or your preferred database)

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd <project-directory>
   ```

2. **Install dependencies**
   ```bash
   npm install
   # or
   yarn install
   ```

3. **Set up environment variables**
   Create a `.env` file in the root directory with the following variables:
   ```env
   # Better Auth Configuration
   BETTER_AUTH_SECRET=your-secret-key-here
   BETTER_AUTH_URL=http://localhost:3000

   # Database Configuration (example for PostgreSQL)
   DATABASE_URL=postgresql://username:password@localhost:5432/book_project

   # Claude API Configuration (if needed)
   CLAUDE_API_KEY=your-claude-api-key
   ```

4. **Initialize the database**
   ```bash
   cd backend
   npx prisma db push
   # or your database initialization command
   ```

5. **Run the development server**
   ```bash
   npm run dev
   # or
   yarn dev
   ```

6. **Access the application**
   - Frontend: http://localhost:3000
   - API: http://localhost:3000/api
   - Admin panel: http://localhost:3000/admin (if applicable)

## Key Features Setup

### Authentication with Better Auth
The application uses Better Auth for user authentication. During signup, users will be prompted to provide their software and hardware background information:

1. Navigate to `/signup`
2. Provide email and password
3. Complete the multi-step form with:
   - Programming experience level
   - Known programming languages
   - Years of development experience
   - Hardware specifications
   - Development focus areas

### Claude Code Subagents
Reusable intelligence components are implemented using Claude Code Subagents:

1. Subagents are configured in the `backend/src/services/subagents` directory
2. Each subagent handles specific tasks like personalization and translation
3. Configuration files define parameters and behavior
4. Subagents can be managed via the `/api/subagents` endpoints

### Personalization Feature
To use the personalization feature:

1. Log in to your account
2. Navigate to any chapter
3. Click the "Personalize Content" button
4. Content will adapt based on your stored background information

### Urdu Translation Feature
To use the Urdu translation feature:

1. Log in to your account
2. Navigate to any chapter
3. Click the "Translate to Urdu" button
4. Content will be displayed in Urdu script with RTL support

## Development Commands

```bash
# Run development server (both frontend and backend)
npm run dev

# Run backend only
cd backend && npm run dev

# Run frontend only
cd frontend && npm run dev

# Build for production
npm run build

# Run tests
npm run test

# Run tests with coverage
npm run test:coverage

# Lint code
npm run lint

# Format code
npm run format

# Run database migrations
cd backend && npx prisma migrate dev

# Generate Prisma client
cd backend && npx prisma generate
```

## API Endpoints

### Authentication
- `POST /api/auth/signup` - User registration with background questions
- `POST /api/auth/login` - User login
- `POST /api/auth/logout` - User logout
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

### Chapters
- `GET /api/chapters/:id` - Get chapter content
- `PUT /api/chapters/:id` - Update chapter content (admin only)

## Testing the Features

### Authentication Flow
1. Visit `/signup` and create a new account
2. Complete the background questionnaire
3. Verify your email (if required)
4. Log in at `/login`
5. Access protected content

### Personalization Testing
1. Create accounts with different background profiles
2. Log in to each account
3. Visit the same chapter and click "Personalize Content"
4. Observe how content adapts to different experience levels

### Translation Testing
1. Log in to an account
2. Navigate to any chapter
3. Click "Translate to Urdu"
4. Verify the content appears in Urdu script with proper RTL rendering

### Subagent Testing
1. Monitor the console/logs for subagent invocations
2. Verify that personalization and translation use Claude Code Subagents
3. Check that subagents are reusable across different chapters
4. Use the `/api/subagents` endpoints to manage subagents

## Troubleshooting

### Common Issues
- **Authentication fails**: Ensure Better Auth is properly configured with correct environment variables
- **Translation not working**: Verify database is properly initialized and subagents are registered
- **Personalization not adapting**: Check that user profile has complete background information
- **Subagents not responding**: Ensure subagent service is initialized and registered

### Environment Configuration
- Ensure all required environment variables are set
- Verify database connection strings are correct
- Confirm API endpoints are accessible

## Project Structure

```
project-root/
├── backend/
│   ├── src/
│   │   ├── models/          # Data models
│   │   ├── services/        # Business logic (subagents, personalization, translation)
│   │   ├── api/            # API routes
│   │   ├── middleware/     # Authentication guards, logging
│   │   ├── lib/            # Authentication library config
│   │   └── config/         # Environment config
│   ├── prisma/             # Database schema
│   └── tests/
├── frontend/
│   ├── src/
│   │   ├── components/     # React components (auth, chapters, subagents)
│   │   ├── pages/         # Page components
│   │   ├── services/      # API clients, user context
│   │   └── styles/        # Global styles (with RTL support)
│   └── tests/
├── specs/                 # Feature specifications
└── .env.example          # Example environment variables
```

## Performance Optimization

The application is designed to maintain no more than 10% performance degradation for the existing RAG chatbot while adding the new features. Personalized content is cached to improve performance on subsequent requests.