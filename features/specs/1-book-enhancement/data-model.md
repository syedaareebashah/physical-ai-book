# Data Model: Hackathon Book Project

## Entity: User Profile
**Description**: Stores user authentication data and background information for personalization

**Fields**:
- `id` (string): Unique user identifier
- `email` (string): User's email address (required, unique)
- `password` (string): Hashed password (managed by Better Auth)
- `createdAt` (timestamp): Account creation date
- `updatedAt` (timestamp): Last update timestamp
- `softwareExperience` (string): Programming experience level (beginner/intermediate/advanced)
- `programmingLanguages` (array of strings): Known programming languages
- `devExperienceYears` (number): Years of development experience
- `hardwareSpecs` (object): Primary hardware information
  - `deviceType` (string): laptop/desktop/tablet
  - `os` (string): Operating system
  - `cpu` (string): CPU specifications
  - `gpu` (string): GPU specifications (if applicable)
  - `ram` (number): RAM in GB
- `developmentFocus` (array of strings): Development areas of focus (web/mobile/AI/ML/etc.)

**Validation Rules**:
- Email must be valid email format
- Software experience must be one of: "beginner", "intermediate", "advanced"
- Programming languages array must contain valid language names
- Dev experience years must be a positive number
- Hardware specs must be properly structured object

**Relationships**:
- One-to-many with user sessions (managed by Better Auth)

## Entity: Chapter Content
**Description**: Represents book chapter content that can be personalized and translated

**Fields**:
- `id` (string): Unique chapter identifier
- `title` (string): Chapter title
- `content` (string): Original English content
- `personalizedContent` (object): Cache of personalized versions
  - keys: user background profiles
  - values: personalized content strings
- `urduContent` (string): Urdu translation of content (if generated)
- `createdAt` (timestamp): Creation timestamp
- `updatedAt` (timestamp): Last update timestamp

**Validation Rules**:
- Title and content must not be empty
- Urdu content must be valid Urdu script when present
- Personalized content cache should have reasonable size limits

**Relationships**:
- Many-to-many with users (through personalization preferences)

## Entity: Subagent Configuration
**Description**: Configuration for Claude Code Subagents used in the system

**Fields**:
- `id` (string): Unique subagent identifier
- `name` (string): Display name for the subagent
- `type` (string): Type of subagent (personalization/translation/other)
- `description` (string): Purpose and functionality description
- `parameters` (object): Configuration parameters for the subagent
- `createdAt` (timestamp): Creation timestamp
- `updatedAt` (timestamp): Last update timestamp

**Validation Rules**:
- Name must be unique
- Type must be one of: "personalization", "translation", "content-summary", "other"
- Parameters must be valid JSON object

**Relationships**:
- Used by various services for processing tasks

## Entity: User Session
**Description**: Authentication session data (managed by Better Auth)

**Fields**:
- `id` (string): Unique session identifier
- `userId` (string): Reference to User Profile
- `token` (string): Session token (encrypted)
- `createdAt` (timestamp): Session creation time
- `expiresAt` (timestamp): Session expiration time
- `lastAccessed` (timestamp): Last access timestamp

**Validation Rules**:
- Session must be valid and not expired
- Token must be properly encrypted
- User reference must exist

**Relationships**:
- Belongs to one User Profile
- Many-to-one with User Profile

## State Transitions

### User Profile
- **New User**: Created during signup with basic auth data
- **Profile Complete**: After background questions are answered
- **Profile Updated**: When user modifies their background information

### Chapter Content
- **Original State**: Content exists in English only
- **Personalized**: After personalization process for specific user
- **Translated**: After Urdu translation is generated
- **Both**: When both personalized and translated versions exist

### User Session
- **Active**: Valid session with authenticated user
- **Expired**: Session has passed expiration time
- **Revoked**: User explicitly logged out or session invalidated