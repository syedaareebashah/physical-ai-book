# API Contract: Personalization Service

## Base URL
`/api/personalization`

## Endpoints

### POST /process
**Description**: Personalize chapter content based on user background

**Headers**:
```
Authorization: Bearer {session_token}
```

**Request**:
```json
{
  "chapterId": "chapter_123",
  "content": "Original chapter content to be personalized...",
  "userId": "user_12345"
}
```

**Response (200 OK)**:
```json
{
  "chapterId": "chapter_123",
  "originalContent": "Original chapter content to be personalized...",
  "personalizedContent": "Personalized chapter content based on user background...",
  "personalizationMetadata": {
    "userExperienceLevel": "intermediate",
    "appliedChanges": [
      "Simplified complex concepts",
      "Added relevant code examples",
      "Adjusted technical terminology"
    ],
    "processingTime": 1250
  }
}
```

**Errors**:
- 401: Invalid or expired session
- 404: Chapter not found
- 422: Unable to process personalization

### GET /preferences
**Description**: Get user's personalization preferences

**Headers**:
```
Authorization: Bearer {session_token}
```

**Response (200 OK)**:
```json
{
  "userId": "user_12345",
  "preferences": {
    "defaultPersonalization": true,
    "contentComplexity": "intermediate",
    "preferredLanguages": ["en", "ur"],
    "lastPersonalizedChapters": ["chapter_123", "chapter_456"]
  }
}
```

**Errors**:
- 401: Invalid or expired session

### PUT /preferences
**Description**: Update user's personalization preferences

**Headers**:
```
Authorization: Bearer {session_token}
```

**Request**:
```json
{
  "preferences": {
    "defaultPersonalization": true,
    "contentComplexity": "advanced",
    "preferredLanguages": ["en", "ur"]
  }
}
```

**Response (200 OK)**:
```json
{
  "success": true,
  "preferences": {
    "defaultPersonalization": true,
    "contentComplexity": "advanced",
    "preferredLanguages": ["en", "ur"]
  }
}
```

**Errors**:
- 401: Invalid or expired session
- 400: Invalid preferences format