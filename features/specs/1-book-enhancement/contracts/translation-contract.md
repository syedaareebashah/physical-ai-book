# API Contract: Translation Service

## Base URL
`/api/translation`

## Endpoints

### POST /urdu
**Description**: Translate chapter content to Urdu

**Headers**:
```
Authorization: Bearer {session_token}
```

**Request**:
```json
{
  "chapterId": "chapter_123",
  "content": "English chapter content to be translated...",
  "preserveFormatting": true
}
```

**Response (200 OK)**:
```json
{
  "chapterId": "chapter_123",
  "originalContent": "English chapter content to be translated...",
  "urduContent": "مترجم شدہ چیپٹر کا مواد...",
  "translationMetadata": {
    "accuracy": 0.92,
    "translatedWords": 1250,
    "processingTime": 2100,
    "fallbackParts": ["code snippets"],
    "rtlSupport": true
  }
}
```

**Errors**:
- 401: Invalid or expired session
- 404: Chapter not found
- 422: Unable to process translation

### GET /status
**Description**: Check translation service status

**Response (200 OK)**:
```json
{
  "status": "available",
  "supportedLanguages": ["en", "ur"],
  "serviceHealth": "operational",
  "lastUpdate": "2025-12-19T10:00:00Z"
}
```

**Errors**:
- 503: Service unavailable

### POST /validate
**Description**: Validate content for translation compatibility

**Headers**:
```
Authorization: Bearer {session_token}
```

**Request**:
```json
{
  "content": "Content to validate for translation..."
}
```

**Response (200 OK)**:
```json
{
  "isValid": true,
  "issues": [],
  "suggestions": [],
  "estimatedProcessingTime": 1500
}
```

**Response (200 OK) - with issues**:
```json
{
  "isValid": false,
  "issues": [
    {
      "type": "untranslatable",
      "position": 150,
      "text": "Code snippet",
      "suggestion": "Preserve in original language"
    }
  ],
  "suggestions": ["Consider adding technical term glossary"],
  "estimatedProcessingTime": 2000
}
```

**Errors**:
- 401: Invalid or expired session
- 400: Invalid content format