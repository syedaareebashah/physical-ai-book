# API Contract: Authentication Service

## Base URL
`/api/auth`

## Endpoints

### POST /signup
**Description**: Register a new user with background information

**Request**:
```json
{
  "email": "user@example.com",
  "password": "securePassword123",
  "softwareExperience": "intermediate",
  "programmingLanguages": ["JavaScript", "Python", "Java"],
  "devExperienceYears": 5,
  "hardwareSpecs": {
    "deviceType": "laptop",
    "os": "Windows 10",
    "cpu": "Intel i7",
    "gpu": "NVIDIA RTX 3070",
    "ram": 16
  },
  "developmentFocus": ["web", "AI/ML"]
}
```

**Response (201 Created)**:
```json
{
  "user": {
    "id": "user_12345",
    "email": "user@example.com",
    "createdAt": "2025-12-19T10:00:00Z"
  },
  "session": {
    "id": "session_67890",
    "token": "encrypted_session_token",
    "expiresAt": "2025-12-20T10:00:00Z"
  }
}
```

**Errors**:
- 400: Invalid input data
- 409: Email already exists

### POST /login
**Description**: Authenticate user and create session

**Request**:
```json
{
  "email": "user@example.com",
  "password": "securePassword123"
}
```

**Response (200 OK)**:
```json
{
  "user": {
    "id": "user_12345",
    "email": "user@example.com"
  },
  "session": {
    "id": "session_67890",
    "token": "encrypted_session_token",
    "expiresAt": "2025-12-20T10:00:00Z"
  }
}
```

**Errors**:
- 400: Invalid credentials
- 401: Authentication failed

### POST /logout
**Description**: End user session

**Headers**:
```
Authorization: Bearer {session_token}
```

**Response (200 OK)**:
```json
{
  "success": true
}
```

**Errors**:
- 401: Invalid or expired session

### GET /me
**Description**: Get current user profile

**Headers**:
```
Authorization: Bearer {session_token}
```

**Response (200 OK)**:
```json
{
  "id": "user_12345",
  "email": "user@example.com",
  "softwareExperience": "intermediate",
  "programmingLanguages": ["JavaScript", "Python", "Java"],
  "devExperienceYears": 5,
  "hardwareSpecs": {
    "deviceType": "laptop",
    "os": "Windows 10",
    "cpu": "Intel i7",
    "gpu": "NVIDIA RTX 3070",
    "ram": 16
  },
  "developmentFocus": ["web", "AI/ML"],
  "createdAt": "2025-12-19T10:00:00Z",
  "updatedAt": "2025-12-19T10:00:00Z"
}
```

**Errors**:
- 401: Invalid or expired session