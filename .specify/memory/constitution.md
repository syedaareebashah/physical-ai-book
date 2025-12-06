<!-- Sync Impact Report:
Version Change: 7.0.0 -> 8.0.0
Modified Principles: Added Design Principles, Technical Stack, Deployment, and Quality Requirements sections
Added Sections: Design Principles, Content Structure (enhanced), Technical Stack, Deployment, Quality Requirements
Templates requiring updates:
- .specify/templates/plan-template.md: ⚠ pending
- .specify/templates/spec-template.md: ⚠ pending
- .specify/templates/tasks-template.md: ⚠ pending
- .specify/templates/commands/sp.adr.md: ⚠ pending
- .specify/templates/commands/sp.analyze.md: ⚠ pending
- .specify/templates/commands/sp.checklist.md: ⚠ pending
- .specify/templates/commands/sp.clarify.md: ⚠ pending
- .specify/templates/commands/sp.git.commit_pr.md: ⚠ pending
- .specify/templates/commands/sp.implement.md: ⚠ pending
- .specify/templates/commands/sp.phr.md: ⚠ pending
- .specify/templates/commands/sp.plan.md: ⚠ pending
- .specify/templates/commands/sp.specify.md: ⚠ pending
- .specify/templates/commands/sp.tasks.md: ⚠ pending
Follow-up TODOs: None
-->
# Physical AI & Humanoid Robotics Textbook Constitution

**Version:** 8.0.0 (MAJOR — Addition of design, technical stack, deployment, and quality requirements)

**Ratified:** 2025-01-17

**Last Amended:** 2025-12-06

**Scope:** Educational content governance for the Physical AI & Humanoid Robotics textbook.

**Audience:** Textbook authors, content creators, AI Agents involved in content generation, general public.

---

## Educational Principles

### Beginner-Friendly Yet Technically Rigorous
Content MUST be accessible to beginners while maintaining a high level of technical accuracy and depth. Concepts SHOULD be introduced with foundational knowledge before progressing to advanced topics.
Rationale: Ensures a broad audience can engage with the material and build a strong, accurate understanding.

### Real-World Analogies First
Each new technical concept MUST be introduced with clear, relatable real-world analogies before delving into specific technical details and code.
Rationale: Facilitates intuitive understanding and bridges the gap between abstract concepts and practical applications.

### Progressive Learning
The curriculum MUST follow a clear progression from simple to intermediate to advanced topics. Each module and chapter SHOULD build upon previously introduced concepts.
Rationale: Supports a structured learning path, preventing cognitive overload and reinforcing knowledge incrementally.

### Tested and Runnable Code Examples
Every code example provided MUST be thoroughly tested, demonstrably runnable, and accompanied by instructions for execution.
Rationale: Guarantees reliability, enables hands-on learning, and reduces friction for students attempting to reproduce results.

### Hands-on Exercises
Hands-on exercises MUST follow each major concept explanation, providing practical application of the learned material. These exercises SHOULD be designed to reinforce understanding and build practical skills.
Rationale: Solidifies theoretical knowledge through practical application and develops problem-solving capabilities.

---

## Design Principles

### Match ai-native.panaversity.org Aesthetic
All website and interface design MUST match the ai-native.panaversity.org aesthetic exactly, with modern, clean, professional design featuring gradients.
Rationale: Ensures consistent branding and professional appearance aligned with the established standard.

### Fully Responsive Mobile-First Approach
All content and interfaces MUST be designed with a fully responsive mobile-first approach.
Rationale: Ensures accessibility and optimal experience across all device types and screen sizes.

### Dark Mode as Default
Dark mode MUST be the default theme with a light mode toggle available.
Rationale: Provides optimal readability and user preference accommodation while reducing eye strain.

### Smooth Animations and Transitions
All user interfaces MUST incorporate smooth animations and transitions for enhanced user experience.
Rationale: Creates a polished, professional feel and improves user engagement.

### WCAG 2.1 AA Compliance
All content and interfaces MUST be accessible and compliant with WCAG 2.1 AA standards.
Rationale: Ensures inclusive access for users with disabilities and meets accessibility best practices.

---

## Technical Standards

### Python with rclpy for ROS 2
All ROS 2 related code examples and implementations MUST exclusively use Python with the `rclpy` client library.
Rationale: Ensures consistency across the textbook and focuses on a widely adopted and accessible programming language for robotics.

### PEP 8 Style Guide Adherence
All Python code, including examples and exercise solutions, MUST strictly follow the PEP 8 style guide for formatting and conventions.
Rationale: Promotes readability, maintainability, and consistency in the codebase.

### Consistent Naming Conventions
Naming conventions for variables, functions, classes, and files MUST be consistent across all modules and chapters of the textbook.
Rationale: Improves code clarity, reduces cognitive load for learners, and fosters good programming practices.

### Comprehensive Code Comments
Code examples MUST include comprehensive, clear, and concise comments explaining non-obvious logic, complex sections, and design choices.
Rationale: Enhances understanding for learners and provides insight into the rationale behind implementations.

### Gazebo and Isaac Sim for Simulations
All simulations and simulated environments MUST be tested and demonstrated using both Gazebo and NVIDIA Isaac Sim where applicable.
Rationale: Exposes students to industry-standard simulation platforms and allows for comparison of their features and workflows.

---

## Content Structure

### Module-Based Learning
Each module MUST follow a structured sequence: Theory → Implementation → Exercise → Challenge.
Rationale: Provides a comprehensive learning cycle, moving from foundational understanding to practical application, reinforcement, and advanced problem-solving.

### Visual Learning Emphasis
Architectural concepts, data flows, and complex system interactions MUST be explained with clear, high-quality diagrams and visual aids.
Rationale: Enhances comprehension, especially for complex robotics concepts, and caters to diverse learning styles.

### Consistent Formatting
The textbook MUST maintain consistent formatting throughout, including dedicated sections for concept boxes, clearly delineated code blocks, and prominent warning/caution notes.
Rationale: Improves readability, guides student attention, and creates a professional, cohesive learning experience.

### Cross-Referencing
Relevant concepts and sections MUST be cross-referenced throughout the textbook to highlight connections and facilitate deeper understanding.
Rationale: Reinforces interdependencies between topics and enables students to navigate related information easily.

### Technical Glossary
A comprehensive glossary of all technical terms and acronyms used in the textbook MUST be provided.
Rationale: Serves as a quick reference for students and ensures a shared understanding of specialized vocabulary.

### Progressive Learning: Theory → Practice → Project
Each module and chapter MUST follow the progressive learning structure: Theory → Practice → Project.
Rationale: Ensures students understand concepts before applying them, then consolidate learning through comprehensive projects.

### Chapter Structure: Objectives, Content, Exercises, Summary
Each chapter MUST follow the structure: Learning Objectives → Content → Exercises → Summary.
Rationale: Provides a consistent learning experience with clear expectations and reinforcement.

### Sidebar Navigation with Progress Tracking
All content MUST include sidebar navigation with progress tracking capabilities.
Rationale: Enables students to navigate efficiently and track their learning progress.

---

## Technical Stack

### Docusaurus v3 + TypeScript Foundation
All documentation and content MUST be built with Docusaurus v3.5+ and TypeScript.
Rationale: Provides a modern, extensible platform with strong type safety and documentation capabilities.

### MDX for Interactive Content
All interactive content MUST be implemented using MDX.
Rationale: Enables rich, interactive content with React components embedded in Markdown.

### React 18+ for Custom Components
All custom components MUST be built with React 18+.
Rationale: Leverages modern React features and ensures compatibility with the latest ecosystem.

### Tailwind CSS for Styling
All styling MUST be implemented using Tailwind CSS.
Rationale: Provides utility-first CSS framework for rapid, consistent styling.

### Mermaid.js for Diagrams
All technical diagrams MUST be created using Mermaid.js.
Rationale: Enables programmatic creation of diagrams with version control and consistency.

### Algolia DocSearch for Search
All documentation MUST include Algolia DocSearch for search functionality.
Rationale: Provides fast, accurate search across the entire documentation set.

---

## Deployment

### GitHub Pages Deployment
The documentation website MUST be deployed using GitHub Pages.
Rationale: Provides reliable, free hosting with seamless integration with GitHub workflows.

### Automatic Builds on Main Branch Push
Automatic builds MUST be triggered on main branch pushes.
Rationale: Ensures content is always up-to-date without manual intervention.

### Preview Deployments for PRs
Preview deployments MUST be available for all pull requests.
Rationale: Enables review of changes in a live environment before merging.

### Custom Domain Support
The deployment MUST support custom domain configuration.
Rationale: Allows for professional branding and URL management.

### Performance Optimization (Lighthouse > 90)
All deployments MUST achieve Lighthouse performance scores above 90.
Rationale: Ensures fast loading times and optimal user experience.

---

## Quality Requirements

### All Links Validated
All links in the documentation MUST be validated and functional.
Rationale: Ensures users can access all referenced resources without encountering broken links.

### Mobile Responsive Testing
All content MUST pass mobile responsive testing across common device sizes.
Rationale: Ensures optimal experience across all devices and screen sizes.

### Cross-Browser Compatibility
All content MUST be compatible across major browsers (Chrome, Firefox, Safari, Edge).
Rationale: Ensures broad accessibility and consistent experience across different browsers.

### SEO Optimization
All content MUST be optimized for search engine visibility.
Rationale: Increases discoverability and reach of the educational content.

### Fast Loading Times
All pages MUST load within 3 seconds on average mobile connections.
Rationale: Ensures optimal user experience and reduces bounce rates.

### Technical Accuracy Verification
All technical information, code implementations, and experimental results MUST be verified for accuracy through rigorous testing, simulation, or expert review.
Rationale: Guarantees the reliability of the educational content and builds trust with the learners.

### Clear Learning Objectives
Each chapter and major section MUST have explicitly stated, measurable learning objectives.
Rationale: Provides students with clear expectations for what they will learn and allows them to track their progress.

### Explicit Prerequisites
Prerequisites for each module, chapter, or major section MUST be clearly and explicitly stated.
Rationale: Helps students assess their readiness for new material and guides them to foundational content if needed.

### Estimated Exercise Time
An estimated time for completion MUST be provided for each hands-on exercise and challenge.
Rationale: Assists students in managing their study time and provides a realistic expectation of effort required.

### Links to Official Documentation
Relevant sections MUST include direct links to official documentation for tools, libraries, and external resources.
Rationale: Encourages students to explore primary sources, provides avenues for deeper learning, and ensures access to the most up-to-date information.