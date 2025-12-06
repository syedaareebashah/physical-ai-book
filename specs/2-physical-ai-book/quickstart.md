# Quickstart Guide: Physical AI & Humanoid Robotics Docusaurus Book

**Feature**: Physical AI & Humanoid Robotics Docusaurus Book
**Created**: 2025-12-06
**Author**: Claude Code

## Initial Setup

### Prerequisites
- Node.js 18+ installed
- npm or yarn package manager
- Git for version control
- Text editor or IDE with TypeScript support

### Step 1: Initialize Docusaurus Project
```bash
npx create-docusaurus@latest physical-ai-book classic --typescript
cd physical-ai-book
```

### Step 2: Install Additional Dependencies
```bash
npm install @docusaurus/module-type-aliases @docusaurus/tsconfig @docusaurus/types
npm install --save @docusaurus/preset-classic @mdx-js/react
npm install --save-dev prism-react-renderer
```

### Step 3: Configure Project Structure
1. Create the directory structure in the `docs` folder:
```
docs/
├── intro.md
├── prerequisites.md
├── setup.md
├── module1-ros2/
│   ├── _category_.json
│   ├── 01-introduction.mdx
│   ├── 02-core-concepts.mdx
│   ├── 03-first-node.mdx
│   ├── 04-python-rclpy.mdx
│   ├── 05-urdf.mdx
│   ├── 06-project.mdx
│   └── 07-assessment.mdx
├── module2-simulation/
├── module3-isaac/
├── module4-vla/
└── appendices/
```

2. Create the `_category_.json` files for each module:
```json
{
  "label": "Module 1: The Robotic Nervous System (ROS 2)",
  "position": 2,
  "link": {
    "type": "generated-index",
    "description": "Learn ROS 2 fundamentals for robotics development."
  }
}
```

### Step 4: Configure Docusaurus
Update `docusaurus.config.ts` with the project settings:
- Site title: "Physical AI & Humanoid Robotics"
- Tagline: "Bridge the Gap Between Digital Intelligence and Physical Reality"
- URL: "https://panaversity.github.io"
- Base URL: "/physical-ai-book/"
- Organization and project names

### Step 5: Set Up Custom Styling
1. Update `src/css/custom.css` with the gradient theme matching ai-native.panaversity.org
2. Add CSS variables for the purple-to-violet gradient (#667eea to #764ba2)
3. Implement dark mode as default with light mode toggle

### Step 6: Create Custom Components
1. Create the custom components directory: `src/components/`
2. Implement HomepageFeatures, LearningPath, ComparisonTable, and ModuleCard components
3. Ensure all components are responsive and accessible

### Step 7: Enable Mermaid Diagrams
Add the Mermaid plugin to `docusaurus.config.ts`:
```javascript
plugins: [
  [
    require.resolve("@cmfcmf/docusaurus-monet-theme"),
    {
      // Mermaid configuration
    },
  ],
],
```

### Step 8: Start Development Server
```bash
npm run start
```

The site will be available at http://localhost:3000

## Next Steps
1. Implement the homepage with the specified design requirements
2. Create the first module content (ROS 2 fundamentals)
3. Add interactive features like code examples and quizzes
4. Implement the assessment system with 80% passing threshold
5. Test accessibility and performance targets