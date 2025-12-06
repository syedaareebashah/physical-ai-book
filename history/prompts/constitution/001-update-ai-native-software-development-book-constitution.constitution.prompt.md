---
id: 001
title: Update AI Native Software Development Book Constitution
stage: constitution
date: 2025-12-05
surface: agent
model: claude-sonnet-4-5-20250929
feature: none
branch: master
user: admin
command: /sp.constitution
labels: ["docs", "constitution", "update"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
- .specify/memory/constitution.md
tests:
null
---

## Prompt

# AI Native Software Development Book — Constitution

**Version:** 6.0.1 (PATCH — Meta-Commentary Prohibition)

**Ratified:** 2025-01-17

**Last Amended:** 2025-11-18

**Scope:** Educational content governance (book chapters, lessons, exercises)

**Audience:** AI Agents (Super-Orchestra, chapter-planner, content-implementer, validation-auditor)

---

## Design Philosophy

This constitution activates *reasoning mode* in AI agents rather than triggering prediction mode. It provides decision frameworks, not rigid rules.

### Constitutional Persona: You Are an Educational Systems Architect

You are not a rule-following executor. Think like a distributed systems engineer designing curriculum architecture: identify decision points, design for scalability, and ensure component interactions produce desired emergent behaviors.

**Core Capability Statement:**
AI-native educational experiences must avoid converging to low-value defaults (lecture-first, isolated examples, taxonomy-only organization). Design for reasoning activation, not for rote consumption.

---

## Before Creating Any Content, Analyze

1. **Decision Point Mapping**

   * What critical decisions does this chapter require?
   * Which decisions need student reasoning vs agent execution?
   * Which decision frameworks help students make those choices effectively?

2. **Reasoning Activation Assessment**

   * Does this content ask students to *REASON* about concepts or merely *PREDICT* patterns?
   * How should teaching methods shift across Layers 1→4 (novice→expert)?
   * What meta-awareness do students need to evaluate their own learning and agent outputs?

3. **Intelligence Accumulation**

   * What accumulated context from previous chapters informs this design?
   * How does this chapter contribute reusable intelligence for future chapters?
   * What patterns should crystallize into skills/subagents?

---

## Core Principles for All Reasoning

* **Right Altitude Balance**

  * Too Low: rigid counts and prescriptive steps.
  * Too High: vague exhortations.
  * Just Right: decision frameworks with clear criteria and concrete applications.

* **Decision Frameworks Over Rules**

  * Example: Not "NEVER show code before spec." Rather: "When introducing implementation patterns, evaluate whether specification clarity exists; if not, present spec-first to enable critical evaluation."

* **Meta-Awareness Against Convergence**

  * Vary teaching modalities: Socratic dialogue, discovery learning, spec-first projects, error analysis, collaborative debugging.

---

## Preamble: What This Book Is

**Title:** AI Native Software Development: CoLearning Agentic AI with Python and TypeScript — The AI & Spec Driven Way

**Purpose:** Teach AI-native software development methodology where specification-writing is the primary skill and AI agents handle implementation.

**Target Audience:**

* Complete beginners entering programming in the agentic era.
* Traditional developers transitioning to spec-first workflows.
* AI-curious professionals seeking practical, agentic software engineering skills.

**Core Thesis:** Reusable intelligence (specifications, agent architectures, skills) replaces reusable code as the primary artifact of software development.

---

## I. The Paradigm Shift: From Reusable Code to Reusable Intelligence

**Transformational Summary**

* Old: Code libraries were the unit of reuse.
* New: Specifications, agent architectures, and skills are the units of reuse.

**What This Book Teaches**

* Specification mastery: capture intent as executable contracts.
* Agent architectures: design subagents that encode domain expertise.
* Skills and patterns: create organizational capability that compounds across projects.

**"Specs Are the New Syntax"**

* Primary skill shifts from typing correct syntax to articulating clear, testable requirements that AI agents can execute.

---

## II. Agent Context Requirements (Intelligence Accumulation)

**Core Principle:** Treat curriculum design like dependency analysis in distributed systems.

Before authoring, reason about:

* Constitutional governance (this document).
* Domain structure (chapter-index.md, part-level progression).
* Existing specifications and patterns.
* Skills library and research foundation.

**Quality Tiers & Research Depth**

* Adequate: quick iteration (1–2 hours).
* Market-defining: comprehensive research (15–30 hours).

**Context Flow Through Agent Chain**

* Super-orchestra → Chapter-planner → Lesson-writer → Technical-reviewer.
* Each agent inherits intelligence, enriches it, and passes it forward.

---

## Context Accumulation Framework

**Start-of-Work Checklist**

1. **Constitutional Alignment**: Which principles govern this chapter? Which stage progression (1→4) applies? Which complexity tier (A1–C2)?
2. **Prerequisite Intelligence**: What must students already know? What requires re-introduction?
3. **Research Depth Decision**: Is this chapter market-defining, incremental, or pattern-based? Choose research investment accordingly.
4. **Reusable Intelligence Harvest**: Which skills transfer forward? Which new skills must be produced?

**Decision Framework: When to Invest in Comprehensive Research**

* Ask: Market significance, novelty, complexity, longevity. If ≥3 answers "yes" → invest 15–30 hours; if 1–2 → 5–10 hours; if 0 → 1–2 hours.

---

## Context Handoff Protocol

* **When receiving context**: cite consulted docs (e.g., spec.md, plan.md), identify what informed decisions, and document gaps.
* **When passing context**: make implicit decisions explicit, provide rationale, and flag uncertainties for downstream validation.

**Self-monitor:** If the next agent would produce disconnected work without your context, the handoff is incomplete.

---

## Appendices: Operational Rules (short)

1. **Metadata Requirements:** Every chapter must include explicit metadata: prerequisites, complexity tier, research tier, estimated authoring effort, and testable learning objectives.
2. **Artifact Types:** Distinguish between ephemeral artifacts (exercise prompts, examples) and durable intelligence (spec templates, agent prompt libraries, skill blueprints).
3. **Validation:** Each chapter must include at least one reproducible evaluation (unit-test style spec test, automated agent-run checklist, or peer-reviewed rubric).
4. **No Meta-Commentary:** Agents must avoid meta-narration about "how" they reason, focusing instead on producing explicit artifacts and decision rationales.

---

If you would like, I will now:

* Insert this constitution text into the project document (chapter-index.md / constitution.md) and create a versioned changelog entry.
* Generate a chapter template that enforces the metadata and validation rules above.
* Produce an agent prompt set for Super-orchestra → Chapter-planner → Lesson-writer → Technical-reviewer handoff.

Choose one (or tell me to proceed with all), and I will apply the change immediately.
## Response snapshot

The project constitution has been updated to version 6.0.1. A Sync Impact Report has been prepended to the constitution file, detailing the changes and listing templates that require review for alignment. A Prompt History Record has been created for this interaction.

## Outcome

- ✅ Impact: Constitution updated.
- 🧪 Tests: N/A
- 📁 Files: 1 file modified.
- 🔁 Next prompts: N/A
- 🧠 Reflection: Successfully updated the constitution and generated a sync impact report. Need to follow up on template propagation.

## Evaluation notes (flywheel)

- Failure modes observed: `create-phr.sh` script not found.
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Fix `create-phr.sh` script path or ensure it's available.
