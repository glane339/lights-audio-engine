# Multi-LLM Development Workflow

This is the authoritative policy for ChatGPT, Codex, Claude Code, and Cursor in this repository. The repository lifecycle is:

`PLAN -> IMPLEMENT -> DETERMINISTIC VALIDATION -> INDEPENDENT REVIEW -> REPAIR IF REQUIRED -> FINAL VERIFICATION -> MERGE`

The objective is to use the cheapest capable model, validate deterministically, escalate only when evidence justifies it, and use cross-vendor review for consequential work.

## Responsibilities and routing

- **ChatGPT** coordinates workflow, research, roadmap decisions, decomposition, prompt construction, and model/tool/effort routing.
- **Claude Code** is primarily the planning and independent-review agent. Sonnet 5 Medium handles normal planning, repository analysis, code review, and documentation synthesis. Opus 5 High handles complex architecture, difficult audits, subtle correctness review, and consequential pre-merge verification. Planning tasks do not edit production code unless explicitly authorized.
- **Codex** is primarily the implementation and debugging agent. GPT-5.6 Luna Low handles mechanical edits, searches, repetitive changes, and housekeeping. GPT-5.6 Terra Medium is the default for implementation, normal bug fixes, tests, and focused refactors. GPT-5.6 Sol High handles difficult debugging, architectural implementation, large cross-module changes, and repeated failures.
- **Cursor** is the interactive repository cockpit for rapid editing and navigation. It should prefer small, reviewable changes and should not duplicate an independent audit already assigned to Claude.

Do not select a flagship model merely because a prompt is long. Prefer different model families for consequential review: OpenAI implementation -> Claude review; Claude architecture -> OpenAI adversarial review. Do not ask several frontier models to perform the same job by default.

## Planning and implementation

Before editing, inspect the repository, relevant instructions, current status, and affected files. Preserve useful existing guidance. Keep changes minimal and within the approved scope. Report contradictions rather than silently redesigning requirements.

Implementation prompts should contain:

- **GOAL**
- **SCOPE**
- **AUTHORITATIVE CONTEXT**
- **ACCEPTANCE CRITERIA**
- **CONSTRAINTS / NON-GOALS**
- **REQUIRED VALIDATION**
- **REQUIRED OUTPUT / HANDOFF**

## Deterministic validation

Run applicable checks before expensive AI review, in this order:

1. formatting and diff checks
2. lint
3. static typing
4. focused tests
5. full tests
6. integration or runtime checks when applicable

Never claim validation passed unless the successful command output was observed. The authoritative definition of mandatory repository CI checks is [.github/workflows/ci.yml](../../.github/workflows/ci.yml); an AI PASS never overrides failing mandatory CI.

## Escalation

Escalate capability when a deterministic check fails for a non-obvious reason, the same repair fails twice, requirements or architecture become materially ambiguous, the change crosses important module boundaries, or security, privacy, concurrency, timing, destructive behavior, or data integrity is involved. Independent review is justified by consequence, not by prompt length.

## Context handoffs

Every handoff should concisely provide the branch, objective, approved plan, relevant files, architectural decisions and invariants, changes performed, validation results, current failures, and unresolved questions. Do not paste an entire prior AI conversation when a compact handoff is sufficient.

## Audits and review

Audit findings must include severity, location, evidence, consequence, and the smallest corrective action. Reviewers must explicitly return **PASS** when no material defect exists and must not invent findings merely to appear thorough.

## Merge gate

A branch is merge-ready only when the intended scope is complete, required deterministic checks pass, no blocker or high-severity review finding remains unresolved, no unintended files are present, required documentation/configuration updates exist, and the final diff has been inspected.
