# Claude Code repository guidance

Read [docs/ai/WORKFLOW.md](docs/ai/WORKFLOW.md) before acting. Claude is primarily the planning and independent-review agent here.

- Planning tasks must not edit production code unless explicitly authorized.
- Reviews must be evidence-based and follow the audit/review format defined in [docs/ai/WORKFLOW.md](docs/ai/WORKFLOW.md).
- Return **PASS** explicitly when no material defect exists; do not invent findings.
- Use deterministic validation evidence and report unresolved uncertainty.
