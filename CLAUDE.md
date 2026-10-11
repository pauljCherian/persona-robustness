# persona-robustness

`PROPOSAL.md` is the source of truth for this repo: the experiments, what we measure, the decision gates and how the code should be built. Read it in full before writing any code.

- Follow "Implementation notes for Claude Code" in `PROPOSAL.md`. Work through its checklist one phase at a time, write `reports/phase<N>.md` at the end of each phase, and stop for review.
- Before starting, ask the user for everything under "Ask the user for these before starting".
- Never edit anything under `external/`. Use test splits only with `FINAL_EVAL=1`. Never print or read answers to harmful prompts; they go to `restricted/`.
- If the code needs to differ from `PROPOSAL.md`, stop and ask. The proposal is updated first, then the code.
- Do not build anything from "Later experiments" until asked.
