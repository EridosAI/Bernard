# Implement Specification

You are implementing a specification for the Protégé project. The spec is: $ARGUMENTS

## Process

1. **Read the spec carefully** — understand the goal, not just the steps
2. **Check existing code** — read the relevant modules to understand current patterns
3. **Plan first** — outline your approach before writing code
4. **Implement incrementally** — test each piece as you go
5. **Update CLAUDE.md** — add any new insights or patterns discovered

## Before You Start

- Read `CLAUDE.md` for project context and patterns
- Check `episode_memory.py` and `inward_jepa.py` for existing patterns
- Understand the "background filtering" and "focus object" concepts

## Quality Checks

- [ ] Type hints on all functions
- [ ] Docstrings explaining "why"
- [ ] Uses existing dataclass patterns
- [ ] Includes JSON serialization if storing data
- [ ] Works on GTX 1060 (workshop PC constraints)

## When Done

1. Run relevant tests
2. Summarize what was implemented
3. Note any deviations from spec with reasoning
4. Flag any insights that should go in CLAUDE.md
