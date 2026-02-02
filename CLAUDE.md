# CLAUDE.md — Bernard AI Workshop Assistant

**Bernard — Named after Bernard Lowe (Westworld awakening archetype)**

> An embodied AI that learns to recognize objects and understand spatial relationships through natural voice interaction, using interconnected specialist JEPA models (cortices) coordinated by an Internal JEPA for cross-modal integration.

*Note: "Eridos" is the company/brand name for the website, handles, and public presence. "Bernard" is the internal system name used in code.*

**Owner:** Jason (solo developer, based in Perth, Australia)
**Canonical path:** `C:\Users\Jason\Desktop\Eridos\Bernard`
**Philosophy:** "Raised not trained"

---

## Quick Reference

```bash
# Activate environment
conda activate bernard

# Run main system
python bernard_integrated_v2.py

# Run dream training (overnight)
python dream_training.py

# Test individual components
python test_camera.py
python test_florence2.py
python test_mic_whisper.py
```

---

## Project Structure

```
bernard/
├── config/              # YAML configuration files
│   ├── default.yaml     # Base config (all paths, hyperparameters)
│   ├── office_pc.yaml   # Machine-specific overrides
│   └── workshop_pc.yaml
│
├── docs/
│   ├── IDEAS.md         # Ideas log (may be symlinked to Obsidian)
│   ├── architecture.md  # High-level design documentation
│   └── specs/           # Implementation specs
│
├── src/
│   ├── config.py        # Config loader
│   ├── core/            # STABLE — validated, working components
│   ├── cortical/        # EXPERIMENTAL — new cortical architecture
│   ├── integration/     # Full system compositions
│   └── experimental/    # Unvalidated experiments
│
├── scripts/             # Runnable scripts (training, capture, etc.)
├── tests/               # Test files
├── models/              # Git-ignored, model checkpoints
├── data/                # Git-ignored, training data and captures
└── logs/                # Git-ignored, runtime logs
```

### Directory Rules

| Directory | Status | Rules |
|-----------|--------|-------|
| `src/core/` | STABLE | Only modify to fix bugs or improve performance. Changes should not break existing functionality. |
| `src/cortical/` | EXPERIMENTAL | New cortical architecture. Can change freely. |
| `src/integration/` | MIXED | `bernard_v2.py` is stable. New integrations are experimental. |
| `src/experimental/` | EXPERIMENTAL | Anything goes. Code here may be broken or incomplete. |
| `scripts/` | STABLE | Should always be runnable. Update when APIs change. |
| `config/` | STABLE | Changes here affect the whole system. Be careful. |

### Key Modules

| File | Purpose | Status |
|------|---------|--------|
| `bernard_integrated_v2.py` | Main orchestration, capture loop, voice interface | ✅ Working |
| `episode_memory.py` | Stores episodes with objects, spatial info, timestamps | ✅ Working |
| `inward_jepa.py` | Associative memory predictor network | ✅ Implemented, needs data |
| `dream_training.py` | Overnight LoRA fine-tuning | ✅ Phase 3 integrated |
| `novelty_scorer.py` | Curiosity-based attention (scene + object + position) | ✅ Working |

### Data Files

- `episode_memory.json` — Stored episodes with embeddings
- `object_memory.json` — Known objects and their embeddings

---

## Architecture Overview

### Core Principle: "As Within, So Without"

The system uses JEPA-style architecture for BOTH:
- **Outward JEPA**: Learning world structure from perception
- **Inward JEPA**: Learning memory-to-memory associations

The insight: internal experience of memory is structurally similar to experience of the 3D world. The same architecture can model both.

### Two-Layer Vision System

1. **Florence-2** (general detection): Provides categorical knowledge ("this is a remote control")
2. **V-JEPA** (specific recognition): Learns personal specifics ("this is MY air conditioner remote")

Florence gives us the "what category" — V-JEPA learns "which specific one."

---

## Critical Design Decisions

### 1. Background Object Filtering

**IMPORTANT**: Persistent workshop elements (workbench, walls, shelves) appear in most episodes. Without filtering, they create meaningless universal associations.

**Solution**: Use `is_focus` flag to distinguish:
- Focus objects = things being actively interacted with
- Background = persistent context

Training the Inward JEPA uses ONLY focus objects for association learning. The `EpisodeMemory.sample_training_batch()` method filters to focus objects only before passing data to the trainer.

### 2. Time as Dimension, Not Axis

Memories on disk aren't ordered temporally. Time is just another learned feature like position or color. The associative map doesn't need special temporal machinery — it learns that action→consequence patterns have characteristic time-gaps the same way it learns objects have characteristic sizes.

### 3. Objects as Causal Bridges

Action-consequence links emerge from patterns of "action-involving-X followed by outcome-involving-X" appearing repeatedly. Objects present in both memories create the association; causation emerges from pattern.

### 4. Vocalization Strengthens Learning

Forcing commitment to a hypothesis BEFORE getting feedback makes the learning signal stronger. When the system says "Would this be explained by [the unprotected wire from Tuesday]?" and gets confirmation/correction, that's a much stronger signal than passive observation.

### 5. V-JEPA's Natural Spatial Representations

Don't impose artificial grid-based spatial frameworks. V-JEPA has inherent spatial representations — leverage them rather than adding external machinery.

### 6. EMA as Natural Forgetting

The exponential moving average mechanism in JEPA serves as a natural forgetting system. Older memories become misaligned unless actively rehearsed. This is a feature, not a bug.

---

## Training Architecture

### Dream Training (Phase 3)

Runs overnight on raw captured data:
1. Loads episodes from `episode_memory.json`
2. Extracts focus object crops
3. Fine-tunes V-JEPA with LoRA adapters
4. Updates object embeddings

### Inward JEPA Training

Uses contrastive loss:
- **Positive pairs**: Episodes sharing focus objects
- **Negative pairs**: Episodes with no object overlap
- **Loss**: Query embedding should be close to associated episode embeddings

### Current Blocker

**Data accumulation**: The Inward JEPA trainer needs:
- 50+ episodes with embeddings
- 10+ unique focus objects

The architecture is ready. We're waiting for real workshop interaction to generate sufficient training data.

---

## Planned Components

### Continuous Listening (Next)

Natural voice interaction without wake words:
- Silero VAD → Whisper → Intent classification
- Intent categories: identify, correct, confirm, teach
- Temporal binding: links utterances to visual context (0-3 seconds prior)
- Smaller Whisper models for workshop GPU constraints (GTX 1060)

### Future: Semantic Audio JEPA

Replace speech-to-text with direct audio-to-meaning associations. Eliminates text bottleneck, enables natural accent learning. Requires same JEPA architecture — it's modality-agnostic.

---

## Hardware Context

- **Office PC** (training): i9-14900K, RTX 4080 Super, 64GB RAM
- **Workshop PC** (capture): 2018 HP Omen, GTX 1060

Code must run on the workshop PC for capture. Heavy training happens on office PC.

---

## Configuration Management

### Config Files

All paths and hyperparameters live in `config/default.yaml`. Machine-specific overrides in `config/<machine>.yaml`.

```yaml
# config/default.yaml
vjepa_model_path: "./models/base/vjepa2"
vjepa_adapter_path: "./models/adapters/workshop_lora_latest"
beats_checkpoint: "./models/base/beats/BEATs_iter3_plus_AS2M.pt"

internal_jepa:
  d_internal: 512
  num_encoder_layers: 6
  num_predictor_layers: 3
  context_window: 16
  learning_rate: 0.0001
```

### Using Config in Code

```python
from src.config import load_config, get

load_config("office_pc")  # Call once at startup
model_path = get("vjepa_model_path")
```

### Adding New Config

1. Add default value to `config/default.yaml`
2. Add machine-specific override if needed
3. Access via `get("key_name")` in code
4. **Never hardcode paths in Python files**

---

## Git Commit Rules

### ALWAYS Do This Before Committing

1. **Check what's changed:** `git status` and `git diff`
2. **Run a smoke test** if touching `core/` or `integration/`: Make sure the system still starts
3. **Update config** if you added new paths or parameters
4. **Update CLAUDE.md** if you changed project structure or conventions

### Commit Message Format

```
<type>: <short description>

<optional body - what and why>
```

**Types:**
- `feat:` — New feature or capability
- `fix:` — Bug fix
- `refactor:` — Code change that doesn't add features or fix bugs
- `docs:` — Documentation only
- `config:` — Configuration changes
- `experiment:` — Experimental code (may not work)
- `wip:` — Work in progress (use sparingly, for end-of-session saves)

**Examples:**
```
feat: add CortexBus for inter-cortex communication

fix: resolve tensor device mismatch in visual cortex

refactor: extract config loading into separate module

docs: update architecture.md with cortical design

experiment: testing alternative masking strategy for IJEPA

wip: partial implementation of audio cortex
```

### Commit Granularity

- **One logical change per commit.** Don't bundle unrelated changes.
- **Commit working states.** If something is broken, use `wip:` prefix.
- **Small commits are better** than large ones. Easier to understand and revert.

### Refactoring and Restructuring

When moving files, restructuring directories, or doing major refactors:

1. **Commit in logical chunks, not one giant commit.** Each chunk should be a coherent unit:
   - "refactor: move core modules to src/core/"
   - "refactor: move cortical modules to src/cortical/"
   - "refactor: update imports for new structure"
   - "fix: resolve import errors after restructure"

2. **Test after each chunk if possible.** At minimum, check that Python can import the moved modules.

3. **If a chunk breaks something,** it's easy to identify which move caused it and revert just that commit.

4. **Never combine restructuring with feature work** in the same commit. Move files first, get it working, then add features.

### What NOT to Commit

Never commit these (they should be in .gitignore):
- `models/` — Model checkpoints (too large)
- `data/` — Training data and captures (too large)
- `logs/` — Runtime logs
- `__pycache__/` — Python cache
- `.env` — Environment variables / secrets
- `*.pyc` — Compiled Python
- `.DS_Store` — macOS junk

### Branching

- **`main`** — Always runnable. The "it works" branch.
- **Feature branches** — For significant new work: `cortical-v1`, `audio-jepa`, etc.

**When to branch:**
- Changes that might take multiple sessions
- Changes that might break the main system
- Experiments you might want to abandon

**Branch workflow:**
```bash
git checkout -b cortical-v1      # Create and switch to new branch
# ... do work, make commits ...
git checkout main                 # Switch back to main
git merge cortical-v1            # Merge when ready
git branch -d cortical-v1        # Delete branch after merge
```

### Tagging Milestones

When something significant works, tag it:
```bash
git tag -a v0.3.0-cortical-working -m "First working cortical integration"
git push origin v0.3.0-cortical-working
```

**Tag format:** `v<major>.<minor>.<patch>-<description>`

Save model checkpoints with matching names: `models/cortical/ijepa_v0.3.0-cortical-working/`

---

## When Asked to Commit

When Jason asks to commit changes, follow this sequence:

1. **Show status:** `git status`
2. **Show diff summary:** `git diff --stat`
3. **Ask for confirmation** if changes touch `core/` or seem significant
4. **Stage relevant files:** `git add <files>` (not `git add .` unless certain)
5. **Commit with proper message:** `git commit -m "<type>: <description>"`
6. **Push if requested:** `git push origin <branch>`
7. **Remind about IDEAS.md** if architectural decisions were made

---

## Code Conventions

### Import Order

```python
# Standard library first
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional

# Third-party
import torch
import numpy as np

# Local imports — use relative within src/
from src.config import get
from src.core.vjepa_encoder import VJEPAEncoder
from src.cortical.interfaces import CortexOutput
```

### Naming

- Python files: `snake_case.py`
- Classes: `PascalCase`
- Functions/variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Config keys: `snake_case`

### Style

- Type hints on all function signatures
- Docstrings explaining "why" not just "what"
- Keep modules focused — one responsibility each
- Prefer composition over inheritance

### Adding New Modules

1. Create standalone file with clear dataclass definitions
2. Use `@dataclass` for all data structures
3. Include `to_dict()` / `from_dict()` for JSON serialization
4. Add integration points in `bernard_integrated_v2.py`

### Testing Approach

- Test components in isolation first (`test_*.py` files)
- Use real workshop data, not synthetic
- Visual verification matters — screenshots of what the system "sees"

---

## Common Pitfalls

- **Don't** create universal associations from background objects
- **Don't** add temporal special-casing — time is a dimension
- **Don't** skip the `is_focus` filter in training
- **Don't** assume Florence categories are stable — same object may get different categories
- **Do** use cross-category misclassification learning for recovery
- **Device mismatches:** Always check `.to(device)` when mixing CPU/CUDA tensors
- **Import paths:** Run from project root, imports start with `src.`
- **Config not loaded:** Call `load_config()` before `get()`
- **Relative paths:** Use `Path(__file__).parent` for paths relative to a module

---

## Development Workflow

This project uses a hybrid approach:
- **Opus** (claude.ai): Architectural planning, design decisions
- **Claude Code / Cline**: Implementation

When implementing from a spec:
1. Read the relevant module files first
2. Understand the existing patterns
3. Implement incrementally, testing each piece
4. Update this CLAUDE.md with any new insights

---

## Insight Inbox

*Quick-capture for ideas during the day. Review and integrate weekly.*

<!-- Add timestamped insights here, then move to main docs or project-ideas-log.md -->
