# Bernard - AI Workshop Assistant

**Bernard - Named after Bernard Lowe (Westworld awakening archetype)**

> An embodied AI that learns to recognize objects and understand spatial relationships through natural voice interaction, using a dual JEPA architecture for world modeling and associative memory.

*Note: "Eridos" is the company/brand name for the website, handles, and public presence. "Bernard" is the internal system name used in code.*

**Canonical path:** `C:\Users\Jason\Desktop\Eridos\Bernard`

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

## Implementation Patterns

### Adding New Modules

1. Create standalone file with clear dataclass definitions
2. Use `@dataclass` for all data structures
3. Include `to_dict()` / `from_dict()` for JSON serialization
4. Add integration points in `bernard_integrated_v2.py`

### Testing Approach

- Test components in isolation first (`test_*.py` files)
- Use real workshop data, not synthetic
- Visual verification matters — screenshots of what the system "sees"

### Code Style

- Type hints on all function signatures
- Docstrings explaining "why" not just "what"
- Keep modules focused — one responsibility each
- Prefer composition over inheritance

## Current Blocker

**Data accumulation**: The Inward JEPA trainer needs:
- 50+ episodes with embeddings
- 10+ unique focus objects

The architecture is ready. We're waiting for real workshop interaction to generate sufficient training data.

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

## Planned Components

### Continuous Listening (Next)

Natural voice interaction without wake words:
- Silero VAD → Whisper → Intent classification
- Intent categories: identify, correct, confirm, teach
- Temporal binding: links utterances to visual context (0-3 seconds prior)
- Smaller Whisper models for workshop GPU constraints (GTX 1060)

### Future: Semantic Audio JEPA

Replace speech-to-text with direct audio-to-meaning associations. Eliminates text bottleneck, enables natural accent learning. Requires same JEPA architecture — it's modality-agnostic.

## Hardware Context

- **Office PC** (training): i9-14900K, RTX 4080 Super, 64GB RAM
- **Workshop PC** (capture): 2018 HP Omen, GTX 1060

Code must run on the workshop PC for capture. Heavy training happens on office PC.

## Common Pitfalls

1. **Don't** create universal associations from background objects
2. **Don't** add temporal special-casing — time is a dimension
3. **Don't** skip the `is_focus` filter in training
4. **Don't** assume Florence categories are stable — same object may get different categories
5. **Do** use cross-category misclassification learning for recovery

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

