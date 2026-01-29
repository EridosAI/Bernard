# NAMING_INSTRUCTIONS.md

## Project Naming Convention — Eridos + Bernard

This document provides authoritative naming guidance for Claude Code and all development work.

---

## Quick Reference

| Element | Name | Usage |
|---------|------|-------|
| **Brand/Company** | Eridos | Public-facing: domain, social, marketing, documentation headers |
| **System/AI** | Bernard | Code: classes, modules, variables, internal references |
| **Logo** | 𒉣𒆠 | Visual branding (cuneiform NUN.KI) |

---

## Brand Name: Eridos

**Pronunciation:** eh-REE-dos (3 syllables)

**Origin:** Greek-ified form of "Eridu" — the first city in Sumerian mythology, where gods shaped humans from clay. The cuneiform logogram for Eridu is NUN.KI (𒉣𒆠), meaning "The Mighty Place."

**Use Eridos for:**
- Domain: eridos.ai
- GitHub organization: github.com/eridos
- Social handles: @eridos
- Public documentation titles
- README headers
- Website copy
- Marketing materials
- Company/venture references
- External communications

**Examples:**
```
✅ "Eridos is building Bernard, an AI workshop assistant..."
✅ "Welcome to Eridos — where we're raising AI, not training it."
✅ "© 2026 Eridos"
✅ eridos.ai
```

---

## System Name: Bernard

**Origin:** Named after Bernard Lowe from Westworld — the AI character who awakens to his true nature. Represents the "Gnostic Awakening" archetype.

**Use Bernard for:**
- Python package/module names
- Class names
- Variable names referencing the system
- Internal documentation
- Code comments
- Config files
- Log prefixes
- CLI tool names

**Examples:**
```python
# ✅ Correct
from bernard.vision import ObjectRecognizer
from bernard.memory import EpisodeMemory
from bernard.jepa import InwardJEPA

class BernardCore:
    """Main orchestrator for the Bernard system."""
    pass

# Config
BERNARD_CONFIG_PATH = "~/.bernard/config.yaml"

# Logging
logger = logging.getLogger("bernard.vision")
```

```bash
# ✅ Correct CLI naming
bernard-capture     # Vision capture daemon
bernard-train       # Training pipeline
bernard-listen      # Audio listener
```

---

## Logo Element: 𒉣𒆠 (Cuneiform NUN.KI)

**Characters:**
- 𒉣 (NUN) — Unicode U+12263 — "prince," "mighty," "lofty"
- 𒆠 (KI) — Unicode U+121A0 — "earth," "place"

**Meaning:** "The Mighty Place" / "The Place of Creation"

**Usage:**
- Visual logo/brandmark
- Favicon (stylized)
- README badges
- Documentation headers (sparingly)

**Do NOT use in:**
- Code (not ASCII-safe)
- File names
- URLs
- Config keys

---

## Naming Migration Checklist

When renaming from previous names (Protégé, Jarvis, Arnold, Nascor), follow this checklist:

### Codebase Renames

```bash
# Find all references to old names
grep -r "protege\|Protege\|PROTEGE" --include="*.py" --include="*.md" --include="*.yaml"
grep -r "jarvis\|Jarvis\|JARVIS" --include="*.py" --include="*.md" --include="*.yaml"
grep -r "arnold\|Arnold\|ARNOLD" --include="*.py" --include="*.md" --include="*.yaml"
grep -r "nascor\|Nascor\|NASCOR" --include="*.py" --include="*.md" --include="*.yaml"
```

### Replacement Rules

| Old Pattern | New Pattern | Context |
|-------------|-------------|---------|
| `protege` | `bernard` | Python packages, modules |
| `Protege` | `Bernard` | Class names, titles |
| `PROTEGE` | `BERNARD` | Constants, env vars |
| `jarvis` | `bernard` | Python packages, modules |
| `Jarvis` | `Bernard` | Class names, titles |
| `arnold` | `bernard` | Python packages, modules |
| `Arnold` | `Bernard` | Class names, titles |
| `nascor` | `eridos` | Brand references only |
| `Nascor` | `Eridos` | Brand references only |

### Files to Update

1. **CLAUDE.md** — Update project description, naming references
2. **README.md** — Header, description, badges
3. **pyproject.toml / setup.py** — Package name
4. **src/ directory** — Rename `protege/` or `jarvis/` to `bernard/`
5. **Import statements** — All `from protege...` to `from bernard...`
6. **Config files** — `.yaml`, `.json`, `.env`
7. **Documentation** — All `.md` files
8. **GitHub repo** — Eventually rename to `eridos/bernard`

### Git Commit Convention

```bash
# For the rename commit
git commit -m "chore: rename project to Eridos/Bernard

- Brand: Eridos (Greek-ified Eridu, the first city)
- System: Bernard (Westworld awakening archetype)
- Logo element: 𒉣𒆠 (cuneiform NUN.KI)

Replaces: Protégé, Jarvis, Arnold, Nascor"
```

---

## Contextual Usage Examples

### README.md Header
```markdown
# Bernard

**An AI workshop assistant that learns through lived experience.**

Bernard is the first project from [Eridos](https://eridos.ai) — exploring post-linguistic architecture, associative memory, and developmental AI.

𒉣𒆠
```

### CLAUDE.md Project Description
```markdown
# Bernard — Eridos Workshop Assistant

Bernard is an AI system being developed by Eridos that learns through lived experience 
rather than traditional training datasets. The core vision is a "post-linguistic 
architecture" where meaning exists in embedding space across all modalities.
```

### Python Package Structure
```
bernard/
├── __init__.py
├── core.py
├── vision/
│   ├── __init__.py
│   ├── capture.py
│   └── jepa.py
├── memory/
│   ├── __init__.py
│   ├── episode.py
│   └── associative.py
└── audio/
    ├── __init__.py
    └── listener.py
```

### Config File
```yaml
# bernard_config.yaml
bernard:
  version: "0.1.0"
  
  vision:
    capture_fps: 1
    model: "facebook/vjepa-v2"
    
  memory:
    episode_dir: "~/.bernard/episodes"
    ltm_path: "~/.bernard/ltm.index"
```

---

## Common Mistakes to Avoid

❌ **Don't mix brand and system names incorrectly:**
```
# Wrong
"Eridos is an AI assistant" (Eridos is the company, Bernard is the AI)
"Bernard Inc." (Bernard is the system, Eridos is the company)
```

✅ **Correct:**
```
"Eridos is building Bernard, an AI workshop assistant"
"Bernard, developed by Eridos, learns through lived experience"
```

❌ **Don't use cuneiform in code:**
```python
# Wrong
𒉣𒆠_CONFIG = {...}
```

✅ **Correct:**
```python
# Right
BERNARD_CONFIG = {...}
```

❌ **Don't use old names:**
```python
# Wrong
from protege.vision import ...
class JarvisCore: ...
ARNOLD_PATH = ...
```

✅ **Correct:**
```python
# Right
from bernard.vision import ...
class BernardCore: ...
BERNARD_PATH = ...
```

---

## Summary

- **Public/External → Eridos** (the venture)
- **Code/Internal → Bernard** (the system)
- **Visual/Logo → 𒉣𒆠** (the mark)

When in doubt: Is this user-facing or code-facing? User-facing = Eridos. Code-facing = Bernard.
