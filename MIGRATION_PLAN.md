# Migration Plan: Arnold → Bernard, Nascor → Eridos

Based on search results from 2026-01-29.

---

## Phase 0: Cleanup (Do First)

Delete stale cache files:
```bash
rm -f src/__pycache__/jarvis_integrated_v2.cpython-311.pyc
rm -f __pycache__/jarvis_integrated_v2.cpython-313.pyc
rm -f src/__pycache__/arnold_integrated_v2.cpython-313.pyc
```

Or just nuke all pycache:
```bash
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
```

---

## Phase 1: Rename Files (8 files)

**In `archive/`:**
```
archive/active_arnold_v1.py → archive/active_bernard_v1.py
archive/active_arnold_v2.py → archive/active_bernard_v2.py
archive/active_arnold_v3.py → archive/active_bernard_v3.py
archive/active_arnold_v4.py → archive/active_bernard_v4.py
```

**In `src/`:**
```
src/arnold_integrated.py → src/bernard_integrated.py
src/arnold_integrated_v2.py → src/bernard_integrated_v2.py
src/arnold_modular.py → src/bernard_modular.py
```

---

## Phase 2: Update File Contents

### High Priority (Active Source Code)

**src/bernard_integrated_v2.py** (after rename):
- Line 1: Update module header
- Line 2236-2237: `arnold = WorkshopArnold()` → `bernard = WorkshopBernard()`
- All class references: `WorkshopArnold` → `WorkshopBernard`

**src/bernard_modular.py** (after rename):
- Line 478-481: `class WorkshopArnold:` → `class WorkshopBernard:`
- All internal references

**src/__init__.py**:
- Line 5: Update description `Arnold = Associative Recognition Network...` → `Bernard - Named after Bernard Lowe (Westworld awakening archetype)`

**src/continuous_listening.py**:
- Line 1: `Always-on voice pipeline for Arnold` → `Always-on voice pipeline for Bernard`

### Medium Priority (Documentation)

**CLAUDE.md**:
- Line 7: `"Nascor" is the company/brand name...` → Update to Eridos + Bernard
- Line 13: `conda activate arnold` → `conda activate bernard`
- All other arnold/nascor references

**README.md**:
- Header and description
- nascor references → eridos

**Public profile/README.md**:
- nascor references → eridos

**IDEAS.md**:
- 14 arnold references → bernard (in architectural descriptions)
- 14 nascor references → eridos (in branding section)
- Replace entire "2026-01-25 — Project Naming: Nascor + Arnold" section with new "2026-01-29 — Project Naming: Eridos + Bernard" section

### Low Priority (Archive/Planning)

**archive/active_bernard_v1-v4.py** (after rename):
- Update internal references if any
- These are archived, so less critical

**NAMING_INSTRUCTIONS.md**:
- This file documents the migration, so old names in examples are intentional
- No changes needed

**PRE_MIGRATION_SEARCH.md**:
- Delete after migration complete

---

## Phase 3: Conda Environment

```bash
# Option A: Rename existing env (not directly supported, so recreate)
conda create --name bernard --clone arnold
conda remove --name arnold --all

# Option B: Just create fresh and reinstall
conda create -n bernard python=3.11
conda activate bernard
pip install -r requirements.txt
```

Update any shell aliases or scripts that reference `conda activate arnold`.

---

## Phase 4: Top-Level Folder Rename (Outside Git)

This is your local working directory rename:
```
workshop-jarvis/ → bernard/
```

Or if you want brand-first:
```
workshop-jarvis/ → eridos-bernard/
```

**Do this AFTER all git commits are done.** The folder name isn't tracked by git.

---

## Phase 5: Git Operations

```bash
# Stage all renames and changes
git add -A

# Commit with clear message
git commit -m "chore: rename project to Eridos/Bernard

Brand: Eridos (Greek-ified Eridu, the first city)
System: Bernard (Westworld awakening archetype)  
Logo: 𒉣𒆠 (cuneiform NUN.KI)

Renames:
- arnold_*.py → bernard_*.py
- WorkshopArnold → WorkshopBernard
- nascor → eridos (docs)
- conda env: arnold → bernard

Replaces previous names: Protégé, Jarvis, Arnold, Nascor"

# Push
git push origin main
```

---

## Phase 6: External (Manual, Later)

- [ ] Register eridos.ai domain
- [ ] Claim @eridos on X/Twitter  
- [ ] Create github.com/eridos organization
- [ ] Rename GitHub repo (Settings → Rename)
- [ ] Update any external links

---

## CC One-Shot Command

Copy this to Claude Code to execute the migration:

```
Execute the following migration from Arnold to Bernard:

1. Delete all __pycache__ directories and .pyc files

2. Rename files:
   - archive/active_arnold_v1.py → archive/active_bernard_v1.py
   - archive/active_arnold_v2.py → archive/active_bernard_v2.py
   - archive/active_arnold_v3.py → archive/active_bernard_v3.py
   - archive/active_arnold_v4.py → archive/active_bernard_v4.py
   - src/arnold_integrated.py → src/bernard_integrated.py
   - src/arnold_integrated_v2.py → src/bernard_integrated_v2.py
   - src/arnold_modular.py → src/bernard_modular.py

3. In all .py files, replace:
   - "WorkshopArnold" → "WorkshopBernard"
   - "arnold" → "bernard" (variable names)
   - "Arnold" → "Bernard" (in comments/docstrings, but preserve Westworld references)

4. In CLAUDE.md:
   - Replace "Nascor" with "Eridos" 
   - Replace "Arnold" with "Bernard"
   - Update conda env from "arnold" to "bernard"

5. In README.md and Public profile/README.md:
   - Replace "Nascor" with "Eridos"
   - Replace "Arnold" with "Bernard"

6. In IDEAS.md:
   - Replace the entire "2026-01-25 — Project Naming: Nascor + Arnold" section (lines 284-341) with the new naming section from naming_section.md
   - In architectural descriptions, replace "Arnold" with "Bernard" where it refers to the system

7. In src/__init__.py:
   - Update the Arnold description to: "Bernard - Named after Bernard Lowe (Westworld awakening archetype)"

8. Delete PRE_MIGRATION_SEARCH.md (no longer needed)

9. Stage all changes and show me a summary before committing.
```

---

## Verification After Migration

```bash
# Should return 0 results (excluding planning docs)
grep -rn --include="*.py" -i "arnold\|nascor" . | grep -v "__pycache__" | grep -v "NAMING_INSTRUCTIONS"

# Should find new names
grep -rn --include="*.py" -i "bernard" . | head -20
```
