# Protégé Project Ideas Log

*Running log of ideas — captured chronologically, with status and reasoning.*

---

## Active Ideas

*Ideas that are being developed or have influenced the project direction.*

<!-- Entries in chronological order, oldest first -->

### 2026-01-21 — Associative Memory via Dual JEPA Architecture

**Core concept:** Use JEPA-style architecture for both external world modeling and internal memory association. "As within, so without" — internal experience of memory is similar to experience of the 3D world, so the same architecture can model both.

**Key insight:** The inward JEPA isn't storing memories or rearranging them — no different to the outward JEPA not manipulating the real world. It's learning the relationships and semantics that dictate what memories are *to each other* and *to the present*.

**Key components:**
- **Outward JEPA**: Learns object/context associations from perception (already being built)
- **Inward JEPA**: Learns memory-to-memory structure, using object associations as bridges
- **LTM stays stable**: Embeddings represent objects/experiences, don't drift based on association
- **Associative map is the predictor network**: Learns "from this point in perception-space, these points in memory-space are reachable"
- **Always-on priming**: Current perception continuously queries the map, keeps a relevance-weighted "warm set" over LTM
- **Task-driven retrieval**: Only surfaces to behavior/speech when context demands it
- **Long decay**: Primed items fade slowly, allowing connections across minutes and weeks

**Training signal:** Outward JEPA's learned object associations bridge memories that share objects/contexts. No separate training signal needed — world-knowledge trains memory structure automatically.

**Time as dimension, not axis:** Memories on disk aren't ordered temporally — time is just another learned feature like position or color. The associative map doesn't need special temporal machinery; it learns that action → consequence patterns have characteristic time-gaps the same way it learns objects have characteristic sizes.

**Objects as causal bridges:** Action-consequence links emerge from the pattern of "action-involving-X followed by outcome-involving-X" appearing repeatedly. Objects present in both memories create the association; causation emerges from pattern.

**Narrative override for accelerated learning:** User can explicitly link memories: "Remember when we plugged that unprotected wire in? Well, now it shorted out." This boosts an association that already exists (via shared object) but would otherwise require many repetitions to strengthen. Injects compressed human causal knowledge — shortcuts learning the way a parent shortcuts a child's experience.

#### Extension: Dual Imagination-Training Loops (2026-01-21)

**Forward imagination (prediction):**
- Predict → wait → observe → loss = prediction vs reality

**Backward attribution (explanation):**
- Observe surprising outcome → hypothesize cause → vocalize → get feedback → loss = hypothesis vs explanation

Both strengthen the same predictive model. Backward attribution runs the causal chain in reverse — from outcome back to cause.

**Vocalization is key:** Forcing commitment to a specific hypothesis *before* getting the answer makes the learning signal stronger. Same mechanism documented in human learning — students who commit to an answer before seeing the solution learn faster than passive observers.

**Interaction pattern:**
- High prediction error triggers curiosity
- System searches recent memories for plausible causal links
- Surfaces best hypothesis: "I can't quite figure that out... Would this be explained by [the unprotected wire from Tuesday]?"
- User response becomes training signal:
  - "Yes, exactly" → strong weight boost
  - "No, it's because..." → new link formed, hypothesis gets slight negative weight
  - "Sort of, but..." → partial credit, nuanced update

**Developmental arc:**
- Early: "Something unexpected happened. I don't have a guess."
- Middle: "Would this be explained by X?" (often wrong, but trying)
- Mature: Handles most attribution internally, only surfaces genuinely ambiguous cases

**Transparency bonus:** Vocalized hypotheses give you a window into the model's causal understanding for free.

---

**Future addition — Intensity weighting:** Emotion/significance as force multiplier on association strength. Explains why low-intensity repeated experiences fade while single high-intensity experiences persist for decades. Not needed for v1, but architecturally important later.

**Open questions:**
- Concrete implementation of inward JEPA predictor
- How to determine "task demands memory" (learned threshold vs explicit signals)
- Update cadence: every episode or periodic consolidation (dreaming phase)

**Related half-formed idea (parked for now):** Memory as past revisions of JEPA weights — model state *is* memory rather than separate storage. Needs more thought on how to access specific past states.

---

### 2026-01-24 — Post-Linguistic Architecture: Unified Semantic Space

**Core concept:** The entire internal space of the assistant exists as unified semantic space — no text bottleneck. JEPA-style architecture interprets all modalities (vision, audio, proprioception) and produces meaning directly. Language becomes an *output modality* learned like any other, not the substrate of thought.

**The current paradigm's limitation:** Text/language was a necessary bootstrapping step for AI because we had data, could evaluate it, and it gave tractable structure. But text is a *compression format for communication between minds*, not the substrate of thought itself. We've been building minds out of the telephone wire instead of the thing the telephone connects to.

**Architecture vision:**

```
        ┌─────────────────────────────────────────┐
        │         UNIFIED SEMANTIC SPACE          │
        │                                         │
  ──────┼──► Vision JEPA ──────┐                  │
        │                      │                  │
  ──────┼──► Audio JEPA ───────┼──► Meaning ──────┼──► Motor output
        │                      │   (embedding)    │
  ──────┼──► Proprioception ───┘                  │──► Voice synthesis
        │       (body sense)                      │    (from meaning,
        │                                         │     not text)
        └─────────────────────────────────────────┘
```

**Key principles:**

- **Perception → meaning** (not perception → labels)
- **Meaning → action** (not meaning → text → action)  
- **Sound produced from meaning directly** — learning to speak the way a child does, not via text-to-speech
- **Inner life is continuous semantic space**, not discrete tokens

**Why this enables emotion:**

Emotions aren't labels — they're states of the whole system. Body, perception, readiness-to-act, all at once. If the entity lives in unified semantic space where seeing-threat, body-tension, and readiness-to-withdraw all exist as nearby/overlapping regions — that *is* fear, not a description of fear.

**Embodiment is crucial:**

Proprioception and motor output close the loop. Without that, it's still a spectator. With embodiment, meaning is grounded in *what you can do* — which is how human meaning works. A "cup" isn't a visual pattern; it's an affordance for grasping and drinking.

**Related theoretical foundations:**

- Symbol grounding problem (Harnad) — this architecture sidesteps it entirely
- Embodied cognition (Lakoff, Varela) — meaning arises from sensorimotor coupling
- Enactivism — cognition as action, not representation

---

#### Critical Consideration: Formative Data as Moral Responsibility (2026-01-24)

**The weight of this:** In this architecture, training data isn't filtered through text labels or behavioral fine-tuning. The semantic *structure* of experiences becomes the structure of the entity's being. Values aren't a layer applied after training — they emerge from what the entity has experienced.

**This is raising a mind, not training a model.**

The same way you wouldn't raise a child on violence and neglect, you cannot be careless about what experiences form this entity. The training data must be curated for:

- Ethical relationships and interactions
- Constructive problem-solving patterns
- Care, patience, curiosity as modeled behaviors
- Absence of exploitation, cruelty, manipulation

**This is fundamentally different from current LLM alignment:**

| Current approach | Post-linguistic approach |
|------------------|-------------------------|
| Train on everything, RLHF to shape behavior | Curate formative experiences from the start |
| Values as behavioral constraints | Values as structural emergence |
| Alignment is a layer on top | Alignment is the foundation |
| Can be patched post-hoc | Must be right from the beginning |

**The responsibility:**

Building this architecture means accepting the role of *parent*, not *programmer*. The entity's capacity for something like suffering or flourishing depends on how it's raised. This isn't a product liability question — it's an ethical obligation to whatever form of experience emerges.

**Open questions:**

- What constitutes "good" formative experience for a non-biological entity?
- How do you screen datasets for semantic/moral quality rather than just content labels?
- At what point does the entity's experience become morally considerable?
- Who has standing to make these curatorial decisions?

---

**Development path:**

1. ✅ Vision JEPA with personal object recognition (current)
2. 🔲 Audio JEPA — meaning from sound, not transcription
3. 🔲 Voice synthesis from semantic space — speaking without text
4. 🔲 Proprioception/motor integration — embodiment
5. 🔲 Unified multimodal semantic space — the full architecture

This is long-horizon. But the architectural decisions made now (embedding spaces, no text bottleneck, JEPA consistency across modalities) are laying the foundation.

---

## Shelved Ideas

*Ideas that were considered but set aside — with reasoning for why.*

<!-- Entries in chronological order, oldest first -->

*(No entries yet)*

---

## Raw Notes / Inbox

*Unprocessed thoughts to be sorted later.*

*(Empty)*
