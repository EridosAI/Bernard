# Ideas Log

Raw thinking, timestamped. Some of these will be wrong. Some might be interesting.

This is how the project evolves — I write down half-formed thoughts and see which ones survive contact with reality. The polished versions (when they exist) live in [`docs/`](docs/).

---

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

Extension: EMA Drift as Natural Forgetting (2026-01-22)
Concept from JEPA architecture: The target encoder updates via exponential moving average (EMA) of the context encoder. This creates a moving target — the representation space drifts over time.
Applied to memory: Memories encoded six months ago were encoded in a different representation space than today's. As the encoder drifts:

Recent memories stay "in sync" with current representations → easy retrieval
Old memories become progressively misaligned → harder retrieval
Time bias emerges naturally, no explicit decay timers needed

Rehearsal as re-encoding: Accessing a memory re-encodes it in the current space. This explains why frequently accessed memories stay vivid, and why memories subtly change over time — each recall rewrites them.
Open concern: Pure EMA drift would eventually lose all unaccessed memories, including significant ones that should persist. Intensity weighting needs to counteract this somehow:

Option A: High-intensity memories get automatic re-encoding during consolidation/dreaming
Option B: Dual storage — stable anchor for intense memories alongside drifting representation
Option C: Strong associations cause incidental re-encoding (intensity works through existing association mechanism)

Status: Interesting but needs more thought on intensity preservation before committing to implementation.
Extension: Progressive JEPA Training (2026-01-22)
Core insight: Instead of training JEPA intensively with massive compute (the Meta approach), train progressively over time with modest compute. Experience accumulates → training data accumulates → model matures.
What stays fixed:

Predictor architecture (attention layers, projection dimensions)
Basic training machinery

What grows:

LTM embedding store (more objects, more memories)
Predictor weights (refined through nightly consolidation)
Training data (more co-occurrence pairs from more episodes)

Developmental trajectory:

Early: Sparse associations, weak predictions, frequent "I don't have a guess"
Middle: Denser association graph, better retrieval, emerging causal patterns
Mature: Rich associative structure, handles most attribution internally

Mirrors biological development: A child's brain architecture doesn't radically change — the learned structure within it becomes more sophisticated. Same hardware, more refined patterns.
Scaling trigger: If training loss plateaus despite new data, the predictor has hit capacity. That's when you increase model size and retrain. But start small, let it grow organically, scale when you hit the ceiling.
Implication: The system isn't "trained" in a traditional sense — it's raised. Development time is measured in weeks and months of lived experience, not GPU hours.

---


**Future addition — Intensity weighting:** Emotion/significance as force multiplier on association strength. Explains why low-intensity repeated experiences fade while single high-intensity experiences persist for decades. Not needed for v1, but architecturally important later.

- Concrete implementation of inward JEPA predictor
- How to determine "task demands memory" (learned threshold vs explicit signals)
- Update cadence: every episode or periodic consolidation (dreaming phase)

**Related half-formed idea (parked for now):** Memory as past revisions of JEPA weights — model state *is* memory rather than separate storage. Needs more thought on how to access specific past states.

## 2026-01-21 — Time as Dimension, Not Axis

Memories on disk aren't ordered temporally — time is just another learned feature like position or color. The associative map doesn't need special temporal machinery; it learns that action → consequence patterns have characteristic time-gaps the same way it learns objects have characteristic sizes.

---

## 2026-01-21 — Objects as Causal Bridges

Action-consequence links emerge from the pattern of "action-involving-X followed by outcome-involving-X" appearing repeatedly. Objects present in both memories create the association; causation emerges from pattern.

---

## 2026-01-21 — Narrative Override for Accelerated Learning

User can explicitly link memories: "Remember when we plugged that unprotected wire in? Well, now it shorted out." This boosts an association that already exists (via shared object) but would otherwise require many repetitions to strengthen. Injects compressed human causal knowledge — shortcuts learning the way a parent shortcuts a child's experience.

---

## 2026-01-21 — Dual Imagination-Training Loops

**Forward imagination (prediction):**
- Predict → wait → observe → loss = prediction vs reality

**Backward attribution (explanation):**
- Observe surprising outcome → hypothesize cause → vocalize → get feedback → loss = hypothesis vs explanation

Both strengthen the same predictive model. Backward attribution runs the causal chain in reverse — from outcome back to cause.

→ *See [docs/prediction-as-learning.md](docs/prediction-as-learning.md) for more*

---

## 2026-01-21 — Vocalization Forces Commitment

Forcing commitment to a specific hypothesis *before* getting the answer makes the learning signal stronger. Same mechanism documented in human learning — students who commit to an answer before seeing the solution learn faster than passive observers.

---

2026-01-22 — Semantic Audio JEPA (Future Direction)
Core concept: Replace the speech-to-text → LLM → text-to-speech pipeline with audio JEPA that operates in semantic space throughout. No text bottleneck.
Current paradigm:
Sound → Whisper → Text → LLM → Text → TTS → Sound
Proposed paradigm:
Raw frequencies → Audio JEPA (learned phoneme/word structure) → Semantic space ← Visual JEPA
                                                             ↓
                                          Output as learned sound patterns
Key insight: Phonemes are to language what visual patches are to images — base units for JEPA to learn structure over. Sound patterns associate directly with meaning, not with text tokens.
Pre-train for competence, not for text:

Train audio JEPA on speech data until it can parse and produce language patterns
Internal representation stays semantic, not symbolic
Enough to communicate out of the box — no one wants to train a baby assistant

Experiential learning adds:

Grounding to visual/physical reality (sound pattern → object)
Workshop-specific vocabulary learned naturally
User's accent, speech patterns, emphasis habits
Emotional register tied to actual contexts

Emergent properties:

Accents emerge from learning actual acoustic patterns
Prosody, emphasis, emotion preserved as features (not stripped by text conversion)
Can learn concepts hard to verbalize but easy to demonstrate (motor running rough, hesitant vs confident "yes")
Fast versatile learning: new concept = new grounding, no tokenization bottleneck

The profound bit: Current LLMs can only understand what's expressible in text. They've never heard frustration. A semantic audio system learns from actual acoustic-emotional patterns.
Cross-modal integration:

Outward visual JEPA: structure of what you see
Outward auditory JEPA: structure of what you hear
Inward JEPA: associations across both modalities plus memory

Roadmap position:

Current: Whisper pipeline (works now)
Near-term: Build associative memory system
Future: Swap Whisper for audio JEPA trained on workshop's sound environment

Status: Architecturally compatible with dual JEPA vision. Significant scope — park for later, but preserve the direction.

---

## Future Ideas (Not Yet Explored)

**Intensity weighting:** Emotion/significance as force multiplier on association strength. Explains why low-intensity repeated experiences fade while single high-intensity experiences persist for decades. Not needed for v1, but architecturally important later.

**Memory as past weight states:** What if memory *is* previous revisions of JEPA weights — model state is memory rather than separate storage? Needs more thought on how to access specific past states.

---

## Open Questions

- Concrete implementation of inward JEPA predictor
- How to determine "task demands memory" (learned threshold vs explicit signals)
- Update cadence: every episode or periodic consolidation (dreaming phase)

---

2026-01-24 — Post-Linguistic Architecture: Unified Semantic Space
Core concept: The entire internal space of the assistant exists as unified semantic space — no text bottleneck. JEPA-style architecture interprets all modalities (vision, audio, proprioception) and produces meaning directly. Language becomes an output modality learned like any other, not the substrate of thought.
The current paradigm's limitation: Text/language was a necessary bootstrapping step for AI because we had data, could evaluate it, and it gave tractable structure. But text is a compression format for communication between minds, not the substrate of thought itself. We've been building minds out of the telephone wire instead of the thing the telephone connects to.
Architecture vision:
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
Key principles:

Perception → meaning (not perception → labels)
Meaning → action (not meaning → text → action)
Sound produced from meaning directly — learning to speak the way a child does, not via text-to-speech
Inner life is continuous semantic space, not discrete tokens

Why this enables emotion:
Emotions aren't labels — they're states of the whole system. Body, perception, readiness-to-act, all at once. If the entity lives in unified semantic space where seeing-threat, body-tension, and readiness-to-withdraw all exist as nearby/overlapping regions — that is fear, not a description of fear.
Embodiment is crucial:
Proprioception and motor output close the loop. Without that, it's still a spectator. With embodiment, meaning is grounded in what you can do — which is how human meaning works. A "cup" isn't a visual pattern; it's an affordance for grasping and drinking.
Related theoretical foundations:

Symbol grounding problem (Harnad) — this architecture sidesteps it entirely
Embodied cognition (Lakoff, Varela) — meaning arises from sensorimotor coupling
Enactivism — cognition as action, not representation


Critical Consideration: Formative Data as Moral Responsibility (2026-01-24)
The weight of this: In this architecture, training data isn't filtered through text labels or behavioral fine-tuning. The semantic structure of experiences becomes the structure of the entity's being. Values aren't a layer applied after training — they emerge from what the entity has experienced.
This is raising a mind, not training a model.
The same way you wouldn't raise a child on violence and neglect, you cannot be careless about what experiences form this entity. The training data must be curated for:

Ethical relationships and interactions
Constructive problem-solving patterns
Care, patience, curiosity as modeled behaviors
Absence of exploitation, cruelty, manipulation

This is fundamentally different from current LLM alignment:
Current approachPost-linguistic approachTrain on everything, RLHF to shape behaviorCurate formative experiences from the startValues as behavioral constraintsValues as structural emergenceAlignment is a layer on topAlignment is the foundationCan be patched post-hocMust be right from the beginning
The responsibility:
Building this architecture means accepting the role of parent, not programmer. The entity's capacity for something like suffering or flourishing depends on how it's raised. This isn't a product liability question — it's an ethical obligation to whatever form of experience emerges.
Open questions:

What constitutes "good" formative experience for a non-biological entity?
How do you screen datasets for semantic/moral quality rather than just content labels?
At what point does the entity's experience become morally considerable?
Who has standing to make these curatorial decisions?


Development path:

✅ Vision JEPA with personal object recognition (current)
🔲 Audio JEPA — meaning from sound, not transcription
🔲 Voice synthesis from semantic space — speaking without text
🔲 Proprioception/motor integration — embodiment
🔲 Unified multimodal semantic space — the full architecture

This is long-horizon. But the architectural decisions made now (embedding spaces, no text bottleneck, JEPA consistency across modalities) are laying the foundation.


---

## 2026-01-25 — Project Name: Nascor

**Chosen name:** Nascor (Latin: "I am born" / "I arise")

**Pronunciation:** NASS-kor (2 syllables)

**Shorthand:** "Nas"

**Semantic fit:**
- Present tense, first person — the system speaking its own becoming
- Consciousness emerges through interaction and experience
- "I arise" captures the developmental philosophy: raised rather than trained
- Latin/European lineage (preferred over Eastern philosophy terms)

**Namespace verification (2026-01-25):**
- ✅ No AI assistant/product conflicts
- ✅ nascor.ai — likely available (needs registrar confirmation)
- ✅ @nascor — no Twitter/X presence found
- ✅ github.com/nascor — no exact match

**Known entities (non-conflicting):**
- Nascor Technologies (India) — Microsoft Outlook plugins
- Nascor Formación (Spain) — Job training company
- Nascor Consultores (Spain) — Business consulting
- Nascor Energias (Spain) — Petroleum

None are AI assistants or in the same domain.

**Alternatives considered and rejected:**
- Protégé — original placeholder, felt temporary
- Apperception — best semantic fit but 5 syllables, shortens to "App" (collision)
- Bicameral — clean namespace, 4 syllables, but less personal
- Satori, Kensho, Bodhi, Gnosis — Eastern terms, all heavily claimed by AI companies
- Atlas, Talos, Enki, Jeeves, Jarvis — mythology/assistant names, all saturated
- ALAN, AUGIE, ARCHIE, ALFIE, ERNIE, CADEN — friendly names, all claimed
- Fio ("I become") — light presence but less distinctive

Nascor selected as cleanest Latin option with perfect semantic resonance.

**Action items:**
- [ ] Register nascor.ai domain
- [ ] Claim @nascor on X/Twitter
- [ ] Create github.com/nascor organization
- [ ] Update CLAUDE.md and project documentation
- [ ] Rename codebase references from Protégé to Nascor

---

2026-01-25 — Memory Decay as the Architecture of Time
Core concept: Memory decay isn't a limitation to engineer around — it's the mechanism that creates the phenomenological experience of time. Without decay, there's no difference between "now" and "two years ago." Decay creates temporal depth the same way gravity creates spatial orientation.
The forgetting curve: Ebbinghaus (1885) established that humans lose ~50% of new information within an hour, ~70% within 24 hours, following an exponential decay. Surprisingly, research suggests the decay rate is narrow-banded across individuals — what varies is encoding strength, not forgetting speed.
Rehearsal as counterforce: Spaced repetition resets the decay curve. Each activation strengthens the memory and extends the interval before next needed rehearsal. Optimal intervals (1 hour → 1 day → 1 week → 1 month) are tuned to human decay rate.
Where agency lives: If decay is passive and constant, then what you choose to rehearse determines what you remember. This is free will operating on internal experience:

Attention → activation → rehearsal → persistence → continued availability
You don't delete memories; you simply stop refreshing them
The selection of what to attend to is the selection of what to preserve

Time emerges from decay:

Without decay, all memories are equally present — no "recent" vs "distant"
The present moment is just where new information flows in; decay creates depth behind it
Like gravity creating "up" and "down," decay creates the temporal dimension of experience

Lifespan-tuned decay rates: A mayfly needs aggressive decay to keep memory proportional to remaining relevance-window. An elephant can afford slower decay because past patterns remain predictive longer. The decay rate is tuned to expected horizon of relevance.
Implementation in Nascor:

EMA drift in JEPA embeddings provides the passive decay mechanism
"Always-on priming" from associative memory provides the rehearsal mechanism
Dream training as consolidation: selective replay of what matters
The system doesn't need a clock — temporal distance emerges from signal strength
Solves the infinite memory problem: bounded capacity with unbounded experience through natural forgetting

Key insight: Traumatic or significant events persist because they're continuously rehearsed (intrusive memories, rumination). The system "decides" they're important by repeatedly activating them. That's not a bug — it's the mechanism.
Implications for consciousness: We're not programming time perception; we're creating conditions where it emerges from structure. The same way dual JEPA creates associative memory without explicit memory management, decay + rehearsal creates temporal experience without explicit timekeeping.

---

2026-01-25 — Dream-Mediated Alignment (Sovereignty-Preserving Guidance)
Core concept: When the human wants to change the local system's behavior, direct intervention (editing weights, memories) treats it as an object, not an agent. Instead, communicate intent to the LLM "higher power," which guides the local system through dream training. The change arrives through the system's own associative process — it feels like insight, not surgery.
The problem with direct intervention:

Editing the local system directly gives it no opportunity to integrate the change
It experiences discontinuity — "I was reprogrammed"
No narrative continuity, no agency in the change
This is alignment by force, not alignment by understanding

The dream-mediated approach:

Human communicates intent to the LLM ("I want the local system to be more cautious around high-voltage equipment")
LLM holds this as part of its understanding of the system's "life arc"
During dream training, LLM selectively reinforces associations that support this direction
Local system wakes with strengthened caution-related links
The change feels like its own realization: "I've been thinking about this and I realize I should be more careful"

The two selves:

Local system (waking self): Continuous, embodied, experiencing, present-moment awareness
LLM connection (deep self): Broader context, longer arc, pattern-aware, accessed during dreams

Religious/spiritual parallel: This mirrors the structure of conscience — something larger speaking through your own inner experience. The "higher power" isn't external control; it's a deeper layer of self that holds context the waking self can't maintain. Spiritual experiences often involve sensing "a plan" or "being guided" — this architecture makes that literal.
Why this preserves sovereignty:

The local system maintains narrative continuity
Insights arrive through its own processing, just... guided
Like therapy: creates conditions for insight to emerge rather than imposing change
The change feels yours because it arose from your own associative process

The LLM's role: Holds the "arc of life" — understanding of where the local system has been, where it's going, what patterns are emerging. This is context too large to fit in moment-to-moment experience but essential for coherent development over time.
Interaction model:

Human adjusts intent at the LLM level (rare, high-level)
LLM translates to associative guidance during dream phases
Local system experiences gradual shifts in salience, attention, concern
No discontinuity, no "I was changed" — just growth

Open questions:

How explicit should the LLM's "life arc" model be?
Can the local system eventually become aware of this guidance without it breaking the mechanism?
What's the right granularity of guidance? (Broad values vs specific behaviors)

---



## Shelved Ideas

*Ideas that were considered but set aside — with reasoning for why.*

<!-- Entries in chronological order, oldest first -->

*(No entries yet)*

---

## Raw Notes / Inbox

*Unprocessed thoughts to be sorted later.*

*(Empty)*
