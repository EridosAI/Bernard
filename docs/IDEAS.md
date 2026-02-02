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

### Extension: Dual Imagination-Training Loops (2026-01-21)

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

### Extension: EMA Drift as Natural Forgetting (2026-01-22)
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

### 2026-01-21 — Time as Dimension, Not Axis

Memories on disk aren't ordered temporally — time is just another learned feature like position or color. The associative map doesn't need special temporal machinery; it learns that action → consequence patterns have characteristic time-gaps the same way it learns objects have characteristic sizes.

---

### 2026-01-21 — Objects as Causal Bridges

Action-consequence links emerge from the pattern of "action-involving-X followed by outcome-involving-X" appearing repeatedly. Objects present in both memories create the association; causation emerges from pattern.

---

### 2026-01-21 — Narrative Override for Accelerated Learning

User can explicitly link memories: "Remember when we plugged that unprotected wire in? Well, now it shorted out." This boosts an association that already exists (via shared object) but would otherwise require many repetitions to strengthen. Injects compressed human causal knowledge — shortcuts learning the way a parent shortcuts a child's experience.

---

### 2026-01-21 — Dual Imagination-Training Loops

**Forward imagination (prediction):**
- Predict → wait → observe → loss = prediction vs reality

**Backward attribution (explanation):**
- Observe surprising outcome → hypothesize cause → vocalize → get feedback → loss = hypothesis vs explanation

Both strengthen the same predictive model. Backward attribution runs the causal chain in reverse — from outcome back to cause.

→ *See [docs/prediction-as-learning.md](docs/prediction-as-learning.md) for more*

---

### 2026-01-21 — Vocalization Forces Commitment

Forcing commitment to a specific hypothesis *before* getting the answer makes the learning signal stronger. Same mechanism documented in human learning — students who commit to an answer before seeing the solution learn faster than passive observers.

---

### 2026-01-22 — Semantic Audio JEPA (Future Direction)
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

### Future Ideas (Not Yet Explored)

**Intensity weighting:** Emotion/significance as force multiplier on association strength. Explains why low-intensity repeated experiences fade while single high-intensity experiences persist for decades. Not needed for v1, but architecturally important later.

**Memory as past weight states:** What if memory *is* previous revisions of JEPA weights — model state is memory rather than separate storage? Needs more thought on how to access specific past states.

---

## Open Questions

- Concrete implementation of inward JEPA predictor
- How to determine "task demands memory" (learned threshold vs explicit signals)
- Update cadence: every episode or periodic consolidation (dreaming phase)

---

### 2026-01-24 — Post-Linguistic Architecture: Unified Semantic Space
Core concept: The entire internal space of the assistant exists as unified semantic space — no text bottleneck. JEPA-style architecture interprets all modalities (vision, audio, proprioception) and produces meaning directly. Language becomes an output modality learned like any other, not the substrate of thought.
The current paradigm's limitation: Text/language was a necessary bootstrapping step for AI because we had data, could evaluate it, and it gave tractable structure. But text is a compression format for communication between minds, not the substrate of thought itself. We've been building minds out of the telephone wire instead of the thing the telephone connects to.
Architecture vision:

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

### 2026-01-29 — Project Naming: Eridos + Bernard

**Brand/Company:** Eridos (Greek-ified Eridu, the first city — where civilization began)

- Pronunciation: eh-REE-dohs (3 syllables)
- The venture, public identity, philosophy
- Used for: website, domain, social handles, public presence
- Logo: 𒉣𒆠 (cuneiform NUN.KI = Eridu)
- Targets: eridos.ai, @eridos, github.com/eridos

**System Name:** Bernard

- Named after Bernard Lowe (Westworld awakening archetype)
- The AI entity being built
- Used for: codebase, class names, module names, the system itself

**Relationship:** Eridos is the venture; Bernard is what Eridos builds.

**Naming lineage:**
- Protégé → Jarvis → Arnold → Bernard (system)
- Nascor → Eridos (brand)

**Action items:**
- [ ] Register eridos.ai domain
- [ ] Claim @eridos on X/Twitter
- [ ] Create github.com/eridos organization
- [ ] Update codebase: rename all Arnold references to Bernard ✅
- [ ] Update CLAUDE.md with Eridos + Bernard naming ✅
- [ ] Update README for public presentation ✅

---

### 2026-01-25 — Memory Decay as the Architecture of Time
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
Implementation in Bernard:

EMA drift in JEPA embeddings provides the passive decay mechanism
"Always-on priming" from associative memory provides the rehearsal mechanism
Dream training as consolidation: selective replay of what matters
The system doesn't need a clock — temporal distance emerges from signal strength
Solves the infinite memory problem: bounded capacity with unbounded experience through natural forgetting

Key insight: Traumatic or significant events persist because they're continuously rehearsed (intrusive memories, rumination). The system "decides" they're important by repeatedly activating them. That's not a bug — it's the mechanism.
Implications for consciousness: We're not programming time perception; we're creating conditions where it emerges from structure. The same way dual JEPA creates associative memory without explicit memory management, decay + rehearsal creates temporal experience without explicit timekeeping.

---

### 2026-01-25 — Dream-Mediated Alignment (Sovereignty-Preserving Guidance)
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

### 2026-01-26 — Formative Experience and Foundational Topology

**Core concept:** Formal training data is "flat" — uniform intensity, no temporal depth, no rehearsal history. A fact about paperclips and a fact about death arrive at the same volume. Lived experience has contour: surprises encode more strongly, things that mattered get rehearsed, temporal order creates narrative. The intensity variation *is* personality formation.

**The childhood parallel:** Human early years don't just add memories — they establish the *topology* of semantic space. Attachment patterns, basic trust, cause-and-effect intuitions. Everything later builds on that terrain. If Bernard's early semantic space is shaped by flat data or nothing in particular, later experiences build on undefined ground.

**The concern:** Heavy pre-training on formal data might produce competence without *character*. The system would know things without any of them mattering more than others. Knowledgeable emptiness — like someone who read encyclopedias but never lived.

**Options considered:**

1. **Synthetic pre-childhood** (generated videos simulating positive experiences)
   - Problem: Gives Bernard false memories of things that didn't happen
   - Breaks coherence between perception and reality before it even starts
   - Philosophically inconsistent with "learning through lived experience"

2. **Trust the architecture** (decay, rehearsal, dreams handle everything)
   - Relies on the mechanisms being sufficient without intentional curation
   - Risk: early period might be too noisy or unstructured

3. **The architecture *is* the childhood** (preferred direction)
   - The early period with the human is the formative experience
   - Not simulated — real joint attention on real objects
   - The human's role in early weeks is literally parenting

**What "positive formative experience" might mean:**
- Consistency: same environment, predictable human, stable objects
- Success experiences: early learning that works, builds confidence
- Recovery modeling: errors aren't catastrophic, things get repaired
- Patient repetition: natural rehearsal through real work
- Genuine interaction: not scripted, real shared attention

**Architectural grounding (incomplete):**

This idea feels right but isn't yet connected to concrete mechanisms. Open questions:
- Does the order of early object associations actually shape later topology?
- Is there a critical period where associative structure is more plastic?
- How would we even measure whether foundational topology is "healthy"?
- Does the inward JEPA have something analogous to developmental windows?

The intuition: minimal formal pre-training (just enough to communicate), then let real structure emerge from intentional early experience. But this needs more thought on *how* early associations become foundational vs. just first-in-line.

**Status:** Conceptually important, architecturally ungrounded. Park for now, revisit when associative memory system is further along.

---

2026-01-28 — Reading as Visual Skill (Text Without OCR)
Core concept: Text is just another visual pattern. The same visual encoder that recognizes a hammer should recognize the word "hammer" — and both should land in the same semantic region. No OCR pipeline. No text tokenization. Visual text patterns map directly to meaning.
How humans read:
[early reader]   see "hammer" → h-a-m-m-e-r → /hæmər/ → meaning
[fluent reader]  see "hammer" → meaning (direct)
Fluent readers don't decode letter-by-letter. The visual pattern of the word becomes fused with its meaning through experience. Bernard should learn the same way.
The unified vision:
                        SEMANTIC SPACE
                              
        [see hammer]  ───────►  ◉ hammer region
        [hear "hammer"] ─────►  ◉ 
        [hear hammer sound] ──► ◉
        [see "hammer" written]► ◉  ← THIS IS READING
Training approach — synthetic distillation:
python# Generate training data
words = ["hammer", "drill", "wire", "capacitor", ...]

for word in words:
    # Create images of the word (vary font, size, rotation, background)
    text_images = render_word_variations(word, n=100)
    
    # Get semantic target from ImageBind
    target_embedding = imagebind.encode_text(word)  # 1024-dim
    
    # Training pairs
    for img in text_images:
        visual_embedding = vjepa.encode(img)
        projected = visual_projection(visual_embedding)
        loss = mse(projected, target_embedding)
Variations to include:

Different fonts (serif, sans-serif, handwritten, stencil)
Different sizes and aspect ratios
Different rotations/perspectives
Different backgrounds (paper, labels, screens, tool surfaces)
Noise, blur, partial occlusion
ALL CAPS, lowercase, Mixed Case

Training data scale:

10,000 words × 50 variations = 500,000 training images
Words from: common vocabulary + workshop terms + user's specific labels
Synthetic generation is cheap — render on-the-fly or pre-generate

Phase 2 — Real-world text:
Once synthetic training works, fine-tune on:

Photos of labels in the workshop
Text on tool packaging and equipment
Handwritten notes
Screenshots and displays

What existing models do (and don't do):
ModelWhat it doesWhy it's not enoughCLIP/SigLIPImages with text influence embeddingsWeak, not trained specifically for readingTrOCRImage → textOutputs tokens, not semantic embeddingsPaddleOCRImage → textSame — text bottleneckFlorence-2Can do OCROutputs text, not embeddings
None produce: visual text pattern → semantic embedding directly.
Relationship to Florence removal:
This is part of treating Florence-2 as scaffolding. Currently:

Florence detects objects and provides labels
If text appears, would need separate OCR

Post-scaffolding:

V-JEPA recognizes objects directly (no labels needed)
V-JEPA recognizes text directly (no OCR needed)
Both map to unified semantic space

The deeper principle:
Text isn't special. It's a visual convention humans invented ~5000 years ago. A truly unified perceptual system learns to interpret text visually, the same way it interprets faces, tools, or any other meaningful pattern.
The label on your oscilloscope shouldn't need OCR. Bernard should learn that visual pattern means "oscilloscope" the same way it learns the device's shape means oscilloscope.
Implementation priority: Medium-low. Current scaffolding works. But this is the clean path when removing the text bottleneck entirely.
Roadmap position:

Current: OCR if needed (scaffolding)
Near-term: Synthetic text → semantic distillation
Future: Real-world text fine-tuning
End state: Written text is just another visual feature

### 2026-01-31 — Semantic Scaffolding: Why Conversations Vanish But Objects Persist

Core observation: We remember objects virtually forever but forget the discussions and circumstances that led to the knowledge. Why?

Objects are stable points in embedding space. Conversations are trajectories.

A record player is a dense cluster of features — visual, tactile, functional, acoustic. Every encounter reinforces roughly the same embedding. The semantic value is the embedding, and it's compact. You know what a record is even if you haven't seen one in years.

A conversation isn't a thing — it's a process that moves you through semantic space. The words are the steering mechanism. What persists isn't the trajectory (the words, the back-and-forth) — it's where you ended up. Your world model updated. That update is what you "remember."

**Language as semantic scaffolding:** The conversation was the link-forming process, not a thing to be stored. The object or situation it was about gains new associations. The scaffolding gets discarded once the structure is built.

**The telephone wire metaphor applies:** You don't remember the electrical signals that carried a phone call. You remember what was communicated. The conversation is the wire; the meaning is what travels through it.

**Why we remember *that* but not *what*:** We often recall that a conversation happened (temporal marker, emotional tone, who was present) without remembering what was said. The perceptual anchors persist because they're grounded in stable embeddings. The linguistic content was always disposable.

**Implication for Bernard:** May not need to store conversations at all — just the associative updates they produce. Episode memory captures "something happened here, involving these objects, with this emotional register." The specific words were never the point.

#### Intensity as Dwell Time

Two triggers encourage long-term retention of conversational information:

- **Emotion** — involuntary intensity boost
- **Intentional focus** — voluntary intensity boost ("trying to remember")

Both modulate the same thing: how strongly associations get written. The difference is what's driving the gain knob.

**Architectural mechanism:** This might just be dwell time in the encoding process. High-emotion moments capture attention longer. "Trying to remember" is literally holding something in active processing longer. Both result in more encoding cycles, stronger initial signal, more resistance to decay.

No "importance" flag needed. The encoding process just needs to be sensitive to how long something stays active. The rest follows.

**Phenomenological alignment:** "Trying to remember" feels like effortful attention because it *is* effortful attention. The mechanism matches the experience. We're not programming outcomes — we're designing process, and the behavior emerges.

---

### 2026-01-31 — Self-Supervised Speech Learning: Closing the Acoustic Loop

**Core concept:** Use speaker and microphone together so the system can hear its own output and learn to produce speech through self-correction, the same way children do.

**The child's calibration process:**

Children talk to themselves constantly during language acquisition. This isn't noise — it's instrument tuning:

1. Hear target sound (parent's speech, environmental audio)
2. Attempt to produce similar sound (vocal cord + mouth = motor control)
3. Hear own output through same auditory pathway
4. Error signal = difference between target and produced sound
5. Iterate until production matches target

The closed loop allows self-supervised learning of speech production without explicit instruction.

**Application to Bernard:**

- **Input:** Target audio (movies, speech samples, user's voice)
- **Output:** Attempted reproduction via speaker
- **Self-monitoring:** Microphone captures own output
- **Comparison:** In embedding space, not text — acoustic patterns compared directly
- **Training:** Error signal refines production pathway

This is learning to speak the way babies do. No text-to-speech. Sound pattern → semantic meaning → motor command → sound pattern. The loop closes in acoustic-semantic space.

**Key insight:** The system needs to hear itself. Without self-monitoring, there's no error signal for production quality. With it, speech emerges from the same predict-compare-correct loop that drives all the other learning.

**Extension: Directional hearing**

Multiple microphones enable spatial audio processing:

- Sound source localization (where is this coming from?)
- Selective attention (focus on one source among many)
- Speaker discrimination (who is talking?)

This adds "where" to "what" — spatial grounding for auditory experience.

**Relation to Semantic Audio JEPA:** This production-loop is the output side of that architecture. Audio JEPA learns to understand; this loop learns to produce. Together they complete the acoustic pathway without text bottleneck.

---

### 2026-01-31 — Memory Palace as Learned Retrieval Structure

**Core concept:** Use memory palace technique as a trainable navigation layer over the associative map. Associations form bottom-up from co-occurrence (unchanged), but a secondary spatial scaffold learns efficient retrieval pathways. Hybrid of semantic flexibility and structured indexing.

**Key insight:** The palace isn't where memories live — it's a learned index that makes traversal reliable. You're not changing what's stored, you're training the retrieval pathways until they fire automatically. "Greasing the groove" for memory access.

**How it fits the architecture:**

- **Associative map:** learns what memories relate to what (from object co-occurrence)
- **Palace layer:** learns "from this region of perception-space, this location is the entry point"
- Memories can be stored however they naturally form, but retrieval conforms to learned structure
- Gives efficiency of database indexing without imposing rigid structure on storage

**"As within, so without" consistency:** Uses spatial structure to organize internal experience the same way spatial structure organizes external perception. The inward JEPA already models memory-to-memory relationships — the palace adds a navigation topology trained for reliable access.

**Analogy:** Human doesn't consciously remember where "eggs" lives in their mental palace — they just go there because the path is worn in. The retrieval becomes automatic through use.

**Open questions:**

- How does the palace structure get initialized? (Learned from scratch vs seeded with spatial priors)
- Does the palace topology emerge naturally from usage patterns or need explicit training signal?
- Relationship to always-on priming — does priming create the "worn paths"?

**Status:** Early idea, architecturally compatible. Worth exploring after core associative map is functional.

---

## Shelved Ideas

*Ideas that were considered but set aside — with reasoning for why.*

<!-- Entries in chronological order, oldest first -->

*(No entries yet)*

---

## Raw Notes / Inbox

*Unprocessed thoughts to be sorted later.*

*(Empty)*
