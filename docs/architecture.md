# Architecture: Dual JEPA for Perception and Memory

*How Arnold uses the same architecture to understand the world and navigate its own memories.*

---

## The Core Intuition

What if memory works like perception?

When you perceive the world, you're not receiving raw data — you're running a predictive model. Your brain constantly predicts what it will see next, and the difference between prediction and reality is what you consciously experience. This is roughly what JEPA (Joint Embedding Predictive Architecture) does for vision.

But consider memory. When you recall something, you're not playing back a recording. You're navigating a space of associations — one memory leads to another through learned connections. The *structure* of memory is itself something your brain has learned.

The insight behind Arnold: **the same architecture that learns to predict the external world can learn to predict the structure of internal memory.**

---

## Two JEPAs

### Outward JEPA: World Modeling

This is the standard use of JEPA for perception:

- **Input:** Visual frames from the workshop camera
- **Learns:** Object appearances, spatial relationships, what-follows-what
- **Output:** Rich embeddings that capture scene semantics

When I place a multimeter on my bench, the outward JEPA encodes not just pixels but *meaning* — this object, in this location, in this context.

### Inward JEPA: Memory Association

This is the experimental part:

- **Input:** Memory embeddings (stored experiences)
- **Learns:** Memory-to-memory relationships, what recalls what
- **Output:** Associative predictions — "from this memory, these others are reachable"

The inward JEPA doesn't store memories or rearrange them. It learns what memories *mean to each other*.

---

## The Training Signal

Here's where it gets elegant.

The outward JEPA learns that certain objects appear together — multimeters with circuit boards, soldering irons with solder, etc. These co-occurrences are the natural structure of the workshop.

These same co-occurrences create bridges between memories. If Memory A contains a multimeter and Memory B contains a multimeter, they share an associative link. The object is the bridge.

**The outward JEPA's learned associations become the training signal for the inward JEPA.**

No separate labeling needed. No human-annotated "these memories are related." The structure of the world teaches the structure of memory.

---

## How It Works in Practice

```
1. EXPERIENCE
   Camera captures me debugging a circuit with a multimeter
   → Outward JEPA encodes the scene
   → Stored as Episode #247 with embedding

2. LATER EXPERIENCE
   Camera captures me reaching for the same multimeter
   → Outward JEPA recognizes the object
   → Object co-occurrence links Episode #247 to current moment

3. ASSOCIATION LEARNING
   Inward JEPA observes: these episodes share objects
   → Strengthens associative pathway between their embeddings
   → Over time, learns the general pattern: shared objects = related memories

4. RETRIEVAL
   I say "what was I doing with this yesterday?"
   → Current perception activates relevant region of memory space
   → Inward JEPA predicts which memories are "reachable" from here
   → Episode #247 surfaces
```

---

## Key Properties

### LTM Stays Stable

Memory embeddings don't drift based on associations. The embedding for Episode #247 captures what happened in that moment — it shouldn't change just because I later associated it with something else.

Associations are learned *over* stable embeddings, not *in* them.

### Always-On Priming

Current perception continuously queries the associative map. There's always a "warm set" of memories that are partially activated — relevant to what's happening now, even if not consciously surfaced.

This creates context. When I'm working on a circuit, circuit-related memories are primed. When I switch to woodworking, different memories warm up. The system carries relevant history without explicit retrieval.

### Task-Driven Surfacing

Primed memories only surface to speech/behavior when context demands it. Most of the time, they just inform — creating the background of "knowing" without the foreground of "recalling."

---

## Time as a Learned Feature

Memories aren't stored in temporal order. Time is just another feature the associative map learns — like position or color.

The system discovers that action → consequence patterns have characteristic time-gaps, the same way it discovers that objects have characteristic sizes. A soldering iron burn follows touching the iron by seconds. A dead battery follows leaving something on by hours. The map learns these patterns from experience.

No special temporal machinery. Just learned structure.

---

## Objects as Causal Bridges

How does causation emerge?

From patterns. When "action involving X" is repeatedly followed by "outcome involving X," the associative map learns this pattern. The object present in both memories creates the bridge.

I plug in an unprotected wire (Memory A). Later, something shorts out (Memory B). Both involve the wire. The association forms. With enough similar experiences, the map learns: unprotected wires → shorts.

Causation isn't programmed. It emerges from pattern.

---

## Narrative Override

Sometimes I can shortcut this process by explicitly linking memories:

> "Remember when we plugged that unprotected wire in? Well, now it shorted out."

This boosts an association that already exists (via the shared object) but would otherwise require many repetitions to strengthen.

This is how humans shortcut children's learning — we inject compressed causal knowledge through narrative rather than waiting for them to experience every consequence directly.

---

## Open Questions

- **Implementation details:** The exact architecture of the inward JEPA predictor is still being worked out.
- **Threshold for surfacing:** How does the system decide when primed memories should become conscious/verbalized? Learned threshold? Explicit signals?
- **Consolidation timing:** Should associations update after every episode, or in periodic "dreaming" phases?

---

## Related Ideas

- [Prediction as Learning](prediction-as-learning.md) — How prediction error drives both forward and backward learning
- [Post-Linguistic Architecture](post-linguistic.md) — Why meaning lives in embedding space, not language
