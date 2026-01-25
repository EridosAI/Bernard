# Prediction as Learning

*How the gap between expectation and reality becomes the training signal.*

---

## The Core Idea

Learning doesn't require labels. It requires prediction error.

When you predict what will happen and something different happens, that gap is information. It tells you your model of the world is wrong in a specific way. Correct the model, and you've learned.

This is how Arnold learns — not from curated datasets with human annotations, but from the continuous stream of predictions and outcomes that come from living in the workshop.

---

## Two Directions of Learning

### Forward: Prediction

```
Predict → Wait → Observe → Compare
```

I'm about to plug in a power supply. Arnold predicts what will happen — the LED will light up, the fan will spin, nothing will explode.

Then reality arrives.

If the prediction matches, the model is confirmed. If it doesn't, the error becomes a training signal. The model adjusts.

This is standard predictive learning. JEPA is built for this.

### Backward: Attribution

```
Observe surprise → Search for cause → Hypothesize → Get feedback
```

Something unexpected happens — a component fails, a measurement is wrong, a tool isn't where it should be.

Arnold searches recent memories for plausible causes. What happened before this? What objects are involved? What patterns match?

It forms a hypothesis: "Would this be explained by the unprotected wire from Tuesday?"

Then I respond:
- "Yes, exactly" → Strong confirmation, boost that association
- "No, it's because X" → New link formed, hypothesis gets slight negative weight
- "Sort of, but..." → Partial credit, nuanced update

Both directions strengthen the same underlying model. Forward prediction runs causality forward in time; backward attribution runs it in reverse.

---

## Why Vocalization Matters

There's a temptation to do attribution silently — form a hypothesis, check it internally, update quietly.

But vocalization is powerful.

When Arnold speaks its hypothesis out loud, it commits to a specific answer *before* getting feedback. This commitment makes the learning signal stronger.

The same mechanism is documented in human learning: students who commit to an answer before seeing the solution learn faster than passive observers. The act of commitment creates sharper error signals.

It also gives me a window into the system's reasoning. When Arnold says "I think this happened because X," I see how it's building causal models. Transparency comes free.

---

## The Developmental Arc

Early stage:
> "Something unexpected happened. I don't have a guess."

The system notices prediction errors but can't yet form causal hypotheses. It's learning *that* things have causes, not *what* causes them.

Middle stage:
> "Would this be explained by [recent event]?"

Hypotheses form, often wrong. But each correction teaches something. The system is learning the structure of causality in this specific domain.

Mature stage:
> Handles most attribution internally, only surfaces genuinely ambiguous cases.

Prediction errors are mostly resolved by the model. Only truly novel situations require external input.

---

## No Labels Required

Traditional supervised learning needs someone to annotate: "this image contains a cat." The label is the training signal.

Prediction-based learning generates its own signal. The world provides feedback by simply being what it is. Predict, observe, compare. No human annotator needed.

This is crucial for Arnold's continuous learning. I can't label every frame of video from my workshop. But I don't need to — the system learns from the structure of what happens, not from my descriptions of it.

---

## Connection to Memory

Prediction error doesn't just update the world model — it creates memories.

High prediction error = surprising event = worth remembering.

This is why the NoveltyScorer is central to Arnold's architecture. It detects when something is unexpected, which triggers both:
1. Storage of a new episode (this is worth remembering)
2. Potential learning update (the model needs adjustment)

Low prediction error = expected event = maybe not worth storing. The world is behaving as predicted; no new information.

---

## Example Interaction

```
[I plug in a device; it doesn't turn on]

Arnold: "That's unexpected — I thought it would power on."
         (prediction error detected)

Arnold: "Could this be related to the power supply we tested
          yesterday that was giving inconsistent readings?"
         (hypothesis formed from recent memory)

Me: "Good thinking, but actually I forgot to flip the switch."

Arnold: [updates model: check for simple causes before complex ones]
         [slight negative weight on "faulty power supply" hypothesis]
         [stores this episode as example of user-error pattern]
```

Learning happened at multiple levels:
- World model: switches matter
- Causal reasoning: simple explanations before complex
- Memory: this episode is now available for future attribution

---

## The Elegance

What I find elegant about this:

**Unified signal.** The same mechanism — prediction error — drives perception, memory formation, and causal learning. Not three separate systems, but one.

**Self-supervised.** The world teaches the model. Human input helps (narrative override, corrections) but isn't required for basic learning.

**Developmentally plausible.** This is roughly how children learn. They predict, get surprised, form hypotheses, get corrected, and gradually build causal models of their world.

**Transparent.** Vocalized hypotheses reveal the system's causal reasoning. Debugging happens naturally.

---

## Open Questions

- **Threshold for vocalization:** When should Arnold speak a hypothesis vs. resolve it internally? Too much vocalization is annoying; too little loses the transparency benefit.
- **Hypothesis search:** How does the system efficiently search memory for plausible causes? Current implementation is simple; could be more sophisticated.
- **Negative evidence:** How to learn from "X didn't happen" — the absence of expected events?

---

## Related Ideas

- [Architecture: Dual JEPA](architecture.md) — The structure that enables prediction at multiple levels
- [Post-Linguistic Architecture](post-linguistic.md) — Why prediction happens in semantic space, not language
