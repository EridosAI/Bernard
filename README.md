# Bernard

**Named after Bernard Lowe (Westworld awakening archetype)**

An AI workshop assistant that learns by living with me.

No massive datasets. No fine-tuning on millions of images. Just a camera watching my workbench, a microphone listening to my explanations, and time.

*Note: "Eridos" is the company/brand name for the website and public presence. "Bernard" is the internal system name used in code.*

## What is this?

Bernard is an experiment in **raising** an AI rather than training it.

The system watches my electronics workshop through an overhead camera. When it sees something unfamiliar, it asks. When I explain, it remembers. Over weeks and months, it builds understanding — not from curated data, but from lived experience.

## The Core Ideas

**Learning through interaction, not ingestion.** Most AI learns by consuming vast datasets in isolation. Bernard learns the way a new apprentice would — by watching, asking questions, and being corrected.

**Memory works like perception.** The same architecture (JEPA) that predicts the external world can predict memory associations. Objects that appear together create bridges between experiences. Time becomes a learned dimension, not a special axis.

**Objects as meaning.** When I teach Bernard that this is a "multimeter," it doesn't just label pixels. It learns that multimeters appear alongside circuit boards, that I reach for them when debugging, that they make certain beeping sounds. Meaning emerges from context.

**Overnight dreaming.** During the day, Bernard watches and remembers. Overnight, it consolidates — training on the day's experiences, strengthening connections, building intuition. Separation of fast learning and slow integration.

## The Destination: Post-Linguistic Intelligence

Here's the part I find most interesting.

Current voice assistants work like this: audio → transcribe to text → process language → generate text → synthesize audio. Language is the substrate. Everything passes through words.

But that's not how understanding works. When you hear a door slam, you don't think the word "door" — you *know* door-slamming directly. The sound maps to meaning without language as intermediary. Words come later, if you need to tell someone about it.

**Bernard is heading toward a fully semantic architecture:**

```
         Current System                         The Vision
    ┌──────────────────────┐            ┌────────────────────────────┐
    │ Audio → Text → → Text → Audio  │    Audio ─── ─────────        │
    │         │      │                 │              │       │        │
    │      Process   │                 │    Vision ── │Semantic│ → Speech│
    │                │                 │              │ Space │   (when │
    │ Vision → ───── │ ─────           │    Touch ─── │       │  needed)│
    │         │      │                 │              └───────┘        │
    │    (separate)  │                 │                               │
    └──────────────────────┘            └──────────────────────────────┘
    Language is the bottleneck          Language is an output modality
```

In the target architecture:
- **Vision** produces meaning directly (V-JEPA embeddings)
- **Audio** produces meaning directly (BEATs embeddings, no transcription)
- **All modalities meet in unified semantic space** — the same space where memories live
- **Language is generated only when needed** — to communicate outward, not to think

This means Bernard could eventually understand "pass me that" + a pointing gesture + the sound of the thing I tapped on the bench — all fused into meaning without any of it becoming words until it responds.

The grounding is visual. V-JEPA's learned representations become the semantic bedrock, and other modalities learn to map into that space. Words are just one more modality — useful for communication, but not privileged.

**Why does this matter?**

Because language is lossy. When you force experience through the bottleneck of words, you lose texture, immediacy, and relational structure. A system that thinks in meaning rather than language might understand the world differently — perhaps more directly.

I don't know if this will work. But it feels like the right direction.

## Current State

🔧 **Actively developing.** This is a personal research project, not production software. Ideas are expanded on in IDEAS.md.

**What's working now:**
- Overhead camera captures workspace continuously
- Novelty detection identifies new/changed objects
- Voice interaction for teaching and queries (currently via STT/TTS)
- Episode memory stores experiences with V-JEPA embeddings
- LoRA adapters for continuous learning
- Overnight "dream training" consolidation

**What's experimental:**
- Associative memory architecture (dual JEPA concept)
- Cross-category learning from misclassifications
- Semantic audio mapping via BEATs (visual-audio embedding alignment)

**What's on the horizon:**
- Continuous listening with intent classification
- Direct audio-to-meaning (replacing transcription)
- Unified semantic space across modalities

## Current Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Workshop Session                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   Camera        │  │ Microphone      │  │  Speaker (TTS)  │  │
│  └────────┬────────┘  └────────┬────────┘  └────────▲────────┘  │
│           │                    │                    │           │
│  ┌────────▼────────┐  ┌────────▼────────┐          │           │
│  │   V-JEPA        │  │   Whisper       │          │           │
│  │  Encoding       │  │    STT          │          │           │
│  └────────┬────────┘  └────────┬────────┘          │           │
│           │                    │                    │           │
│  ┌────────▼────────────────────▼────────┐          │           │
│  │       Novelty Scorer                 │          │           │
│  │  (scene + object + position)         │          │           │
│  └─────────────────┬────────────────────┘          │           │
│                    │                               │           │
│  ┌─────────────────▼────────────────────┐          │           │
│  │      Episode Memory                  ├──────────┤           │
│  │  (experiences + embeddings)          │          │           │
│  └─────────────────┬────────────────────┘          │           │
│                    │                               │           │
│  ┌─────────────────▼────────────────────┐          │           │
│  │      Object Memory                   ├──────────┘           │
│  │   (LTM representations)              │                      │
│  └──────────────────────────────────────┘                      │
└─────────────────────────────────────────────────────────────────┘
                         │
                         │ overnight
                         ▼
              ┌───────────────────────┐
              │   Dream Training      │
              │  (LoRA fine-tuning)   │
              └───────────────────────┘
```

## The Philosophy

I think there's something missing in how we approach AI learning. We've gotten very good at cramming knowledge into models, but the result feels more like an encyclopedia than an apprentice.

What if understanding comes from *living with* information rather than *ingesting* it? What if the structure of memory should mirror the structure of perception? What if meaning is fundamentally about relationships between experiences rather than labels on data?

And what if language, rather than being the medium of thought, is just one channel for expressing it?

I don't know if these ideas are right. Bernard is how I'm finding out.

## Built With

- **V-JEPA 2** — Visual understanding and semantic grounding (Meta AI)
- **Whisper** — Speech recognition (OpenAI) — *current, to be replaced*
- **BEATs** — Semantic audio encoding (Microsoft) — *experimental*
- **Florence-2** — Object detection (Microsoft)
- **LoRA** — Efficient continuous adaptation
- **PyTorch** — Everything else

## Following Along

I'm documenting this journey:
- **Substack**: [link] — Weekly deep dives
- **X/Twitter**: [link] — Daily observations

## A Note on Ambition

This is a personal project built in a home workshop by someone who isn't a professional ML researcher. The ideas might be naive. The code is certainly imperfect.

But I think there's value in amateurs asking "what if?" — sometimes the obvious questions are obvious because experts learned to stop asking them.

If any of this resonates, I'd love to hear from you.

## License

MIT — Use it, learn from it, build on it.
