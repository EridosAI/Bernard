# test_intent.py - Phase 3: Test intent classification
"""
Tests intent classification in two modes:

1. Unit tests (no hardware needed):
    python tests/test_intent.py --unit

2. Live mic test (full pipeline):
    python tests/test_intent.py

Unit tests verify classify_intent() against known phrases.
Live test runs the full VAD -> Whisper -> Intent pipeline.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.continuous_listening import classify_intent, Intent


# =============================================================================
# UNIT TESTS
# =============================================================================

def run_unit_tests() -> bool:
    """Test classify_intent against known phrases. Returns True if all pass."""

    test_cases = [
        # (input_transcript, expected_intent, expected_name_or_None)

        # --- CONFIRM ---
        ("yes", Intent.CONFIRM, None),
        ("Yeah", Intent.CONFIRM, None),
        ("yep", Intent.CONFIRM, None),
        ("Right", Intent.CONFIRM, None),
        ("correct", Intent.CONFIRM, None),
        ("That's right", Intent.CONFIRM, None),
        ("yup", Intent.CONFIRM, None),

        # --- IDENTIFY ---
        ("What's this?", Intent.IDENTIFY, None),
        ("What is this?", Intent.IDENTIFY, None),
        ("What's that?", Intent.IDENTIFY, None),
        ("What is that?", Intent.IDENTIFY, None),
        ("Whats this", Intent.IDENTIFY, None),

        # --- TEACH ---
        ("This is a soldering iron", Intent.TEACH, "soldering iron"),
        ("This is called a multimeter", Intent.TEACH, "multimeter"),
        ("That's a Phillips head", Intent.TEACH, "phillips head"),
        ("This is my oscilloscope", Intent.TEACH, "oscilloscope"),
        ("That's a wire stripper", Intent.TEACH, "wire stripper"),
        ("It's called a desoldering pump", Intent.TEACH, "desoldering pump"),

        # --- CORRECT ---
        ("No, that's a flathead", Intent.CORRECT, "flathead"),
        ("Actually, it's a resistor", Intent.CORRECT, "resistor"),
        ("No that's my Phillips screwdriver", Intent.CORRECT, "phillips screwdriver"),
        ("Nope, that's a capacitor", Intent.CORRECT, "capacitor"),

        # --- IGNORE ---
        ("The weather is nice today", Intent.IGNORE, None),
        ("I need to grab some lunch", Intent.IGNORE, None),
        ("Hello there", Intent.IGNORE, None),
    ]

    print("=" * 60)
    print("Intent Classification Unit Tests")
    print("=" * 60 + "\n")

    passed = 0
    failed = 0

    for transcript, expected_intent, expected_name in test_cases:
        actual_intent, actual_name = classify_intent(transcript)

        intent_ok = actual_intent == expected_intent
        # For name comparison, normalize both to lowercase
        if expected_name is None:
            name_ok = actual_name is None or actual_name == expected_name
        else:
            name_ok = (actual_name is not None and
                       actual_name.lower() == expected_name.lower())

        if intent_ok and name_ok:
            passed += 1
            status = "PASS"
        else:
            failed += 1
            status = "FAIL"
            print(f"  {status}: \"{transcript}\"")
            print(f"         expected: {expected_intent.value}, name={expected_name}")
            print(f"         got:      {actual_intent.value}, name={actual_name}")
            print()

    print(f"\nResults: {passed}/{passed + failed} passed", end="")
    if failed > 0:
        print(f" ({failed} FAILED)")
    else:
        print(" - ALL PASS")

    return failed == 0


# =============================================================================
# LIVE MIC TEST
# =============================================================================

def run_live_test():
    """Full pipeline: VAD -> Whisper -> Intent classification via mic."""
    import queue
    import whisper
    from src.core.continuous_listening import ContinuousListener

    print("=" * 60)
    print("Intent Classification Live Test - Phase 3")
    print("=" * 60)

    # Load Whisper
    print("\nLoading Whisper base model...")
    whisper_model = whisper.load_model("base")
    print("Whisper loaded")

    # Event queue
    event_queue = queue.Queue()

    # Create listener
    listener = ContinuousListener(
        whisper_model=whisper_model,
        event_queue=event_queue,
        vad_threshold=0.5,
        silence_duration=0.8
    )

    listener.start()

    print("\n" + "=" * 60)
    print("Listening... Speak to test intent classification")
    print()
    print("Try phrases like:")
    print("  \"What's this?\"        -> IDENTIFY")
    print("  \"This is a hammer\"    -> TEACH")
    print("  \"No, that's a wrench\" -> CORRECT")
    print("  \"Yes\" / \"Correct\"     -> CONFIRM")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    event_count = 0

    try:
        while True:
            try:
                event = event_queue.get(timeout=0.5)
                event_count += 1
                print(f"  [{event.intent.value.upper():>8}] \"{event.transcript}\"")
                if event.extracted_name:
                    print(f"           -> object: {event.extracted_name}")
            except queue.Empty:
                pass

    except KeyboardInterrupt:
        print(f"\n\nStopping... ({event_count} events captured)")
        listener.stop()
        print("Done")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    if "--unit" in sys.argv:
        success = run_unit_tests()
        sys.exit(0 if success else 1)
    elif "--live" in sys.argv:
        run_live_test()
    else:
        # Default: run unit tests first, then offer live test
        success = run_unit_tests()
        if success:
            print("\n" + "-" * 60)
            response = input("\nUnit tests passed. Run live mic test? [y/N] ")
            if response.strip().lower() in ("y", "yes"):
                run_live_test()
        else:
            print("\nFix unit test failures before running live test.")
            sys.exit(1)
