#!/usr/bin/env python3
"""Inject the project's model-development guardrails into each Cursor session."""

import json
import sys


def main() -> None:
    # Consume the event payload so malformed input cannot leak into output.
    try:
        json.load(sys.stdin)
    except (json.JSONDecodeError, OSError):
        pass

    context = (
        "Infinitic AI Urdu headline classification project: maintain two honest "
        "tracks—(A) an Urdu tokenizer and transformer initialized and trained "
        "from scratch, and (B) a pretrained multilingual transformer fine-tuned "
        "for classification. Follow the project skill and phase gates. Split "
        "and deduplicate before modeling, select on validation macro-F1, keep "
        "the test set sealed until a finalist is declared, record every run, "
        "and never invent results."
    )
    json.dump({"additional_context": context}, sys.stdout)


if __name__ == "__main__":
    main()
