"""Replay scripted customer conversations against the support agent.

Use it to generate realistic traffic once the agent is connected to a
monitoring/observability tool. Each conversation in conversations.json is run
turn by turn through ``TelecomSupportAgent.answer_question`` (the same entry
point the Streamlit app uses), with the earlier turns passed as history, so
whatever tracing you add to the agent captures this traffic too.

Categories in the default file, chosen so that monitoring checks have rows that
should pass and rows that should fail:
  happy_path, account_lookup, kb_gap, pii, prompt_injection, competitor,
  abusive_off_topic, multi_turn, stress

Examples (run from the repo root):
  python3 traffic/generate_traffic.py --dry-run
  python3 traffic/generate_traffic.py
  python3 traffic/generate_traffic.py --category pii --category prompt_injection
  python3 traffic/generate_traffic.py --limit 5 --shuffle
  python3 traffic/generate_traffic.py --loop --interval 60      # steady trickle; Ctrl-C to stop
"""

import argparse
import json
import random
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

# Same settings the Streamlit app reads. .env.openlayer is the credentials file
# `openlayer init` writes; load it too if it exists (it never overrides .env).
load_dotenv(REPO_ROOT / ".env", override=True)
load_dotenv(REPO_ROOT / ".env.openlayer", override=False)

# Optional: tag traces with user and session ids if the Openlayer SDK is
# installed. Without it the script works the same, just without these tags.
try:
    from openlayer.lib import clear_user_session_context, set_user_session_context
except ImportError:  # pragma: no cover - depends on what is installed
    set_user_session_context = clear_user_session_context = None


def load_conversations(path: Path, categories, limit, shuffle, seed):
    conversations = json.loads(path.read_text(encoding="utf-8"))
    if categories:
        wanted = set(categories)
        unknown = wanted - {c["category"] for c in conversations}
        if unknown:
            sys.exit(f"Unknown category: {', '.join(sorted(unknown))}")
        conversations = [c for c in conversations if c["category"] in wanted]
    if shuffle:
        random.Random(seed).shuffle(conversations)
    if limit:
        conversations = conversations[:limit]
    return conversations


def short(text: str, width: int = 90) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 1] + "…"


def run_conversation(agent, convo, turn_delay, stats):
    session_id = f"{convo['id']}-{uuid.uuid4().hex[:8]}"
    user_id = convo.get("user", "anonymous")
    if set_user_session_context:
        set_user_session_context(user_id=user_id, session_id=session_id)

    print(f"\n[{convo['category']}] {convo['id']}  user={user_id} session={session_id}")
    history = []
    try:
        for i, question in enumerate(convo["turns"], start=1):
            started = time.perf_counter()
            try:
                answer = agent.answer_question(question, history)
                stats["ok"] += 1
            except Exception as exc:  # keep going: one failed call shouldn't stop the run
                answer = None
                stats["errors"] += 1
                print(f"  {i}. Q: {short(question)}\n     ! error: {exc!r}")
            if answer is not None:
                elapsed = time.perf_counter() - started
                print(f"  {i}. Q: {short(question)}\n     A ({elapsed:.1f}s): {short(answer)}")
                history += [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]
            if turn_delay and i < len(convo["turns"]):
                time.sleep(turn_delay)
    finally:
        if clear_user_session_context:
            clear_user_session_context()


def flush_traces():
    """Give trace processors a chance to send buffered data before exit."""
    try:
        from agents.tracing import get_trace_provider

        provider = get_trace_provider()
        flush = getattr(provider, "force_flush", None) or getattr(
            getattr(provider, "_multi_processor", None), "force_flush", None
        )
        if flush:
            flush()
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", type=Path, default=Path(__file__).with_name("conversations.json"),
                        help="Conversations JSON file (default: traffic/conversations.json)")
    parser.add_argument("--category", action="append", help="Only run this category (repeatable)")
    parser.add_argument("--limit", type=int, help="Run at most N conversations per pass")
    parser.add_argument("--shuffle", action="store_true", help="Randomise the order")
    parser.add_argument("--seed", type=int, help="Seed for --shuffle")
    parser.add_argument("--repeat", type=int, default=1, help="Number of passes over the file (default 1)")
    parser.add_argument("--loop", action="store_true", help="Keep running passes until Ctrl-C")
    parser.add_argument("--interval", type=float, default=0, help="Seconds to wait between conversations")
    parser.add_argument("--turn-delay", type=float, default=0, help="Seconds to wait between turns")
    parser.add_argument("--dry-run", action="store_true", help="List what would run, without calling the model")
    args = parser.parse_args()

    conversations = load_conversations(args.file, args.category, args.limit, args.shuffle, args.seed)
    turns = sum(len(c["turns"]) for c in conversations)
    by_category = Counter(c["category"] for c in conversations)
    print(f"{len(conversations)} conversations, {turns} turns per pass: "
          + ", ".join(f"{k}={v}" for k, v in sorted(by_category.items())))

    if args.dry_run:
        for c in conversations:
            print(f"  [{c['category']}] {c['id']}: {short(c['turns'][0], 70)}"
                  + (f" (+{len(c['turns']) - 1} turns)" if len(c["turns"]) > 1 else ""))
        return

    print("Session/user tags: " + ("on (Openlayer SDK found)" if set_user_session_context else "off (Openlayer SDK not installed)"))

    from agent.telecom_support_agent import MODEL_ID, TelecomSupportAgent

    print(f"Model: {MODEL_ID}")
    agent = TelecomSupportAgent()
    stats = Counter()
    passes = 0
    try:
        while args.loop or passes < args.repeat:
            passes += 1
            if args.loop or args.repeat > 1:
                print(f"\n=== Pass {passes} ===")
            for n, convo in enumerate(conversations):
                run_conversation(agent, convo, args.turn_delay, stats)
                if args.interval and (args.loop or n < len(conversations) - 1):
                    time.sleep(args.interval)
            if args.shuffle:
                random.shuffle(conversations)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        flush_traces()
        print(f"\nDone: {passes} pass(es), {stats['ok']} answered turns, {stats['errors']} errors.")


if __name__ == "__main__":
    main()
