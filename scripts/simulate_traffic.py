"""
scripts/simulate_traffic.py
==============================

Feeds a running server a stream of mixed tickets (typed text, "voice", and
images, alone and combined) so the live dashboard at /dashboard has something
to show. Start the server first:

    DEMO_MODE=1 uvicorn --factory mmrag.main:create_app --port 8000
    python scripts/simulate_traffic.py --count 30 --interval 0.4

Demo-mode conventions (see mmrag/providers/fakes.py):
- "audio" is the transcript text sent as bytes, or a TRANSCRIPT:n marker payload;
- images are tiny PNGs whose WIDTH selects a canned scenario (scenario 3
  contains an injection attempt, so you will see "injection_flagged" rows).

Against a REAL deployment (DEMO_MODE off), use real files with curl instead:
    curl -F "text=..." -F "image=@photo.jpg" -F "audio=@call.wav" http://localhost:8000/tickets
"""

import argparse
import json
import os
import random
import sys
import time

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmrag.demo_media import demo_png_for_scenario  # noqa: E402

VOICE_LINES = [
    "Hi, my order ORD-1042 arrived with a cracked screen and I would like a refund please, it is urgent.",
    "I was charged twice for order ORD-2201. Can someone fix my billing?",
    "I cannot log in to my account, it keeps saying locked. This is really frustrating.",
    "The box for ORD-3310 was crushed during delivery, please send a replacement.",
]


def load_text_tickets(path: str) -> list[str]:
    with open(path, encoding="utf-8") as handle:
        return [e["text"] for e in json.load(handle) if "text" in e]


def main() -> None:
    parser = argparse.ArgumentParser(description="Send synthetic tickets to a running server.")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--interval", type=float, default=0.5, help="seconds between tickets")
    parser.add_argument("--golden", default="data/golden_tickets.json")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    texts = load_text_tickets(args.golden)
    kinds = ["text", "text", "voice", "image", "image+text", "voice+image"]

    with httpx.Client(timeout=30.0) as client:
        for i in range(args.count):
            kind = rng.choice(kinds)
            data: dict = {}
            files: dict = {}
            if "text" in kind:
                data["text"] = rng.choice(texts)
            if "voice" in kind:
                files["audio"] = ("call.wav", rng.choice(VOICE_LINES).encode("utf-8"), "audio/wav")
            if "image" in kind:
                files["image"] = ("photo.png", demo_png_for_scenario(rng.randrange(4)), "image/png")

            response = client.post(f"{args.url}/tickets", data=data or None, files=files or None)
            body = response.json()
            record = body.get("record") or {}
            print(
                f"[{i + 1:>3}/{args.count}] {kind:<12} -> {body.get('status'):<19} "
                f"cost=${body.get('cost_usd', 0):.6f} category={record.get('category')} flags={body.get('flags')}"
            )
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
