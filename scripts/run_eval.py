"""
scripts/run_eval.py
======================

Runs the golden ticket set through the REAL pipeline wiring and reports
field-level accuracy next to cost per ticket. Exits non-zero below
--min-accuracy, so it can gate CI exactly like Week 5's regression gate.

IMPORTANT — what a pass means:
- In DEMO_MODE the "model" is a keyword rule set, so this validates the
  PLUMBING (channels, retrieval, validation, cost accounting), not extraction
  quality. The default threshold (0.5) is deliberately low for that reason.
- With real providers, tickets marked "demo_only" (voice/image scenarios
  that need synthetic media) are skipped, and you should raise --min-accuracy
  (e.g. 0.85) to a level your own baseline supports.

Usage (from the project root):
    DEMO_MODE=1 python scripts/run_eval.py
"""

import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmrag.bootstrap import build_engine  # noqa: E402
from mmrag.config import Settings  # noqa: E402
from mmrag.demo_media import demo_png_for_scenario  # noqa: E402
from mmrag.evaluation import aggregate_scores, score_record  # noqa: E402
from mmrag.pipeline import TicketInput  # noqa: E402


async def run(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    engine = await build_engine(settings)
    with open(args.golden, encoding="utf-8") as handle:
        golden = json.load(handle)

    all_scores = []
    for entry in golden:
        if entry.get("demo_only") and not settings.demo_mode:
            continue
        ticket = TicketInput(
            text=entry.get("text"),
            audio=entry["voice_transcript"].encode("utf-8") if "voice_transcript" in entry else None,
            image=demo_png_for_scenario(entry["image_scenario"]) if "image_scenario" in entry else None,
            ticket_id=f"eval-{entry['id']}",
        )
        result = await engine.pipeline.process(ticket)
        scores = score_record(result.record, entry["expected"])
        all_scores.append(scores)
        wrong = [name for name, ok in scores.items() if not ok]
        print(f"{entry['id']:>4}  {result.status:<19} cost=${result.cost_usd:.6f}  wrong={wrong or '-'}")

    summary = aggregate_scores(all_scores)
    metrics = engine.api.metrics()
    print("\n=== Evaluation ===")
    print(f"tickets: {summary['tickets']}   overall accuracy: {summary['overall']:.1%}")
    for name, accuracy in summary["per_field"].items():
        print(f"  {name:<22} {accuracy:.1%}")
    print(f"avg cost/ticket: ${metrics['avg_cost_per_ticket_usd']:.6f}   p95 latency: {metrics['p95_latency_ms']:.0f} ms")
    if settings.demo_mode:
        print("(DEMO MODE: costs are synthetic and the 'model' is a rule set — this checks plumbing, not quality.)")

    passed = summary["overall"] >= args.min_accuracy
    print(f"\nRESULT: {'PASS' if passed else 'FAIL'} (threshold {args.min_accuracy:.0%})")
    return 0 if passed else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate extraction accuracy and cost on the golden set.")
    parser.add_argument("--golden", default="data/golden_tickets.json")
    parser.add_argument("--min-accuracy", type=float, default=0.5)
    raise SystemExit(asyncio.run(run(parser.parse_args())))


if __name__ == "__main__":
    main()
