#!/usr/bin/env python3
"""Weekly knowledge gap report + authority refresh. Cron: Monday 06:00."""
import asyncio, sys
from datetime import datetime
from pathlib import Path

OBSIDIAN_REPORTS = Path.home() / "Library/Mobile Documents/iCloud~md~obsidian/Documents/PARA/!inbox/!reports"

async def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from app.reasoning.knowledge_gaps import generate_gap_report, format_obsidian_report
    report = await generate_gap_report(days=days)
    if report.get("status") == "no_data":
        print(f"No traces in last {days} days"); return

    md = format_obsidian_report(report)
    date_str = datetime.utcnow().strftime("%y%m%d")
    OBSIDIAN_REPORTS.mkdir(parents=True, exist_ok=True)
    (OBSIDIAN_REPORTS / f"{date_str}-cc-knowledge-gaps.md").write_text(md, encoding="utf-8")
    print(f"Report saved. Traces: {report['total_traces']}, Success: {report['success_rate']:.0%}")

    from app.reasoning.authority_learner import refresh_adjustments_cache
    adj = await refresh_adjustments_cache(days=30)
    print(f"Authority: {len(adj)} categories refreshed")

if __name__ == "__main__":
    asyncio.run(main())
