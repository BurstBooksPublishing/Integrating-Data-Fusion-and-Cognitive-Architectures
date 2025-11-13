#!/usr/bin/env python3
"""
Compute daily burn rate and forecast closure date.
JSON file format: list of items with fields:
  id, created (ISO), closed (ISO or null), severity (1..5), owner
"""
import json,sys
from datetime import datetime, timedelta

def parse_iso(s): return datetime.fromisoformat(s) if s else None

with open(sys.argv[1]) as f: items=json.load(f)  # e.g., \lstinline|proof_debt.json|
today=datetime.utcnow()
# compute daily closures over last 30 days
closed_dates=[]
open_count=0
for it in items:
    closed=parse_iso(it.get('closed'))
    if closed: closed_dates.append(closed.date())
    else: open_count+=1

# burn rate = average closures/day over last 30 days
from collections import Counter
cnt=Counter(closed_dates)
days=30
window=[today.date()-timedelta(days=i) for i in range(days)]
closures=sum(cnt[d] for d in window)
r = closures / days if days>0 else 0.0
print(f"Open items: {open_count}, closures/last{days}d: {closures}, burn_rate/day: {r:.2f}")
if r>0:
    days_left = open_count / r
    eta = today + timedelta(days=days_left)
    print(f"Forecast closure in {days_left:.1f} days, ETA: {eta.date()}")
else:
    print("Burn rate zero; escalate ownership and re-prioritize.")
# minimal actionable outputs written to stdout for dashboards