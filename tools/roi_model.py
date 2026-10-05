"""Break-even model for a small factory: how many bad lots must the station stop to pay for itself?

EVERY INPUT IS AN ASSUMPTION, not a finding. Replace them with figures you can source (a hardware
price, a customer's chargeback terms, your own measured detection rate) before presenting.

  python tools/roi_model.py
  python tools/roi_model.py --catch-rate 0.9 --cost-per-lot 30000
"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def required_escapes(system_cost_year, cost_per_returned_lot, catch_rate):
    """Defective lots per year that must be reaching customers today for the station to break even."""
    return system_cost_year / (cost_per_returned_lot * catch_rate)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system-costs", type=float, nargs="+", default=[20000, 40000, 80000],
                    help="assumed yearly cost of one inspection station (hardware amortised + support)")
    ap.add_argument("--cost-per-lot", type=float, nargs="+", default=[5000, 20000, 50000],
                    help="assumed total cost of ONE returned lot (credit, re-sort, freight, lost orders)")
    ap.add_argument("--catch-rate", type=float, default=0.80,
                    help="share of today's escaping defective lots the station would stop (replace with measured)")
    ap.add_argument("--out", default=str(ROOT / "results"))
    a = ap.parse_args()

    rows = []
    print(f"ASSUMPTION: the station stops {a.catch_rate:.0%} of the defective lots that escape today\n")
    print("system cost/yr".ljust(16) + "".join(f"cost per returned lot {c:>7,.0f}".rjust(34) for c in a.cost_per_lot))
    for sc in a.system_costs:
        line = f"{sc:>13,.0f}   "
        for c in a.cost_per_lot:
            n = required_escapes(sc, c, a.catch_rate)
            line += f"{n:>8.1f} lots/yr ({n / 12:>5.2f}/month)".rjust(34)
            rows.append({"system_cost_year": sc, "cost_per_returned_lot": c, "catch_rate": a.catch_rate,
                         "escaped_lots_needed_per_year": round(n, 2), "per_month": round(n / 12, 2)})
        print(line)
    print("\nRead: if MORE defective lots than this reach customers each year, the station pays for itself.")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "roi_breakeven.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "roi_breakeven.json").write_text(json.dumps({"assumptions": vars(a), "rows": rows}, indent=2),
                                            encoding="utf-8")


if __name__ == "__main__":
    main()
