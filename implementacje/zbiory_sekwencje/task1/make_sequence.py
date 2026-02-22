import csv
import os
import argparse
from collections import defaultdict

DATA_DIR = "data"


def item_key(x: str):
    if x.startswith("i") and x[1:].isdigit():
        return int(x[1:])
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transactions", default=os.path.join(DATA_DIR, "transactions.csv"))
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "sequences.csv"))
    ap.add_argument("--order", choices=["item_id", "as_is"], default="item_id")
    args = ap.parse_args()

    by_tid = defaultdict(list)
    with open(args.transactions, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            by_tid[row["transaction_id"]].append(row["item"])

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sequence_id", "pos", "item"])

        for tid in sorted(by_tid.keys(), key=lambda t: int(t[1:]) if t.startswith("t") and t[1:].isdigit() else t):
            items = by_tid[tid]
            if args.order == "item_id":
                items = sorted(set(items), key=item_key)
            else:
                seen = set()
                tmp = []
                for it in items:
                    if it not in seen:
                        seen.add(it)
                        tmp.append(it)
                items = tmp

            for i, it in enumerate(items, start=1):
                w.writerow([tid, i, it])

    print(f"OK: zapisano {args.out}")


if __name__ == "__main__":
    main()
