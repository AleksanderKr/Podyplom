import csv
import itertools
import os
import argparse
from collections import defaultdict

DATA_DIR = "data"
OUT_DIR = "out"

def read_transactions_long(path):
    by_tid = defaultdict(set)
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            by_tid[row["transaction_id"]].add(row["item"])
    return [frozenset(v) for v in by_tid.values()]

def read_mapping(path):
    mapping = {}
    if not os.path.exists(path):
        return mapping
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            mapping[row["item_id"]] = row["item_name"]
    return mapping

def read_itemsets(path):
    itemsets = {}
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            items = tuple(row["itemset"].split())
            itemsets[items] = int(row["support_count"])
    return itemsets

def all_nonempty_proper_subsets(items):
    n = len(items)
    for k in range(1, n):
        for comb in itertools.combinations(items, k):
            yield comb

def support_count_memo(itemset_tuple, transactions, memo):
    if itemset_tuple in memo:
        return memo[itemset_tuple]
    s = set(itemset_tuple)
    c = 0
    for t in transactions:
        if s.issubset(t):
            c += 1
    memo[itemset_tuple] = c
    return c

def apply_mapping_str(items, mapping):
    return ", ".join(mapping.get(x, x) for x in items)

def write_rules(path, rules, human):
    def fmt_side(s: str) -> str:
        if not human:
            return s
        parts = [p.strip() for p in s.split(",") if p.strip()]
        return " + ".join(parts)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)

        if not human:
            w.writerow([
                "rule",
                "antecedent", "consequent",
                "support", "confidence", "lift",
                "support_count_xy", "support_count_x", "support_count_y"
            ])
        else:
            w.writerow([
                "rule",
                "antecedent_human", "consequent_human",
                "support", "confidence", "lift",
                "support_count_xy", "support_count_x", "support_count_y"
            ])

        for row in rules:
            ant = fmt_side(row[0])
            cons = fmt_side(row[1])
            rule = f"{ant} \u2192 {cons}"  # strzałka →
            w.writerow([rule, ant, cons, *row[2:]])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transactions", default=os.path.join(DATA_DIR, "transactions.csv"))
    ap.add_argument("--mapping", default=os.path.join(DATA_DIR, "mapping.csv"))
    ap.add_argument("--itemsets", default=os.path.join(OUT_DIR, "frequent_itemsets.csv"))
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--min-conf", type=float, default=0.60)
    ap.add_argument("--min-lift", type=float, default=1.20)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    itemsets = read_itemsets(args.itemsets)
    transactions = read_transactions_long(args.transactions)
    mapping = read_mapping(args.mapping)
    n_tx = len(transactions)

    memo = {}
    rules = []

    for items, sc_xy in itemsets.items():
        if len(items) < 2:
            continue

        items_sorted = tuple(sorted(items))
        for X in all_nonempty_proper_subsets(items_sorted):
            Y = tuple(sorted(set(items_sorted) - set(X)))

            sc_x = itemsets.get(tuple(sorted(X)))
            if sc_x is None:
                sc_x = support_count_memo(tuple(sorted(X)), transactions, memo)

            sc_y = itemsets.get(tuple(sorted(Y)))
            if sc_y is None:
                sc_y = support_count_memo(tuple(sorted(Y)), transactions, memo)

            if sc_x == 0 or sc_y == 0:
                continue

            support = sc_xy / n_tx
            confidence = sc_xy / sc_x
            sup_y = sc_y / n_tx
            lift = confidence / sup_y if sup_y > 0 else 0.0

            if confidence >= args.min_conf and lift >= args.min_lift:
                rules.append((
                    " ".join(X), " ".join(Y),
                    f"{support:.6f}", f"{confidence:.6f}", f"{lift:.6f}",
                    str(sc_xy), str(sc_x), str(sc_y)
                ))

    rules.sort(key=lambda r: float(r[4]), reverse=True)

    out_rules = os.path.join(args.out_dir, "rules.csv")
    write_rules(out_rules, rules, human=False)

    if mapping:
        out_rules_h = os.path.join(args.out_dir, "rules_human.csv")
        human_rules = []
        for r in rules:
            X = r[0].split()
            Y = r[1].split()
            human_rules.append((
                apply_mapping_str(X, mapping),
                apply_mapping_str(Y, mapping),
                r[2], r[3], r[4], r[5], r[6], r[7]
            ))
        write_rules(out_rules_h, human_rules, human=True)

    print(f"OK: wygenerowano {len(rules)} reguł (min_conf={args.min_conf}, min_lift={args.min_lift}). Zapisano: {out_rules}")

if __name__ == "__main__":
    main()
