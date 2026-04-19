import csv
import os
import argparse
from collections import defaultdict

DATA_DIR = "data"
OUT_DIR = "out"


def ensure_dirs(out_dir):
    os.makedirs(out_dir, exist_ok=True)


def read_transactions_long(path):
    by_tid = defaultdict(list)
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            by_tid[row["transaction_id"]].append(row["item"])
    transactions = []
    for items in by_tid.values():
        uniq = sorted(set(items), key=item_key)
        if uniq:
            transactions.append(uniq)
    return transactions


def read_mapping(path):
    mapping = {}
    if not os.path.exists(path):
        return mapping
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            mapping[row["item_id"]] = row["item_name"]
    return mapping


def item_key(x: str):
    if x.startswith("i") and x[1:].isdigit():
        return int(x[1:])
    return x


class FPNode:
    __slots__ = ("item", "count", "parent", "children", "link")

    def __init__(self, item, parent):
        self.item = item
        self.count = 0
        self.parent = parent
        self.children = {}
        self.link = None


def build_fptree(transactions, min_sup_count):
    freq = defaultdict(int)
    for t in transactions:
        for it in t:
            freq[it] += 1

    freq = {it: c for it, c in freq.items() if c >= min_sup_count}
    if not freq:
        return None, None, None

    order = sorted(freq.items(), key=lambda kv: (-kv[1], item_key(kv[0])))
    rank = {it: i for i, (it, _) in enumerate(order)}

    header = {it: [freq[it], None] for it in freq.keys()}
    root = FPNode(None, None)

    def link_node(item, node):
        head = header[item][1]
        if head is None:
            header[item][1] = node
        else:
            cur = head
            while cur.link is not None:
                cur = cur.link
            cur.link = node

    for t in transactions:
        items = [it for it in t if it in freq]
        if not items:
            continue
        items.sort(key=lambda it: rank[it])

        cur = root
        for it in items:
            nxt = cur.children.get(it)
            if nxt is None:
                nxt = FPNode(it, cur)
                cur.children[it] = nxt
                link_node(it, nxt)
            nxt.count += 1
            cur = nxt

    return root, header, rank


def ascend_path(node):
    path = []
    cur = node.parent
    while cur is not None and cur.item is not None:
        path.append(cur.item)
        cur = cur.parent
    path.reverse()
    return path


def build_conditional_transactions(header_item_node):
    cond = []
    node = header_item_node
    while node is not None:
        path = ascend_path(node)
        if path:
            cond.append((path, node.count))
        node = node.link
    return cond


def build_conditional_tree(cond_transactions, min_sup_count):
    freq = defaultdict(int)
    for items, cnt in cond_transactions:
        for it in items:
            freq[it] += cnt

    freq = {it: c for it, c in freq.items() if c >= min_sup_count}
    if not freq:
        return None, None, None

    order = sorted(freq.items(), key=lambda kv: (-kv[1], item_key(kv[0])))
    rank = {it: i for i, (it, _) in enumerate(order)}
    header = {it: [freq[it], None] for it in freq.keys()}

    root = FPNode(None, None)

    def link_node(item, node):
        head = header[item][1]
        if head is None:
            header[item][1] = node
        else:
            cur = head
            while cur.link is not None:
                cur = cur.link
            cur.link = node

    for items, cnt in cond_transactions:
        filtered = [it for it in items if it in freq]
        if not filtered:
            continue
        filtered.sort(key=lambda it: rank[it])

        cur = root
        for it in filtered:
            nxt = cur.children.get(it)
            if nxt is None:
                nxt = FPNode(it, cur)
                cur.children[it] = nxt
                link_node(it, nxt)
            nxt.count += cnt
            cur = nxt

    return root, header, rank


def mine_tree(header, min_sup_count, prefix, out_counts):
    items = sorted(header.items(), key=lambda kv: (kv[1][0], item_key(kv[0])))

    for it, (sup, node) in items:
        new_prefix = prefix + (it,)
        out_counts[new_prefix] = sup

        cond_trans = build_conditional_transactions(node)
        cond_root, cond_header, _ = build_conditional_tree(cond_trans, min_sup_count)
        if cond_header:
            mine_tree(cond_header, min_sup_count, new_prefix, out_counts)


def fpgrowth(transactions, min_sup_count):
    root, header, _ = build_fptree(transactions, min_sup_count)
    if not header:
        return {}

    counts = {}
    mine_tree(header, min_sup_count, tuple(), counts)

    normalised = {}
    for items, sc in counts.items():
        key = tuple(sorted(items, key=item_key))
        normalised[key] = max(normalised.get(key, 0), sc)
    return normalised


def write_itemsets(path, itemsets, n_transactions):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["itemset", "support_count", "support"])
        for items in sorted(itemsets.keys(), key=lambda x: (len(x), tuple(item_key(i) for i in x))):
            sc = itemsets[items]
            sup = sc / n_transactions
            w.writerow([" ".join(items), sc, f"{sup:.6f}"])


def write_itemsets_human(path, itemsets, n_transactions, mapping):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["itemset_human", "support_count", "support"])
        for items in sorted(itemsets.keys(), key=lambda x: (len(x), tuple(item_key(i) for i in x))):
            sc = itemsets[items]
            sup = sc / n_transactions
            human = [mapping.get(it, it) for it in items]
            w.writerow([" + ".join(human), sc, f"{sup:.6f}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transactions", default=os.path.join(DATA_DIR, "transactions.csv"))
    ap.add_argument("--mapping", default=os.path.join(DATA_DIR, "mapping.csv"))
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--min-sup-count", type=int, default=8)
    args = ap.parse_args()

    ensure_dirs(args.out_dir)

    transactions = read_transactions_long(args.transactions)
    mapping = read_mapping(args.mapping)

    itemsets = fpgrowth(transactions, min_sup_count=args.min_sup_count)

    out1 = os.path.join(args.out_dir, "frequent_itemsets.csv")
    write_itemsets(out1, itemsets, len(transactions))

    if mapping:
        out2 = os.path.join(args.out_dir, "frequent_itemsets_human.csv")
        write_itemsets_human(out2, itemsets, len(transactions), mapping)

    print(f"OK: znaleziono {len(itemsets)} częstych zbiorów (min_sup_count={args.min_sup_count})")
    print("Zapisano:", out1)


if __name__ == "__main__":
    main()
