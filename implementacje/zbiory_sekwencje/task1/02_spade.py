import csv
import os
import argparse
from collections import defaultdict

DATA_DIR = "data"
OUT_DIR = "out"

POS_KEYS = ("pos", "position", "event_idx", "idx", "order", "time", "t")


def item_key(x: str):
    if x.startswith("i") and x[1:].isdigit():
        return int(x[1:])
    return x


def read_mapping(path):
    mapping = {}
    if not os.path.exists(path):
        return mapping
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            mapping[row["item_id"]] = row["item_name"]
    return mapping


def read_sequences_long(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        fieldnames = [c.strip() for c in (r.fieldnames or [])]

        if "sequence_id" not in fieldnames:
            raise RuntimeError("Brak kolumny sequence_id w data/sequences.csv")

        pos_key = None
        for k in POS_KEYS:
            if k in fieldnames:
                pos_key = k
                break
        if pos_key is None:
            raise RuntimeError("Brak kolumny pozycji (np. pos / order / time) w data/sequences.csv")

        if "item" not in fieldnames:
            raise RuntimeError("Brak kolumny item w data/sequences.csv")

        by_sid = defaultdict(list)
        for row in r:
            sid = row["sequence_id"]
            pos = int(row[pos_key])
            it = row["item"].strip()
            if not it:
                continue
            by_sid[sid].append((pos, it))

    sequences = []
    for sid, events in by_sid.items():
        events.sort(key=lambda x: x[0])
        seq = [it for _, it in events]
        if seq:
            sequences.append(seq)
    return sequences


def build_idlists(sequences):
    idlists = defaultdict(lambda: defaultdict(list))
    for sid, seq in enumerate(sequences):
        seen = defaultdict(set)
        for pos, it in enumerate(seq, start=1):
            if pos not in seen[it]:
                idlists[it][sid].append(pos)
                seen[it].add(pos)
    for it in idlists:
        for sid in idlists[it]:
            idlists[it][sid].sort()
    return idlists


def support_of_idlist(idlist):
    return len(idlist)


def temporal_join(prefix_idlist, item_idlist):
    out = {}
    for sid, p_positions in prefix_idlist.items():
        i_positions = item_idlist.get(sid)
        if not i_positions:
            continue
        res = []
        j = 0
        for pp in p_positions:
            while j < len(i_positions) and i_positions[j] <= pp:
                j += 1
            if j < len(i_positions):
                res.extend(i_positions[j:])
        if res:
            uniq = sorted(set(res))
            out[sid] = uniq
    return out


def spade(sequences, min_sup_count, max_len=None):
    idlists = build_idlists(sequences)

    items = []
    for it, idl in idlists.items():
        sc = support_of_idlist(idl)
        if sc >= min_sup_count:
            items.append(it)

    items.sort(key=item_key)

    freq = {}
    one = {}
    for it in items:
        one[(it,)] = idlists[it]
        freq[(it,)] = support_of_idlist(idlists[it])

    def dfs(prefix, prefix_idlist):
        if max_len is not None and len(prefix) >= max_len:
            return
        for it in items:
            new_pat = prefix + (it,)
            joined = temporal_join(prefix_idlist, idlists[it])
            sc = support_of_idlist(joined)
            if sc >= min_sup_count:
                freq[new_pat] = sc
                dfs(new_pat, joined)

    for pat, idl in one.items():
        dfs(pat, idl)

    return freq


def write_sequences(path, seq_counts, n_sequences):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sequence", "support_count", "support"])
        for s in sorted(seq_counts.keys(), key=lambda x: (len(x), tuple(item_key(i) for i in x))):
            sc = seq_counts[s]
            sup = sc / n_sequences
            w.writerow([" -> ".join(s), sc, f"{sup:.6f}"])


def write_sequences_human(path, seq_counts, n_sequences, mapping):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sequence_human", "support_count", "support"])
        for s in sorted(seq_counts.keys(), key=lambda x: (len(x), tuple(item_key(i) for i in x))):
            sc = seq_counts[s]
            sup = sc / n_sequences
            human = [mapping.get(it, it) for it in s]
            w.writerow([" -> ".join(human), sc, f"{sup:.6f}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences", default=os.path.join(DATA_DIR, "sequences.csv"))
    ap.add_argument("--mapping", default=os.path.join(DATA_DIR, "mapping.csv"))
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--min-sup-count", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    sequences = read_sequences_long(args.sequences)
    mapping = read_mapping(args.mapping)

    seq_counts = spade(sequences, min_sup_count=args.min_sup_count, max_len=args.max_len)

    out1 = os.path.join(args.out_dir, "frequent_sequences.csv")
    write_sequences(out1, seq_counts, len(sequences))

    if mapping:
        out2 = os.path.join(args.out_dir, "frequent_sequences_human.csv")
        write_sequences_human(out2, seq_counts, len(sequences), mapping)

    print(f"OK: znaleziono {len(seq_counts)} częstych sekwencji (min_sup_count={args.min_sup_count})")
    print("Zapisano:", out1)


if __name__ == "__main__":
    main()
