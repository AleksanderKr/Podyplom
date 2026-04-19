import csv
import os
import argparse
from collections import defaultdict
from itertools import combinations

DATA_DIR = "data"
OUT_DIR = "out"

POS_KEYS = ("pos", "position", "event_idx", "idx", "order", "time", "t")


def item_key(x: str):
    """
    Zwraca klucz do sortowania identyfikatorów itemów.
    i123 sortuje jako 123, reszta leksykograficznie.
    """
    if x.startswith("i") and x[1:].isdigit():
        return int(x[1:])
    return x


def read_mapping(path: str) -> dict:
    """
    Wczytuje mapowanie item_id -> item_name z CSV (opcjonalnie).
    """
    mapping = {}
    if not os.path.exists(path):
        return mapping
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            mapping[row["item_id"]] = row["item_name"]
    return mapping


def read_sequences_long_itemsets(path: str):
    """
    Wczytuje sekwencje w formacie long i buduje sekwencje zdarzeń (itemsetów).
    Te same (sequence_id, pos) tworzą jeden event: {item1, item2, ...}.
    Zwraca: List[List[frozenset[str]]].
    """
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

        by_sid_pos = defaultdict(lambda: defaultdict(set))
        for row in r:
            sid = row["sequence_id"]
            pos = int(row[pos_key])
            it = row["item"].strip()
            if it:
                by_sid_pos[sid][pos].add(it)

    sequences = []
    for sid, pos_map in by_sid_pos.items():
        events = []
        for pos in sorted(pos_map.keys()):
            items = pos_map[pos]
            if items:
                events.append(frozenset(items))
        if events:
            sequences.append(events)
    return sequences


def seq_to_string(seq) -> str:
    """
    Formatuje sekwencję itemsetów do zapisu zbliżonego do formalnego: <{a,b},{c},{d,e}>.
    """
    parts = []
    for ev in seq:
        inner = ",".join(sorted(ev, key=item_key))
        parts.append("{" + inner + "}")
    return "<" + ",".join(parts) + ">"


def canonical_event(ev) -> frozenset:
    """
    Normalizuje event do postaci zbioru (frozenset).
    """
    return frozenset(ev)


def canonical_sequence(seq) -> tuple:
    """
    Normalizuje sekwencję do postaci hashowalnej: tuple[frozenset, ...].
    """
    return tuple(canonical_event(ev) for ev in seq)


def all_items_in_db(sequences) -> set:
    """
    Zbiera zbiór wszystkich itemów występujących w bazie sekwencji.
    """
    items = set()
    for seq in sequences:
        for ev in seq:
            items |= set(ev)
    return items


def event_tuple(ev) -> tuple:
    """
    Zwraca posortowaną krotkę itemów eventu (dla porównań i kanoniczności).
    """
    return tuple(sorted(ev, key=item_key))


def seq_sort_key(s) -> tuple:
    """
    Klucz sortowania sekwencji: najpierw liczba eventów, potem zawartość eventów.
    """
    return (len(s), [event_tuple(e) for e in s])


def find_frequent_litemsets(sequences, min_sup_count):
    """
    Faza 1: Litemset Phase.
    Szuka częstych litemsetów używając podejścia Apriori na zbiorach.
    Uwaga: w AprioriAll support = liczba sekwencji (klientów) zawierających dany itemset.
    """
    items = set()
    for seq in sequences:
        for ev in seq:
            items |= set(ev)

    litemsets = {}
    current_level = {frozenset([it]) for it in items}

    while current_level:
        freq_k = {}
        for cand in current_level:
            sup = 0
            for seq in sequences:
                # Sekwencja wspiera litemset, jeśli przynajmniej jeden jej event go zawiera
                if any(cand.issubset(ev) for ev in seq):
                    sup += 1
            if sup >= min_sup_count:
                freq_k[cand] = sup

        litemsets.update(freq_k)

        # Generowanie kandydatów na k+1 itemów (standardowy Apriori Join)
        freq_list = list(freq_k.keys())
        next_level = set()
        for i in range(len(freq_list)):
            for j in range(i + 1, len(freq_list)):
                l1 = sorted(freq_list[i], key=item_key)
                l2 = sorted(freq_list[j], key=item_key)
                # Apriori join condition: prefiksy muszą być identyczne
                if l1[:-1] == l2[:-1]:
                    new_cand = frozenset(freq_list[i] | freq_list[j])
                    next_level.add(new_cand)
        current_level = next_level

    return litemsets


def transform_database(sequences, frequent_litemsets):
    """
    Faza 2: Transformation Phase.
    Zastępuje każdy event w sekwencjach zbiorem częstych litemsetów, które w nim "siedzą".
    Zwraca ztransformowaną bazę (listę sekwencji).
    """
    freq_set = set(frequent_litemsets.keys())
    transformed_db = []

    for seq in sequences:
        t_seq = []
        for ev in seq:
            # Szukamy wszystkich częstych litemsetów, które są podzbiorem tego eventu
            t_ev = {litem for litem in freq_set if litem.issubset(ev)}
            if t_ev:
                t_seq.append(t_ev)
        if t_seq:
            transformed_db.append(t_seq)

    return transformed_db


def aprioriall_gen_candidates(prev_seqs):
    """
    Faza 3 (część): Generowanie kandydatów o długości k (w litemsetach)
    z sekwencji k-1 (Sequence Phase).
    """
    cands = set()
    prev_list = list(prev_seqs)
    for s1 in prev_list:
        for s2 in prev_list:
            # W AprioriAll join polega na przesunięciu sekwencji o 1 element
            if s1[1:] == s2[:-1]:
                cand = s1 + (s2[-1],)

                # Pruning: sprawdzamy czy wszystkie podsekwencje k-1 są częste
                is_valid = True
                for drop_idx in range(len(cand)):
                    sub = cand[:drop_idx] + cand[drop_idx + 1:]
                    if sub not in prev_seqs:
                        is_valid = False
                        break
                if is_valid:
                    cands.add(cand)
    return cands


def aprioriall_support_count(cand_seq, transformed_sequences):
    """
    Faza 3 (część): Liczenie supportu kandydata w ztransformowanej bazie.
    """
    count = 0
    cand_len = len(cand_seq)

    for t_seq in transformed_sequences:
        cand_idx = 0
        for t_event in t_seq:  # t_event to zbiór (set) częstych litemsetów
            # Szukamy litemsetu bezpośrednio w ztransformowanym evencie
            if cand_seq[cand_idx] in t_event:
                cand_idx += 1
                if cand_idx == cand_len:
                    count += 1
                    break
    return count


def apriori_all_itemsets(sequences, min_sup_count: int) -> dict:
    """
    Główna orkiestracja 4 faz czystego algorytmu AprioriAll.
    """
    n_seq = len(sequences)
    if n_seq == 0:
        return {}

    # FAZA 1: Litemset Phase
    frequent_litemsets = find_frequent_litemsets(sequences, min_sup_count)
    if not frequent_litemsets:
        return {}

    # FAZA 2: Transformation Phase
    transformed_db = transform_database(sequences, frequent_litemsets)

    # FAZA 3: Sequence Phase
    L_all = {}
    prev_level = set()

    # K=1 to po prostu sekwencje złożone z jednego litemsetu
    for litemset, sup in frequent_litemsets.items():
        seq_1 = (litemset,)
        L_all[seq_1] = sup
        prev_level.add(seq_1)

    k = 2
    while prev_level:
        Ck = aprioriall_gen_candidates(prev_level)
        if not Ck:
            break

        Lk_counts = {}
        for cand in Ck:
            sup = aprioriall_support_count(cand, transformed_db)
            if sup >= min_sup_count:
                Lk_counts[cand] = sup

        if not Lk_counts:
            break

        L_all.update(Lk_counts)
        prev_level = set(Lk_counts.keys())
        k += 1

    # FAZA 4: Maximal Phase
    # Zostawiamy Twoją funkcję filter_maximal_sequences,
    # ponieważ jej logika (ev_s.issubset(t[i])) perfekcyjnie pokrywa się z AprioriAll!
    L_maximal = filter_maximal_sequences(L_all)

    return L_maximal

def filter_maximal_sequences(seq_counts: dict) -> dict:
    """
    Zostawia tylko sekwencje maksymalne:
    sekwencja S jest niemaksymalna, jeśli istnieje inna sekwencja T,
    taka że S jest podsekwencją T (w sensie itemsetów w eventach).
    """
    # Sortuj od najdłuższych, żeby szybciej eliminować krótsze
    seqs = sorted(seq_counts.keys(), key=lambda s: (len(s), sum(len(ev) for ev in s)), reverse=True)

    maximal = {}
    kept = []  # lista sekwencji już uznanych za maksymalne (zwykle dłuższe)

    for s in seqs:
        is_sub = False
        for t in kept:
            # S jest podsekwencją T jeśli każdy event S da się dopasować do eventów T
            # z warunkiem zawierania (ev_s ⊆ ev_t) i rosnących indeksów
            i = 0
            for ev_s in s:
                found = False
                while i < len(t):
                    if ev_s.issubset(t[i]):
                        found = True
                        i += 1
                        break
                    i += 1
                if not found:
                    break
            else:
                is_sub = True
                break

        if not is_sub:
            maximal[s] = seq_counts[s]
            kept.append(s)

    return maximal

def write_sequences(path: str, seq_counts: dict, n_sequences: int):
    """
    Zapisuje częste sekwencje do CSV w formacie formalnym-ish: <{...},{...},...>.
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sequence", "support_count", "support"])
        for s in sorted(seq_counts.keys(), key=seq_sort_key):
            sc = seq_counts[s]
            sup = sc / n_sequences
            w.writerow([seq_to_string(s), sc, f"{sup:.6f}"])


def write_sequences_human(path: str, seq_counts: dict, n_sequences: int, mapping: dict):
    """
    Zapisuje częste sekwencje do CSV, mapując item_id na nazwy przyjazne dla człowieka.
    """
    def human_event(ev):
        return frozenset(mapping.get(it, it) for it in ev)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sequence_human", "support_count", "support"])
        for s in sorted(seq_counts.keys(), key=seq_sort_key):
            sc = seq_counts[s]
            sup = sc / n_sequences
            human_seq = tuple(human_event(ev) for ev in s)
            w.writerow([seq_to_string(human_seq), sc, f"{sup:.6f}"])


def main():
    """
    Punkt wejścia: wczytuje sekwencje, uruchamia AprioriAll i zapisuje wyniki do katalogu out.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences", default=os.path.join(DATA_DIR, "sequences.csv"))
    ap.add_argument("--mapping", default=os.path.join(DATA_DIR, "mapping.csv"))
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--min-sup-count", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    sequences = read_sequences_long_itemsets(args.sequences)
    mapping = read_mapping(args.mapping)

    seq_counts = apriori_all_itemsets(sequences, min_sup_count=args.min_sup_count)

    out1 = os.path.join(args.out_dir, "frequent_sequences.csv")
    write_sequences(out1, seq_counts, len(sequences))

    if mapping:
        out2 = os.path.join(args.out_dir, "frequent_sequences_human.csv")
        write_sequences_human(out2, seq_counts, len(sequences), mapping)

    print(f"OK: znaleziono {len(seq_counts)} częstych sekwencji (min_sup_count={args.min_sup_count})")
    print("Zapisano:", out1)


if __name__ == "__main__":
    main()
