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


def is_subsequence_itemsets_indexed(seq_index, cand) -> bool:
    """
    Szybkie sprawdzenie dopasowania kandydata jako podsekwencji:
    - seq_index["pos"][item] daje listę indeksów eventów zawierających item
    - dla eventu kandydata E={a,b,c} bierzemy przecięcie pozycji a∩b∩c
    - wybieramy najmniejszą pozycję >= aktualnej
    """
    pos_map = seq_index["pos"]
    current = 0

    for ev in cand:
        # Jeśli event jest pusty (teoretycznie nie powinien), to dopasowanie jest trywialne
        if not ev:
            continue

        # Startujemy od listy pozycji pierwszego itemu, potem przecinamy
        ev_items = list(ev)
        first = ev_items[0]
        if first not in pos_map:
            return False

        possible = set(pos_map[first])
        for it in ev_items[1:]:
            if it not in pos_map:
                return False
            possible &= set(pos_map[it])
            if not possible:
                return False

        # Szukamy najmniejszego indeksu >= current
        nxt = None
        for p in sorted(possible):
            if p >= current:
                nxt = p
                break
        if nxt is None:
            return False

        current = nxt + 1

    return True


def support_count(cand, seq_indexes, cand_items=None) -> int:
    """
    Liczy support przez skanowanie bazy, ale:
    - odrzuca sekwencje, które nie zawierają wszystkich itemów kandydata
    - używa indeksowanego dopasowania subsekwencji
    """
    if cand_items is None:
        cand_items = cand_items_union(cand)

    c = 0
    cand_len = len(cand)

    for i, seq_index in enumerate(seq_indexes):
        # Tani filtr długości
        if seq_index["len"] < cand_len:
            continue

        # Tani filtr po zbiorze itemów
        if not cand_items.issubset(seq_index["union"]):
            continue

        if is_subsequence_itemsets_indexed(seq_index, cand):
            c += 1

    return c


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


def apriori_prune_event_drop(cand, prev_set) -> bool:
    """
    Pruning apriori po usuwaniu całych eventów: każda (k-1)-eventowa podsekwencja musi być częsta.
    """
    k = len(cand)
    for drop in range(k):
        sub = cand[:drop] + cand[drop + 1 :]
        if sub not in prev_set:
            return False
    return True


def apriori_prune_item_drop(cand, L_set) -> bool:
    """
    Pruning apriori po usuwaniu pojedynczych itemów z eventów: każda sekwencja po takim usunięciu musi być częsta.
    To jest istotne, gdy dopuszczamy wzrost itemsetów wewnątrz eventu.
    """
    for idx, ev in enumerate(cand):
        if len(ev) <= 1:
            continue
        ev_items = sorted(ev, key=item_key)
        for it in ev_items:
            new_ev = frozenset(x for x in ev if x != it)
            sub = list(cand)
            sub[idx] = new_ev
            sub = canonical_sequence(sub)
            if sub not in L_set:
                return False
    return True


def gen_candidates_append_event(prev_freq_seqs):
    """
    Generuje kandydatów o długości k eventów przez join (AprioriAll/GSP):
    - dla sekwencji długości m (=k-1) łączymy te, które mają taki sam sufiks/prefiks
    - działa także dla m=1 (start do k=2)
    Optymalizacja: grupowanie po kluczu join zamiast pełnego O(n^2).
    """
    prev = list(prev_freq_seqs)
    prev_set = set(prev)

    buckets = defaultdict(list)

    # Grupowanie po sufiksie (dla m=1 sufiks jest pusty tuple, więc wszystkie trafią do jednego koszyka)
    for s in prev:
        key = s[1:]  # sufiks bez pierwszego eventu
        buckets[key].append(s)

    cands = set()

    # W każdej grupie łączymy wszystko ze wszystkim: a + last(b)
    for key, seqs in buckets.items():
        # deterministycznie (opcjonalnie): seqs.sort(key=seq_sort_key)
        for a in seqs:
            for b in seqs:
                cand = a + (b[-1],)

                # Pruning event-drop: wszystkie (k-1)-eventowe podsekwencje po usunięciu eventu muszą być częste
                if apriori_prune_event_drop(cand, prev_set):
                    cands.add(cand)

    return cands

def gen_candidates_itemset_growth_full(frontier_seqs, all_items, Lk_set):
    """
    Generuje kandydatów przez pełny itemset growth dla stałej liczby eventów k:
    - dodaje jeden item do dowolnego eventu w sekwencji,
    - zachowuje kanoniczność wewnątrz eventu (żeby nie tworzyć duplikatów),
    - stosuje pruning po usunięciu pojedynczego itemu z eventu względem Lk_set.

    frontier_seqs: zbiór sekwencji, z których rozszerzamy (np. nowo znalezione w domknięciu).
    all_items: posortowana lista wszystkich itemów w bazie.
    Lk_set: zbiór już znanych częstych sekwencji o tej samej liczbie eventów k.
    """
    cands = set()

    for s in frontier_seqs:
        k = len(s)
        for ev_idx in range(k):
            ev = s[ev_idx]
            ev_sorted = sorted(ev, key=item_key)
            ev_max = ev_sorted[-1] if ev_sorted else None

            for it in all_items:
                if it in ev:
                    continue

                # Kanoniczność: dodajemy tylko itemy "większe" od maksymalnego w evencie
                if ev_max is not None and item_key(it) <= item_key(ev_max):
                    continue

                new_ev = frozenset(set(ev) | {it})
                cand_list = list(s)
                cand_list[ev_idx] = new_ev
                cand = canonical_sequence(cand_list)

                # Pruning po usunięciu pojedynczego itemu z eventu
                if apriori_prune_item_drop(cand, Lk_set):
                    cands.add(cand)

    return cands

def close_itemset_growth_level(Lk_counts, sequences, seq_indexes, all_items, min_sup_count):
    """
    Domyka poziom k pod itemset growth, ale liczy support szybciej dzięki indeksom.
    """
    Lk_set = set(Lk_counts.keys())
    frontier = set(Lk_counts.keys())

    while frontier:
        cands = gen_candidates_itemset_growth_full(frontier, all_items, Lk_set)
        if not cands:
            break

        new_freq = {}
        for cand in cands:
            if cand in Lk_set:
                continue
            citems = cand_items_union(cand)
            sc = support_count(cand, seq_indexes, cand_items=citems)
            if sc >= min_sup_count:
                new_freq[cand] = sc

        if not new_freq:
            break

        Lk_counts.update(new_freq)
        newly = set(new_freq.keys())
        Lk_set |= newly
        frontier = newly

    return Lk_counts


def apriori_all_itemsets(sequences, min_sup_count: int) -> dict:
    """
    Implementuje levelwise AprioriAll dla sekwencji itemsetów:
    - poziomy po liczbie eventów k (k=1,2,3,...),
    - na każdym poziomie k wykonuje domknięcie itemset growth (pełne: dowolny event),
    - support liczony przez wielokrotne skanowanie bazy (bez ID-list).
    Zwraca słownik: sekwencja(tuple[frozenset]) -> support_count.
    """
    n_seq = len(sequences)
    if n_seq == 0:
        return {}

    all_items = sorted(all_items_in_db(sequences), key=item_key)
    seq_indexes = prepare_sequence_indexes(sequences)

    # Wynik globalny: wszystkie częste sekwencje (różne k)
    L_all = {}

    # =========================
    # Poziom k=1: <{...}>
    # =========================
    # Najpierw singletony <{i}>
    item_sup = defaultdict(int)
    for seq in sequences:
        seen_items = set()
        for ev in seq:
            seen_items |= set(ev)
        for it in seen_items:
            item_sup[(frozenset([it]),)] += 1

    L1_counts = {s: c for s, c in item_sup.items() if c >= min_sup_count}

    if not L1_counts:
        return {}

    # Domknięcie itemsetów dla k=1 (to daje <{a,b}>, <{a,b,c}>, ...)
    L1_counts = close_itemset_growth_level(L1_counts, sequences, seq_indexes, all_items, min_sup_count)

    L_all.update(L1_counts)

    # =========================
    # Poziomy k>=2: sekwencje eventów
    # =========================
    k = 2
    prev_level = set(L1_counts.keys())  # to są częste sekwencje o długości 1 eventu

    while prev_level:
        # 1) Kandydaci przez dopięcie eventu (append event)
        Ck = gen_candidates_append_event(prev_level)
        if not Ck:
            break

        # 2) Liczenie supportu i wybór częstych na poziomie k
        Lk_counts = {}
        for cand in Ck:
            citems = cand_items_union(cand)
            sc = support_count(cand, seq_indexes, cand_items=citems)

            if sc >= min_sup_count:
                Lk_counts[cand] = sc

        if not Lk_counts:
            break

        # 3) Domknięcie poziomu k pod pełnym itemset growth (dowolny event)
        Lk_counts = close_itemset_growth_level(Lk_counts, sequences, seq_indexes, all_items, min_sup_count)

        # 4) Dopisanie do wyniku globalnego i przejście do następnego poziomu
        L_all.update(Lk_counts)

        prev_level = set(Lk_counts.keys())
        k += 1

    L_all = filter_maximal_sequences(L_all)

    return L_all

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

def prepare_sequence_indexes(sequences):
    """
    Buduje indeksy pomocnicze dla szybkiego liczenia supportu:
    - union_items: zbiór wszystkich itemów w sekwencji (prefilter)
    - pos_map: mapowanie item -> posortowana lista indeksów eventów, gdzie item występuje
    Zwraca listę struktur: [{"union": set, "pos": dict[item, list[int]], "len": int}, ...]
    """
    indexes = []
    for seq in sequences:
        union_items = set()
        pos_map = defaultdict(list)

        for idx, ev in enumerate(seq):
            union_items |= ev
            for it in ev:
                pos_map[it].append(idx)

        indexes.append({
            "union": union_items,
            "pos": dict(pos_map),
            "len": len(seq),
        })
    return indexes

def cand_items_union(cand):
    """
    Zwraca zbiór wszystkich itemów występujących w kandydacie (we wszystkich eventach).
    """
    u = set()
    for ev in cand:
        u |= set(ev)
    return u


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
