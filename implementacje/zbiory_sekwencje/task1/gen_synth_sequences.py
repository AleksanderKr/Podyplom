import argparse
import csv
import os
import random
from collections import defaultdict

DATA_DIR = "data"


def _choice_weighted(rng: random.Random, items, weights):
    """
    Losuje element z listy items zgodnie z wagami (weights).
    """
    return rng.choices(items, weights=weights, k=1)[0]


def _make_zipf_weights(n: int, alpha: float):
    """
    Tworzy rozkład Zipfa: kilka itemów bardzo częstych, reszta rzadsza.
    """
    ws = [1.0 / ((i + 1) ** alpha) for i in range(n)]
    s = sum(ws)
    return [w / s for w in ws]


def _normalise_event(event_items):
    """
    Normalizuje event: usuwa duplikaty i sortuje leksykograficznie.
    """
    return sorted(set(event_items))


def _inject_patterns(rng: random.Random, seq_events, patterns, p_apply: float):
    """
    Wstrzykuje do sekwencji kilka wzorców (ciąg eventów), z pewnym prawdopodobieństwem.
    """
    if rng.random() > p_apply:
        return seq_events

    # ile wzorców wstrzyknąć do tej sekwencji (1..2)
    k = 1 if rng.random() < 0.75 else 2
    chosen = rng.sample(patterns, k=k)

    # wstawiamy wzorce w losowe miejsca (z zachowaniem kolejności w sekwencji)
    for pat in chosen:
        if len(pat) > len(seq_events):
            continue
        start = rng.randrange(0, len(seq_events) - len(pat) + 1)
        for i, ev in enumerate(pat):
            # łączymy eventy (żeby zachować „naturalność” i pozwolić na itemsety)
            seq_events[start + i].extend(ev)

    return seq_events


def generate_sequences(
    n_sequences: int,
    min_events: int,
    max_events: int,
    min_event_size: int,
    max_event_size: int,
    n_items: int,
    zipf_alpha: float,
    p_apply_pattern: float,
    p_noise_item: float,
    seed: int,
):
    """
    Generuje sekwencje itemsetów z ukrytymi wzorcami + szumem.
    Zwraca: List[List[List[str]]], gdzie:
      - sekwencja = lista eventów
      - event = lista itemów (potem normalizowana)
    """
    rng = random.Random(seed)

    items = [f"i{i+1}" for i in range(n_items)]
    weights = _make_zipf_weights(n_items, zipf_alpha)

    # Wzorce sekwencyjne (celowo zawierają itemsety)
    # Możesz je zmieniać, ale ważne: to ma być ciąg eventów.
    patterns = [
        [["i1", "i2"], ["i3"], ["i4", "i5"]],
        [["i10"], ["i20", "i21"], ["i22"]],
        [["i7", "i8"], ["i9"]],
        [["i30"], ["i31"], ["i32", "i33"], ["i34"]],
        [["i50", "i51"], ["i52"]],
    ]

    sequences = []
    for _ in range(n_sequences):
        n_ev = rng.randint(min_events, max_events)

        seq_events = []
        for _e in range(n_ev):
            ev_size = rng.randint(min_event_size, max_event_size)
            ev = [_choice_weighted(rng, items, weights) for _ in range(ev_size)]

            # Szum: czasem dokładamy losowy item (symuluje „losowe kliknięcie” itp.)
            if rng.random() < p_noise_item:
                ev.append(_choice_weighted(rng, items, weights))

            seq_events.append(ev)

        # Wstrzyknięcie wzorców
        seq_events = _inject_patterns(rng, seq_events, patterns, p_apply_pattern)

        # Normalizacja eventów (unikamy duplikatów)
        seq_events = [_normalise_event(ev) for ev in seq_events]

        # Usuwamy puste eventy (na wszelki wypadek)
        seq_events = [ev for ev in seq_events if ev]
        if seq_events:
            sequences.append(seq_events)

    return sequences


def write_long_csv(path: str, sequences):
    """
    Zapisuje sekwencje do CSV w formacie long: sequence_id,pos,item.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sequence_id", "pos", "item"])

        for si, seq in enumerate(sequences, start=1):
            sid = f"s{si}"
            for pos, ev in enumerate(seq, start=1):
                for it in ev:
                    w.writerow([sid, pos, it])


def main():
    """
    Generuje syntetyczny zbiór sekwencji z itemsetami i ukrytymi wzorcami.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "synthetic_sequences.csv"))
    ap.add_argument("--n-sequences", type=int, default=500)
    ap.add_argument("--min-events", type=int, default=6)
    ap.add_argument("--max-events", type=int, default=14)
    ap.add_argument("--min-event-size", type=int, default=1)
    ap.add_argument("--max-event-size", type=int, default=4)
    ap.add_argument("--n-items", type=int, default=120)
    ap.add_argument("--zipf-alpha", type=float, default=0.7)
    ap.add_argument("--p-apply-pattern", type=float, default=0.5)
    ap.add_argument("--p-noise-item", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seqs = generate_sequences(
        n_sequences=args.n_sequences,
        min_events=args.min_events,
        max_events=args.max_events,
        min_event_size=args.min_event_size,
        max_event_size=args.max_event_size,
        n_items=args.n_items,
        zipf_alpha=args.zipf_alpha,
        p_apply_pattern=args.p_apply_pattern,
        p_noise_item=args.p_noise_item,
        seed=args.seed,
    )

    write_long_csv(args.out, seqs)

    # Krótka diagnostyka: ile eventów i ile rekordów long
    n_events = sum(len(s) for s in seqs)
    n_rows = sum(len(ev) for s in seqs for ev in s)
    print(f"OK: zapisano {args.out}")
    print(f"  sekwencje: {len(seqs)}")
    print(f"  eventy:    {n_events}")
    print(f"  wiersze:   {n_rows}")


if __name__ == "__main__":
    main()


r"""
python .\gen_synth_sequences.py --out .\data\synthetic_sequences.csv --n-sequences 500 --zipf-alpha 0.85 --p-apply-pattern 0.40 --seed 42
"""

r"""
python .\run_pipeline.py --skip-generate --task sequence --seq-algo apriori_all --sequences .\data\synthetic_sequences.csv --min-sup-count 20 --skip-rules
"""