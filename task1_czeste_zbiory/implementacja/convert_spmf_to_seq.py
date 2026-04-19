import argparse
import csv
import os

DATA_DIR = "data"


def main():
    """
    Konwertuje bazę sekwencyjną w formacie SPMF (np. Sign)
    do formatu long:
        sequence_id,pos,item
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "sequences.csv"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    sid = 0
    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.out, "w", encoding="utf-8", newline="") as fout:

        w = csv.writer(fout)
        w.writerow(["sequence_id", "pos", "item"])

        for line in fin:
            line = line.strip()
            if not line:
                continue

            tokens = line.split()
            if "-2" not in tokens:
                continue  # niepełna sekwencja

            sid += 1
            if args.limit is not None and sid > args.limit:
                break

            seq_id = f"s{sid}"
            pos = 1

            for t in tokens:
                if t == "-1":
                    pos += 1
                elif t == "-2":
                    break
                else:
                    w.writerow([seq_id, pos, t])

    print(f"OK: zapisano {args.out} (sekwencje={sid})")


if __name__ == "__main__":
    main()
