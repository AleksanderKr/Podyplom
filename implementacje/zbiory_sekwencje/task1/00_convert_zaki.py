import argparse
import csv
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Plik tekstowy (format Zaki/IBM)")
    ap.add_argument("--out", default="data/sequences.csv", help="Plik wynikowy CSV")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    count = 0
    with open(args.input, "r", encoding="utf-8", errors="replace") as fin, \
            open(args.out, "w", encoding="utf-8", newline="") as fout:

        writer = csv.writer(fout)
        writer.writerow(["sequence_id", "time", "item"])

        for line in fin:
            parts = line.strip().split()
            if len(parts) < 4:
                continue

            sid = parts[0]
            time = parts[1]
            # parts[2] to liczba elementów w koszyku, omijamy ją
            items = parts[3:]

            for item in items:
                # Usuwamy ewentualne przecinki z tagów (jak w delicious)
                clean_item = item.strip(",")
                writer.writerow([sid, time, clean_item])
                count += 1

    print(f"OK: Przekonwertowano {args.input} -> {args.out}. Zapisano {count} wierszy.")


if __name__ == "__main__":
    main()