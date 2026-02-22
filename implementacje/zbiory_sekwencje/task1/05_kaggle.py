import csv
import os
import argparse

DATA_DIR = "data"

def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)

def normalise_item(x: str) -> str:
    x = x.strip()
    x = x.replace("\ufeff", "")
    return x

def import_market_basket(input_csv: str, limit: int | None):
    transactions: list[list[str]] = []
    items_set: set[str] = set()

    with open(input_csv, "r", encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        for i, row in enumerate(r):
            if limit is not None and i >= limit:
                break
            basket: list[str] = []
            for cell in row:
                cell = normalise_item(cell)
                if not cell:
                    continue
                basket.append(cell)
                items_set.add(cell)
            transactions.append(basket)

    items_sorted = sorted(items_set)
    item_to_id = {name: f"i{idx}" for idx, name in enumerate(items_sorted, start=1)}
    id_to_item = {v: k for k, v in item_to_id.items()}

    return transactions, item_to_id, id_to_item

def write_mapping(id_to_item: dict[str, str]):
    path = os.path.join(DATA_DIR, "mapping.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "item_name"])
        for item_id in sorted(id_to_item.keys(), key=lambda x: int(x[1:])):
            w.writerow([item_id, id_to_item[item_id]])

def write_transactions_long(transactions: list[list[str]], item_to_id: dict[str, str]):
    path = os.path.join(DATA_DIR, "transactions.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["transaction_id", "item"])
        for t_idx, basket in enumerate(transactions, start=1):
            tid = f"t{t_idx}"
            for name in basket:
                w.writerow([tid, item_to_id[name]])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Ścieżka do Market_Basket_Optimisation.csv")
    ap.add_argument("--limit", type=int, default=None, help="Ogranicz liczbę transakcji (np. 1000) do testów")
    args = ap.parse_args()

    ensure_dirs()
    transactions, item_to_id, id_to_item = import_market_basket(args.input, args.limit)
    write_mapping(id_to_item)
    write_transactions_long(transactions, item_to_id)

    print(f"OK: zaimportowano transakcje={len(transactions)}, unikalne_itemy={len(item_to_id)}")
    print("Zapisano: data/mapping.csv oraz data/transactions.csv")

if __name__ == "__main__":
    main()


r"""
python .\run_pipeline.py --skip-generate --min-sup  -count 30 --min-conf 0.5 --min-lift 1.2
"""