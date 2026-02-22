import csv
import os
import random
import string
import argparse

DATA_DIR = "data"

CORE = [
    "milk", "cereal", "yoghurt", "bananas",
    "bread", "butter", "ham", "cheese", "tomatoes", "lettuce",
    "pasta", "rice", "chicken", "onions",
    "beer", "chips", "cola", "water", "chocolate",
    "coffee", "tea", "sugar",
]

def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)

def make_product_name(rng):
    a = "".join(rng.choices(string.ascii_lowercase, k=rng.randint(4, 8)))
    b = "".join(rng.choices(string.ascii_lowercase, k=rng.randint(4, 10)))
    return f"{a}_{b}"

def build_mapping(target_items, seed):
    rng = random.Random(seed)
    mapping = {}
    reverse = {}
    for idx, name in enumerate(CORE, start=1):
        item_id = f"i{idx}"
        mapping[item_id] = name
        reverse[name] = item_id

    for idx in range(len(CORE) + 1, target_items + 1):
        item_id = f"i{idx}"
        name = make_product_name(rng)
        mapping[item_id] = name

    return mapping, reverse

def write_mapping(mapping):
    path = os.path.join(DATA_DIR, "mapping.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "item_name"])
        for item_id in sorted(mapping.keys(), key=lambda x: int(x[1:])):
            w.writerow([item_id, mapping[item_id]])

def write_transactions_long(transactions):
    path = os.path.join(DATA_DIR, "transactions.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["transaction_id", "item"])
        for tid, items in transactions:
            for it in items:
                w.writerow([tid, it])

def generate_transactions(n_transactions, seed, reverse, target_items, tail_prob, basket_min, basket_max):
    rng = random.Random(seed)

    def id_of(name):
        return reverse[name]

    id_milk = id_of("milk")
    id_cereal = id_of("cereal")
    id_yoghurt = id_of("yoghurt")
    id_bananas = id_of("bananas")
    id_bread = id_of("bread")
    id_butter = id_of("butter")
    id_ham = id_of("ham")
    id_cheese = id_of("cheese")
    id_tomatoes = id_of("tomatoes")
    id_pasta = id_of("pasta")
    id_onions = id_of("onions")
    id_beer = id_of("beer")
    id_chips = id_of("chips")
    id_cola = id_of("cola")
    id_coffee = id_of("coffee")
    id_sugar = id_of("sugar")

    scenarios = [
        ([id_bread, id_ham, id_cheese], 0.25),
        ([id_pasta, id_tomatoes, id_onions], 0.20),
        ([id_milk, id_cereal, id_bananas], 0.18),
        ([id_beer, id_chips], 0.16),
        ([id_coffee, id_sugar], 0.14),
    ]

    core_ids = [f"i{i}" for i in range(1, len(CORE) + 1)]
    tail_ids = [f"i{i}" for i in range(len(CORE) + 1, target_items + 1)]

    core_weights = []
    for name in CORE:
        w = 1.0
        if name in {"bread", "milk", "cheese", "tomatoes", "pasta", "beer", "chips", "coffee", "cereal", "bananas"}:
            w = 2.4
        core_weights.append(w)

    def weighted_pick_core():
        total = sum(core_weights)
        r = rng.random() * total
        s = 0.0
        for it, w in zip(core_ids, core_weights):
            s += w
            if s >= r:
                return it
        return core_ids[-1]

    def conditional_add(basket):
        if id_bread in basket and rng.random() < 0.70:
            basket.add(id_butter)
        if id_pasta in basket and id_tomatoes in basket and rng.random() < 0.60:
            basket.add(id_cheese)
        if id_milk in basket and rng.random() < 0.35:
            basket.add(id_yoghurt)
        if id_beer in basket and rng.random() < 0.25:
            basket.add(id_cola)

    transactions = []
    for t in range(1, n_transactions + 1):
        basket = set()
        chosen = 0
        for items, p in scenarios:
            if rng.random() < p:
                basket.update(items)
                chosen += 1
                if chosen >= 2:
                    break

        conditional_add(basket)

        target_size = rng.randint(basket_min, basket_max)
        while len(basket) < target_size:
            if tail_ids and rng.random() < tail_prob:
                basket.add(rng.choice(tail_ids))
            else:
                basket.add(weighted_pick_core())

        conditional_add(basket)
        transactions.append((f"t{t}", sorted(basket)))

    return transactions

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-transactions", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-items", type=int, default=len(CORE))
    ap.add_argument("--tail-prob", type=float, default=0.0)
    ap.add_argument("--basket-min", type=int, default=3)
    ap.add_argument("--basket-max", type=int, default=7)
    args = ap.parse_args()

    if args.target_items < len(CORE):
        raise RuntimeError(f"target-items ({args.target_items}) < {len(CORE)}")

    ensure_dirs()
    mapping, reverse = build_mapping(args.target_items, seed=args.seed)
    transactions = generate_transactions(
        n_transactions=args.n_transactions,
        seed=args.seed,
        reverse=reverse,
        target_items=args.target_items,
        tail_prob=args.tail_prob,
        basket_min=args.basket_min,
        basket_max=args.basket_max,
    )

    write_mapping(mapping)
    write_transactions_long(transactions)
    print(f"OK: wygenerowano data/mapping.csv oraz data/transactions.csv (transactions={args.n_transactions}, items={args.target_items}, tail_prob={args.tail_prob})")

if __name__ == "__main__":
    main()
