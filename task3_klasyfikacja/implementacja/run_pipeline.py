"""
Experiment pipeline for the decision-tree lab.

Loads a CSV file, optionally holds out a test set, builds ID3 and/or C4.5
trees, prints them with their size and accuracy and saves the drawings in out/.

Examples:
    python run_pipeline.py                                   # Play Tennis (numeric), ID3 + C4.5
    python run_pipeline.py --data data/play_tennis.csv --algo id3
    python run_pipeline.py --data data/play_tennis_missing.csv --algo c45
    python run_pipeline.py --data data/my.csv --target class --exclude id --test-size 0.3
"""

import argparse
import importlib
import os
import sys

import pandas as pd

id3 = importlib.import_module("01_id3")
c45 = importlib.import_module("02_c45")
viz = importlib.import_module("04_visualize")


def load_data(path, target, exclude):
    """Read a CSV file ("?" and empty cells are missing values), drop excluded
    columns (e.g. row identifiers) and rows without a class label."""
    if not os.path.exists(path):
        sys.exit(f"Error: file {path} not found.")
    df = pd.read_csv(path, na_values=["?"], skipinitialspace=True)
    if target not in df.columns:
        sys.exit(f"Error: target column '{target}' not in {list(df.columns)}.")

    df = df.drop(columns=[c for c in exclude if c in df.columns])
    unlabeled = int(df[target].isna().sum())
    if unlabeled:
        print(f"note: {unlabeled} row(s) without a class label removed")
    return df[df[target].notna()].reset_index(drop=True)


def train_test_split(df, test_size, seed):
    if test_size <= 0:
        return df, None
    test = df.sample(frac=test_size, random_state=seed)
    return df.drop(test.index), test


def accuracy(predict, tree, data, target):
    if data is None or data.empty:
        return None
    predictions = pd.Series(predict(tree, data), index=data.index)
    return float((predictions == data[target]).mean())


def report(name, tree, predict, train, test, target, out_path, show):
    print(f"\n=== {name} ===")
    viz.print_tree(tree)
    stats = viz.tree_stats(tree)
    line = (f"internal nodes: {stats['internal']}, leaves: {stats['leaves']}, "
            f"depth: {stats['depth']}, "
            f"train accuracy: {accuracy(predict, tree, train, target):.3f}")
    test_acc = accuracy(predict, tree, test, target)
    if test_acc is not None:
        line += f", test accuracy: {test_acc:.3f}"
    print(line)
    viz.plot_tree(tree, title=name, path=out_path, show=show)
    print(f"saved: {out_path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="ID3 / C4.5 decision trees")
    ap.add_argument("--data", default="data/play_tennis_numeric.csv")
    ap.add_argument("--target", default="Play")
    ap.add_argument("--algo", choices=["id3", "c45", "both"], default="both")
    ap.add_argument("--exclude", nargs="*", default=["Day"],
                    help="columns to ignore if present, e.g. row identifiers (default: Day)")
    ap.add_argument("--test-size", type=float, default=0.0,
                    help="fraction of rows held out for testing (0 = train on all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cf", type=float, default=0.25,
                    help="C4.5 pruning confidence factor (smaller = more pruning)")
    ap.add_argument("--min-cases", type=float, default=1,
                    help="C4.5: minimum (weighted) rows in at least two branches")
    ap.add_argument("--out", default="out")
    ap.add_argument("--show", action="store_true", help="also open plot windows")
    args = ap.parse_args(argv)

    df = load_data(args.data, args.target, args.exclude)
    train, test = train_test_split(df, args.test_size, args.seed)
    features = [c for c in df.columns if c != args.target]
    name = os.path.splitext(os.path.basename(args.data))[0]
    print(f"data: {args.data} | rows: {len(df)} (train {len(train)}, "
          f"test {0 if test is None else len(test)}) | target: {args.target} | "
          f"attributes: {features}")

    if args.algo in ("id3", "both"):
        try:
            tree = id3.fit(train, args.target, features)
            report(f"ID3 - {name}", tree, id3.predict, train, test, args.target,
                   os.path.join(args.out, f"{name}_id3.png"), args.show)
        except ValueError as err:
            print(f"\n=== ID3 - {name} ===\nskipped: {err}")

    if args.algo in ("c45", "both"):
        raw, pruned = c45.fit(train, args.target, features,
                              min_cases=args.min_cases, cf=args.cf)
        report(f"C4.5 before pruning - {name}", raw, c45.predict, train, test,
               args.target, os.path.join(args.out, f"{name}_c45_unpruned.png"), args.show)
        report(f"C4.5 after pruning (CF={args.cf}) - {name}", pruned, c45.predict,
               train, test, args.target,
               os.path.join(args.out, f"{name}_c45_pruned.png"), args.show)


if __name__ == "__main__":
    main()