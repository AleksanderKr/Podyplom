"""
ID3 decision tree (Quinlan, 1986).

The algorithm in one sentence: at every node pick the attribute with the
highest information gain, split the data on *every* value of that attribute,
and repeat recursively until the node is pure or no attributes are left.

Assumptions of classic ID3 (kept on purpose, they are discussed in the lab):
  * every attribute is treated as categorical - a numeric column is split
    into one branch per distinct number (see Task 1 in the lab script),
  * there are no missing values (use C4.5 or clean the data first),
  * there is no pruning.

Tree representation (shared with 02_c45.py and 04_visualize.py):
  leaf: {"type": "leaf", "label": <class>, "counts": {<class>: <n>, ...}}
  node: {"type": "node", "feature": <attribute>, "threshold": None,
         "label": <majority class>, "counts": {...},
         "branches": {<value>: <subtree>, ...}}
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Measures
# ---------------------------------------------------------------------------

def entropy(labels):
    """Shannon entropy H(S) = -sum p_i * log2(p_i) of a column of class labels.
    Only positive p_i are used (0 log 0 = 0); an empty set has entropy 0."""
    p = labels.value_counts(normalize=True)
    p = p[p > 0]
    if p.empty:
        return 0.0
    return float(-(p * np.log2(p)).sum())


def information_gain(data, feature, target):
    """Gain(S, A) = H(S) - sum |S_v|/|S| * H(S_v) over the values v of A."""
    n = len(data)
    remainder = sum(
        len(subset) / n * entropy(subset[target])
        for _, subset in data.groupby(feature)
    )
    return entropy(data[target]) - remainder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def majority_class(labels):
    """Most frequent class; ties are broken by the sorted class order."""
    return labels.mode().iloc[0]


def class_counts(labels):
    return {cls: int(n) for cls, n in labels.value_counts().sort_index().items()}


def make_leaf(label, labels):
    return {"type": "leaf", "label": label, "counts": class_counts(labels)}


# ---------------------------------------------------------------------------
# Learning
# ---------------------------------------------------------------------------

def build_tree(data, target, features, domains, parent_label=None):
    """Recursive ID3.

    data         - training rows that reached this node
    features     - attributes still available on this path
    domains      - {attribute: all values seen in the full training set};
                   a branch is created for every value, even if no row
                   reaches it here (such a leaf gets the parent's majority)
    parent_label - majority class of the parent node
    """
    # 1. No rows reached this branch -> use the parent's majority class.
    if data.empty:
        return make_leaf(parent_label, data[target])

    label = majority_class(data[target])

    # 2. Pure node or no attributes left -> leaf with the majority class.
    if data[target].nunique() == 1 or not features:
        return make_leaf(label, data[target])

    # 3. Choose the attribute with the highest information gain.
    #    (max() returns the first one on ties, i.e. column order decides.)
    best = max(features, key=lambda f: information_gain(data, f, target))

    node = {
        "type": "node",
        "feature": best,
        "threshold": None,
        "label": label,
        "counts": class_counts(data[target]),
        "branches": {},
    }

    # 4. One branch per value; the used attribute is removed on this path.
    remaining = [f for f in features if f != best]
    for value in domains[best]:
        subset = data[data[best] == value]
        node["branches"][value] = build_tree(subset, target, remaining, domains, label)

    return node


def fit(data, target, features=None):
    """Build an ID3 tree from a DataFrame."""
    if features is None:
        features = [c for c in data.columns if c != target]

    missing = [c for c in features + [target] if data[c].isna().any()]
    if missing:
        raise ValueError(
            f"ID3 does not handle missing values (columns: {missing}). "
            "Clean the data or use C4.5."
        )

    domains = {f: sorted(data[f].unique().tolist(), key=str) for f in features}
    return build_tree(data, target, list(features), domains)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_one(tree, row):
    node = tree
    while node["type"] == "node":
        value = row[node["feature"]]
        if value not in node["branches"]:   # value never seen in training
            return node["label"]
        node = node["branches"][value]
    return node["label"]


def predict(tree, data):
    return [predict_one(tree, row) for _, row in data.iterrows()]


if __name__ == "__main__":
    # Same CLI as before: python 01_id3.py --data data/spacer.csv --target Spacer
    import sys
    import run_pipeline
    run_pipeline.main(["--algo", "id3", *sys.argv[1:]])