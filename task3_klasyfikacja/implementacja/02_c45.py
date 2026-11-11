"""
C4.5 decision tree as described in Quinlan (1993), i.e. C4.5 Release 7.
The Release 8 changes for numeric attributes (Quinlan, 1996: threshold chosen
by gain and an MDL penalty) are NOT applied.

What C4.5 adds to ID3:
  1. Gain ratio = gain / split info, so attributes with many values are not
     favoured. A test is chosen only among tests whose gain is at least the
     average gain of all candidate tests.
  2. Numeric attributes: binary test "A <= t", t is a midpoint between two
     adjacent distinct values, chosen by gain ratio (ties -> smaller t).
     A numeric attribute may be tested again deeper in the tree.
  3. Missing attribute values: every row carries a weight (1.0 at the start).
     A row with an unknown value is sent down all branches with its weight
     split in proportion to the known rows in each branch.
  4. Error-based post-pruning: a subtree is replaced by a leaf when the
     leaf's predicted errors (upper limit of the binomial confidence interval,
     confidence factor CF = 25% by default) are not larger than the subtree's.

Tree representation (shared with 01_id3.py and 04_visualize.py):
  leaf: {"type": "leaf", "label": <class>, "counts": {<class>: <weight>}}
  node: {"type": "node", "feature": <attribute>,
         "threshold": None (categorical) or <t> (numeric),
         "label": <majority class>, "counts": {...},
         "fractions": {<branch>: <share of known weight>},
         "branches": {<value>: subtree}  or  {"<=": subtree, ">": subtree}}
"""

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

W = "__weight__"   # internal column with row weights


# ---------------------------------------------------------------------------
# Weighted class counts and entropy
# ---------------------------------------------------------------------------

def class_counts(data, target):
    """Sum of row weights per class, e.g. No: 5.0, Yes: 9.0."""
    return data.groupby(target)[W].sum()


def entropy(counts):
    """H = -sum p_i * log2(p_i); only positive p_i are used (0 log 0 = 0)."""
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts > 0]
    if counts.size == 0:
        return 0.0
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def to_dict(counts):
    return {cls: round(float(w), 6) for cls, w in counts.items() if w > 0}


# ---------------------------------------------------------------------------
# Evaluating a candidate test
# ---------------------------------------------------------------------------

def evaluate_test(known, missing_weight, target, parts):
    """Gain, split info and gain ratio of a test.

    known          - rows with a known value of the tested attribute
    missing_weight - total weight of rows with an unknown value
    parts          - known rows of each branch

    Without missing values this is exactly the textbook definition.
    With missing values (Quinlan, 1993):
      gain       = F * (H(known) - sum |T_i|/|known| * H(T_i)),  F = |known|/|all|
      split info = entropy of the branch sizes plus one extra "unknown" outcome
    """
    sizes = [p[W].sum() for p in parts]
    known_weight = sum(sizes)
    remainder = sum(s / known_weight * entropy(class_counts(p, target))
                    for p, s in zip(parts, sizes) if s > 0)
    fraction_known = known_weight / (known_weight + missing_weight)
    gain = fraction_known * (entropy(class_counts(known, target)) - remainder)
    split_info = entropy(sizes + [missing_weight])
    ratio = gain / split_info if split_info > 0 else 0.0
    return {"gain": gain, "split_info": split_info, "gain_ratio": ratio}


def valid_split(parts, min_cases):
    """At least two branches must receive min_cases (weighted) rows."""
    return sum(p[W].sum() >= min_cases for p in parts) >= 2


def evaluate_categorical(data, feature, target, domain, min_cases):
    known = data[data[feature].notna()]
    parts = [known[known[feature] == v] for v in domain]
    if not valid_split(parts, min_cases):
        return None
    result = evaluate_test(known, data[W].sum() - known[W].sum(), target, parts)
    return {"feature": feature, "threshold": None, **result}


def evaluate_numeric(data, feature, target, min_cases):
    """Release 7: try every midpoint, keep the one with the best gain ratio."""
    known = data[data[feature].notna()]
    missing_weight = data[W].sum() - known[W].sum()
    values = np.sort(known[feature].unique())
    best = None
    for t in (values[:-1] + values[1:]) / 2:          # ascending, so ties keep the smaller t
        parts = [known[known[feature] <= t], known[known[feature] > t]]
        if not valid_split(parts, min_cases):
            continue
        result = evaluate_test(known, missing_weight, target, parts)
        if best is None or result["gain_ratio"] > best["gain_ratio"]:
            best = {"feature": feature, "threshold": float(t), **result}
    return best


def candidate_tests(data, target, features, domains, numeric, min_cases):
    """One test per attribute (the best threshold for numeric attributes)."""
    tests = []
    for f in features:
        if f in numeric:
            test = evaluate_numeric(data, f, target, min_cases)
        else:
            test = evaluate_categorical(data, f, target, domains[f], min_cases)
        if test is not None:
            tests.append(test)
    return tests


def choose_test(tests):
    """Best gain ratio among tests with gain >= average gain (first one on ties)."""
    if not tests:
        return None
    average_gain = np.mean([t["gain"] for t in tests])
    eligible = [t for t in tests if t["gain"] >= average_gain - 1e-12]
    best = max(eligible, key=lambda t: t["gain_ratio"])
    return best if best["gain"] > 1e-12 else None


# ---------------------------------------------------------------------------
# Partitioning rows at a node (learning, pruning)
# ---------------------------------------------------------------------------

def partition(node, data):
    """{branch: rows}. Rows with an unknown value go to every branch with
    weight multiplied by the share of known weight in that branch."""
    values = data[node["feature"]]
    known = values.notna()
    if node["threshold"] is None:
        masks = {v: known & (values == v) for v in node["branches"]}
    else:
        masks = {"<=": known & (values <= node["threshold"]),
                 ">": known & (values > node["threshold"])}

    unknown = data[~known]
    parts = {}
    for key, mask in masks.items():
        part = data[mask]
        share = node["fractions"][key]
        if not unknown.empty and share > 0:
            extra = unknown.copy()
            extra[W] = extra[W] * share
            part = pd.concat([part, extra])
        parts[key] = part
    return parts


# ---------------------------------------------------------------------------
# Learning
# ---------------------------------------------------------------------------

def make_leaf(label, counts):
    return {"type": "leaf", "label": label, "counts": to_dict(counts)}


def build_tree(data, target, features, domains, numeric, min_cases, parent_label=None):
    counts = class_counts(data, target)
    total = counts.sum() if len(counts) else 0.0

    # 1. No rows reached this branch -> parent's majority class.
    if total <= 0:
        return make_leaf(parent_label, counts)

    label = counts.idxmax()

    # 2. Pure node or no attributes left -> leaf.
    if (counts > 0).sum() == 1 or not features:
        return make_leaf(label, counts)

    # 3. Choose a test; no useful test -> leaf.
    test = choose_test(candidate_tests(data, target, features, domains, numeric, min_cases))
    if test is None:
        return make_leaf(label, counts)

    feature, threshold = test["feature"], test["threshold"]
    keys = list(domains[feature]) if threshold is None else ["<=", ">"]
    node = {"type": "node", "feature": feature, "threshold": threshold,
            "label": label, "counts": to_dict(counts),
            "branches": {k: None for k in keys}, "fractions": {}}

    # 4. Share of known weight per branch (used for rows with unknown values).
    known = data[data[feature].notna()]
    for key in keys:
        if threshold is None:
            in_branch = known[feature] == key
        else:
            in_branch = (known[feature] <= threshold) if key == "<=" else (known[feature] > threshold)
        node["fractions"][key] = known.loc[in_branch, W].sum() / known[W].sum()

    # 5. Recurse. A categorical attribute is used up, a numeric one is not.
    remaining = features if threshold is not None else [f for f in features if f != feature]
    for key, part in partition(node, data).items():
        node["branches"][key] = build_tree(part, target, remaining, domains,
                                           numeric, min_cases, label)
    return node


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------

def pessimistic_errors(n, e, cf=0.25):
    """Predicted errors U_CF(e, n) * n of a leaf with n cases and e errors:
    the upper limit of the binomial confidence interval at level CF
    (Quinlan, 1993). Computed as in C4.5's code (it also works for
    fractional n and e, which appear with missing values):
      e = 0      exact binomial bound: n * (1 - CF^(1/n))
      0 < e < 1  linear interpolation between e = 0 and e = 1
      e >= 1     normal approximation with continuity correction (e + 0.5)
    """
    if n <= 0:
        return 0.0
    if e < 1e-6:
        return n * (1 - cf ** (1 / n))
    if e < 1:
        at_0 = n * (1 - cf ** (1 / n))
        return at_0 + e * (pessimistic_errors(n, 1.0, cf) - at_0)
    if e + 0.5 >= n:
        return e + 0.67 * (n - e)
    z = NormalDist().inv_cdf(1 - cf)          # 0.674 for CF = 25%
    k = e + 0.5
    p = (k + z * z / 2 + z * math.sqrt(k * (1 - k / n) + z * z / 4)) / (n + z * z)
    return n * p


def leaf_errors(data, target, label, cf):
    counts = class_counts(data, target)
    n = counts.sum() if len(counts) else 0.0
    return pessimistic_errors(n, n - counts.get(label, 0.0), cf)


def subtree_errors(tree, data, target, cf):
    """Sum of the predicted errors of the leaves of a subtree."""
    if tree["type"] == "leaf":
        return leaf_errors(data, target, tree["label"], cf)
    parts = partition(tree, data)
    return sum(subtree_errors(tree["branches"][k], parts[k], target, cf) for k in parts)


def prune(tree, data, target, cf=0.25):
    """Bottom-up pruning at every node (subtree replacement). Returns a NEW
    tree; the input tree is not modified. Ties favour the leaf."""
    if tree["type"] == "leaf":
        return tree

    parts = partition(tree, data)
    node = dict(tree)
    node["branches"] = {k: prune(sub, parts[k], target, cf)
                        for k, sub in tree["branches"].items()}

    if leaf_errors(data, target, node["label"], cf) <= subtree_errors(node, data, target, cf):
        return make_leaf(node["label"], class_counts(data, target))
    return node


# ---------------------------------------------------------------------------
# Training and prediction
# ---------------------------------------------------------------------------

def is_numeric(column):
    return pd.api.types.is_numeric_dtype(column) and not pd.api.types.is_bool_dtype(column)


def fit(data, target, features=None, min_cases=1, cf=0.25):
    """Build a C4.5 tree. Returns (unpruned_tree, pruned_tree).
    Rows without a class label must be removed beforehand."""
    if features is None:
        features = [c for c in data.columns if c != target]
    if data[target].isna().any():
        raise ValueError("Rows with a missing class label must be removed first.")

    data = data.copy()
    data[W] = 1.0
    numeric = {f for f in features if is_numeric(data[f])}
    domains = {f: sorted(data[f].dropna().unique().tolist(), key=str)
               for f in features if f not in numeric}

    tree = build_tree(data, target, list(features), domains, numeric, min_cases)
    return tree, prune(tree, data, target, cf)


def class_distribution(node, row):
    """Class probabilities for one row. An unknown value is followed down all
    branches and the results are combined with the branch fractions."""
    if node["type"] == "leaf":
        total = sum(node["counts"].values())
        if total <= 0:
            return {node["label"]: 1.0}
        return {c: w / total for c, w in node["counts"].items()}

    value = row[node["feature"]]
    if pd.isna(value):
        result = {}
        for key, child in node["branches"].items():
            for c, p in class_distribution(child, row).items():
                result[c] = result.get(c, 0.0) + node["fractions"][key] * p
        return result

    if node["threshold"] is not None:
        key = "<=" if value <= node["threshold"] else ">"
    elif value in node["branches"]:
        key = value
    else:                                    # value not seen in training
        return {node["label"]: 1.0}
    return class_distribution(node["branches"][key], row)


def predict(tree, data):
    predictions = []
    for _, row in data.iterrows():
        dist = class_distribution(tree, row)
        predictions.append(max(dist, key=dist.get))
    return predictions


if __name__ == "__main__":
    # python 02_c45.py --data data/spacer_liczbowy.csv --target Spacer
    import sys
    import run_pipeline
    run_pipeline.main(["--algo", "c45", *sys.argv[1:]])