import pandas as pd
import numpy as np
import argparse
import importlib

visualize = importlib.import_module("04_visualize")

def entropy(target_col):
    probs = target_col.value_counts(normalize=True)
    return -np.sum(probs * np.log2(probs + 1e-9)) if not probs.empty else 0

def calc_gain_ratio(data, splits, target_name):
    total_entropy = entropy(data[target_name])
    n_total = len(data)
    if n_total == 0:
        return 0
    
    weights = [len(s) / n_total for s in splits]
    weighted_entropy = sum(w * entropy(s[target_name]) for w, s in zip(weights, splits))
    
    probs = np.array([w for w in weights if w > 0])
    split_info = -np.sum(probs * np.log2(probs + 1e-9))
    
    return (total_entropy - weighted_entropy) / split_info if split_info > 0 else 0

def evaluate_feature(data, original_data, feature, target_name):
    if pd.api.types.is_numeric_dtype(data[feature]):
        vals = np.sort(data[feature].unique())
        thresholds = (vals[:-1] + vals[1:]) / 2.0
        best_gr, best_thresh = 0, None
        for t in thresholds:
            splits = [data[data[feature] <= t], data[data[feature] > t]]
            gr = calc_gain_ratio(data, splits, target_name)
            if gr > best_gr:
                best_gr, best_thresh = gr, t
        return best_gr, best_thresh
    else:
        vals = np.unique(original_data[feature])
        splits = [data[data[feature] == v] for v in vals]
        return calc_gain_ratio(data, splits, target_name), None

def get_majority_class(data, target_name):
    return data[target_name].mode()[0] if not data.empty else None

def build_tree(data, original_data, features, target_name="class", parent_class=None):
    if data.empty:
        return get_majority_class(original_data, target_name)
    
    majority = get_majority_class(data, target_name)
    if len(data[target_name].unique()) <= 1 or not features:
        return majority

    best_gr, best_feat, best_thresh = 0, None, None
    for feat in features:
        gr, thresh = evaluate_feature(data, original_data, feat, target_name)
        if gr > best_gr:
            best_gr, best_feat, best_thresh = gr, feat, thresh

    if best_gr <= 0 or best_feat is None:
        return majority

    tree = {best_feat: {}}
    if best_thresh is not None:
        tree[best_feat][f"<= {best_thresh}"] = build_tree(data[data[best_feat] <= best_thresh], original_data, features, target_name, majority)
        tree[best_feat][f"> {best_thresh}"] = build_tree(data[data[best_feat] > best_thresh], original_data, features, target_name, majority)
    else:
        rem_features = [f for f in features if f != best_feat]
        for val in np.unique(original_data[best_feat]):
            tree[best_feat][val] = build_tree(data[data[best_feat] == val], original_data, rem_features, target_name, majority)

    return tree

def calculate_pep_error(errors, total, z=0.69):
    if total == 0:
        return 0
    p = errors / total
    num = p + (z**2)/(2*total) + z * np.sqrt((p*(1-p))/total + (z**2)/(4*total**2))
    return (num / (1 + (z**2)/total)) * total

def get_subset(data, feature, key):
    if isinstance(key, str) and key.startswith("<= "):
        return data[data[feature] <= float(key.split("<= ")[1])]
    elif isinstance(key, str) and key.startswith("> "):
        return data[data[feature] > float(key.split("> ")[1])]
    return data[data[feature] == key]

def post_prune(tree, data, target_name):
    if not isinstance(tree, dict) or data.empty:
        return tree

    feature = list(tree.keys())[0]
    branches = tree[feature]

    for key, subtree in branches.items():
        branches[key] = post_prune(subtree, get_subset(data, feature, key), target_name)

    majority = get_majority_class(data, target_name)
    leaf_pep = calculate_pep_error(len(data[data[target_name] != majority]), len(data))

    if all(not isinstance(sub, dict) for sub in branches.values()):
        split_pep = 0
        for key, leaf_class in branches.items():
            sub_d = get_subset(data, feature, key)
            errs = len(sub_d[sub_d[target_name] != leaf_class]) if not sub_d.empty else 0
            split_pep += calculate_pep_error(errs, len(sub_d))

        if leaf_pep <= split_pep:
            return majority

    return {feature: branches}

def print_tree(tree, indent=""):
    if not isinstance(tree, dict):
        print(f" -> {tree}")
        return
    for attr, branches in tree.items():
        for val, subtree in branches.items():
            print(f"{indent}[{attr}: {val}]", end="")
            if isinstance(subtree, dict):
                print()
                print_tree(subtree, indent + "  ")
            else:
                print(f" -> {subtree}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--target", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    features = [c for c in df.columns if c != args.target]

    raw_tree = build_tree(df, df, features, args.target)
    print_tree(raw_tree)

    pruned_tree = post_prune(raw_tree, df, args.target)
    print_tree(pruned_tree)

    visualize.plot_tree(raw_tree, title="C4.5 - Before pruning")
    visualize.plot_tree(pruned_tree, title="C4.5 - After pruning")

if __name__ == "__main__":
    main()
