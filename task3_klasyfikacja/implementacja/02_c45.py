import pandas as pd
import numpy as np
import argparse
import importlib

visualize = importlib.import_module("04_visualize")

def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    return np.sum([(-counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts)) for i in range(len(elements))])

def calculate_discrete_gain_ratio(data, attribute_name, target_name):
    total_entropy = entropy(data[target_name])
    vals, counts = np.unique(data[attribute_name], return_counts=True)

    weighted_entropy = np.sum(
        [(counts[i] / np.sum(counts)) * entropy(data[data[attribute_name] == vals[i]][target_name])
         for i in range(len(vals))])
    information_gain = total_entropy - weighted_entropy
    split_info = np.sum([(-counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts)) for i in range(len(vals))])

    if split_info == 0:
        return 0, None
    return information_gain / split_info, None

def calculate_continuous_gain_ratio(data, attribute_name, target_name):
    total_entropy = entropy(data[target_name])
    sorted_data = data.sort_values(by=attribute_name)
    unique_vals = sorted_data[attribute_name].unique()

    best_gain_ratio = 0
    best_threshold = None

    for i in range(len(unique_vals) - 1):
        threshold = (unique_vals[i] + unique_vals[i + 1]) / 2.0
        left_split = data[data[attribute_name] <= threshold]
        right_split = data[data[attribute_name] > threshold]

        n_left, n_right, n_total = len(left_split), len(right_split), len(data)
        if n_left == 0 or n_right == 0:
            continue

        weighted_entropy = (n_left / n_total) * entropy(left_split[target_name]) + (n_right / n_total) * entropy(
            right_split[target_name])
        information_gain = total_entropy - weighted_entropy
        split_info = -(
                (n_left / n_total) * np.log2(n_left / n_total) + (n_right / n_total) * np.log2(n_right / n_total))

        if split_info != 0:
            gain_ratio = information_gain / split_info
            if gain_ratio > best_gain_ratio:
                best_gain_ratio = gain_ratio
                best_threshold = threshold

    return best_gain_ratio, best_threshold

def is_continuous(data, attribute_name):
    return pd.api.types.is_numeric_dtype(data[attribute_name])

def build_tree(data, original_data, features, target_name="class", parent_node_class=None):
    if len(np.unique(data[target_name])) <= 1:
        return np.unique(data[target_name])[0]
    elif len(data) == 0:
        return np.unique(original_data[target_name])[
            np.argmax(np.unique(original_data[target_name], return_counts=True)[1])]
    elif len(features) == 0:
        return parent_node_class

    parent_node_class = np.unique(data[target_name])[np.argmax(np.unique(data[target_name], return_counts=True)[1])]

    best_gain_ratio = -1
    best_feature, best_threshold = None, None

    for feature in features:
        if is_continuous(data, feature):
            gr, threshold = calculate_continuous_gain_ratio(data, feature, target_name)
        else:
            gr, threshold = calculate_discrete_gain_ratio(data, feature, target_name)

        if gr > best_gain_ratio:
            best_gain_ratio, best_feature, best_threshold = gr, feature, threshold

    if best_gain_ratio <= 0 or best_feature is None:
        return parent_node_class

    tree = {}
    if is_continuous(data, best_feature):
        node_name = f"{best_feature}"
        tree[node_name] = {}
        left_data = data[data[best_feature] <= best_threshold]
        tree[node_name][f"<= {best_threshold}"] = build_tree(left_data, original_data, features, target_name,
                                                             parent_node_class)
        right_data = data[data[best_feature] > best_threshold]
        tree[node_name][f"> {best_threshold}"] = build_tree(right_data, original_data, features, target_name,
                                                            parent_node_class)
    else:
        tree[best_feature] = {}
        remaining_features = [f for f in features if f != best_feature]
        for value in np.unique(data[best_feature]):
            sub_data = data[data[best_feature] == value]
            tree[best_feature][value] = build_tree(sub_data, original_data, remaining_features, target_name,
                                                   parent_node_class)

    return tree

def calculate_pep_error(errors, total, z=0.69):
    if total == 0:
        return 0
    p = errors / total
    numerator = p + (z ** 2) / (2 * total) + z * np.sqrt((p * (1 - p)) / total + (z ** 2) / (4 * total ** 2))
    denominator = 1 + (z ** 2) / total
    upper_bound_rate = numerator / denominator
    return upper_bound_rate * total

def get_majority_class(data, target_name):
    if len(data) == 0: return None
    vals, counts = np.unique(data[target_name], return_counts=True)
    return vals[np.argmax(counts)]

def post_prune(tree, data, target_name):
    if not isinstance(tree, dict) or len(data) == 0:
        return tree

    feature = list(tree.keys())[0]
    branches = tree[feature]

    for key, subtree in branches.items():
        if isinstance(key, str) and key.startswith("<="):
            threshold = float(key.split(" ")[1])
            sub_data = data[data[feature] <= threshold]
        elif isinstance(key, str) and key.startswith(">"):
            threshold = float(key.split(" ")[1])
            sub_data = data[data[feature] > threshold]
        else:
            sub_data = data[data[feature] == key]

        branches[key] = post_prune(subtree, sub_data, target_name)

    majority_class = get_majority_class(data, target_name)
    leaf_errors = len(data[data[target_name] != majority_class])
    leaf_pep_error = calculate_pep_error(leaf_errors, len(data))

    split_pep_error = 0
    is_all_leaves = True

    for key, subtree in branches.items():
        if isinstance(subtree, dict):
            is_all_leaves = False
            break

        if isinstance(key, str) and key.startswith("<="):
            threshold = float(key.split(" ")[1])
            sub_data = data[data[feature] <= threshold]
        elif isinstance(key, str) and key.startswith(">"):
            threshold = float(key.split(" ")[1])
            sub_data = data[data[feature] > threshold]
        else:
            sub_data = data[data[feature] == key]

        child_errors = len(sub_data[sub_data[target_name] != subtree]) if len(sub_data) > 0 else 0
        split_pep_error += calculate_pep_error(child_errors, len(sub_data))

    if is_all_leaves and leaf_pep_error <= split_pep_error:
        return majority_class

    return {feature: branches}

def print_tree(tree, indent=""):
    if not isinstance(tree, dict):
        print(f" -> {tree}")
        return

    for attribute, branches in tree.items():
        for value, subtree in branches.items():
            print(f"{indent}[{attribute}: {value}]", end="")
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
    features = df.columns.tolist()
    features.remove(args.target)

    raw_tree = build_tree(df, df, features, args.target)
    print_tree(raw_tree)

    pruned_tree = post_prune(raw_tree, df, args.target)
    print_tree(pruned_tree)

    visualize.plot_tree(raw_tree, title="C4.5 - Before pruning")
    visualize.plot_tree(pruned_tree, title="C4.5 - After pruning")

if __name__ == "__main__":
    main()