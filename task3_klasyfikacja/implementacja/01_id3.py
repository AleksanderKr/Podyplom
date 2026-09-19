import pandas as pd
import numpy as np
import argparse
import importlib

visualize = importlib.import_module("04_visualize")

def entropy(target_col):
    _, counts = np.unique(target_col, return_counts=True)
    probabilities = counts / counts.sum()
    entropy_val = -np.sum(probabilities * np.log2(probabilities))
    return entropy_val

def info_gain(data, split_attribute_name, target_name="class"):
    total_entropy = entropy(data[target_name])
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    
    weighted_entropy = 0
    total_counts = np.sum(counts)
    
    for i, val in enumerate(vals):
        subset = data[data[split_attribute_name] == val]
        weight = counts[i] / total_counts
        weighted_entropy += weight * entropy(subset[target_name])
        
    information_gain = total_entropy - weighted_entropy
    return information_gain

def id3(data, original_data, features, target_attribute_name="class", parent_node_class=None):
    if len(data) == 0:
        return parent_node_class

    unique_classes = np.unique(data[target_attribute_name])
    if len(unique_classes) <= 1:
        return unique_classes[0]

    if len(features) == 0:
        return parent_node_class

    target_vals, target_counts = np.unique(data[target_attribute_name], return_counts=True)
    parent_node_class = target_vals[np.argmax(target_counts)]

    item_values = [info_gain(data, feature, target_attribute_name) for feature in features]
    best_feature_index = np.argmax(item_values)
    best_feature = features[best_feature_index]

    tree = {best_feature: {}}
    
    remaining_features = [f for f in features if f != best_feature]

    for value in np.unique(original_data[best_feature]):
        sub_data = data[data[best_feature] == value]
        subtree = id3(sub_data, original_data, remaining_features, target_attribute_name, parent_node_class)
        tree[best_feature][value] = subtree

    return tree

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
    
    if args.target in features:
        features.remove(args.target)
    else:
        print(f"Błąd: Kolumna docelowa '{args.target}' nie istnieje w danych.")
        return

    tree = id3(df, df, features, args.target)

    print_tree(tree)
    visualize.plot_tree(tree, title=f"ID3 Tree - {args.data}")

if __name__ == "__main__":
    main()
