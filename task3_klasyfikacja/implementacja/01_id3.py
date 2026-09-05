import pandas as pd
import numpy as np
import argparse
import importlib

visualize = importlib.import_module("04_visualize")

def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy_val = np.sum(
        [(-counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts)) for i in range(len(elements))])
    return entropy_val

def info_gain(data, split_attribute_name, target_name="class"):
    total_entropy = entropy(data[target_name])
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)

    weighted_entropy = np.sum(
        [(counts[i] / np.sum(counts)) * entropy(data[data[split_attribute_name] == vals[i]][target_name])
         for i in range(len(vals))])

    information_gain = total_entropy - weighted_entropy
    return information_gain

def id3(data, original_data, features, target_attribute_name="class", parent_node_class=None):
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]

    elif len(data) == 0:
        return np.unique(original_data[target_attribute_name])[
            np.argmax(np.unique(original_data[target_attribute_name], return_counts=True)[1])]

    elif len(features) == 0:
        return parent_node_class

    else:
        parent_node_class = np.unique(data[target_attribute_name])[
            np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]

        item_values = [info_gain(data, feature, target_attribute_name) for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]

        tree = {best_feature: {}}
        features = [i for i in features if i != best_feature]

        for value in np.unique(data[best_feature]):
            sub_data = data[data[best_feature] == value]
            subtree = id3(sub_data, original_data, features, target_attribute_name, parent_node_class)
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
    features.remove(args.target)

    tree = id3(df, df, features, args.target)

    print_tree(tree)
    visualize.plot_tree(tree, title=f"ID3 Tree - {args.data}")

if __name__ == "__main__":
    main()