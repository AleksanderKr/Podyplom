import pandas as pd
import numpy as np
import argparse
import pprint


def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    return np.sum([(-counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts)) for i in range(len(elements))])


def gain_ratio(data, split_attribute_name, target_name="class"):
    total_entropy = entropy(data[target_name])
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)

    # Split Information (unikalna cecha C4.5)
    split_info = np.sum([(-counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts)) for i in range(len(vals))])

    weighted_entropy = np.sum(
        [(counts[i] / np.sum(counts)) * entropy(data.where(data[split_attribute_name] == vals[i]).dropna()[target_name])
         for i in range(len(vals))])

    information_gain = total_entropy - weighted_entropy

    # Zabezpieczenie przed dzieleniem przez zero
    if split_info == 0:
        return 0
    return information_gain / split_info


def c45(data, original_data, features, target_attribute_name="class", parent_node_class=None):
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

        # Wybierz cechę z największym GAIN RATIO (Różnica względem ID3)
        item_values = [gain_ratio(data, feature, target_attribute_name) for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]

        tree = {best_feature: {}}
        features = [i for i in features if i != best_feature]

        for value in np.unique(data[best_feature]):
            sub_data = data.where(data[best_feature] == value).dropna()
            subtree = c45(sub_data, original_data, features, target_attribute_name, parent_node_class)
            tree[best_feature][value] = subtree

        return tree


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Sciezka do pliku CSV")
    parser.add_argument("--target", required=True, help="Nazwa kolumny z etykieta (klasa)")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    features = df.columns.tolist()
    features.remove(args.target)

    print(f"Trenowanie C4.5 (Gain Ratio) na danych: {args.data}")
    tree = c45(df, df, features, args.target)

    print("\nZbudowane drzewo decyzyjne C4.5:")
    pprint.pprint(tree)


if __name__ == "__main__":
    main()