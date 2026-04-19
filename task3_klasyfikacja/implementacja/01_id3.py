import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt


def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy_val = np.sum(
        [(-counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts)) for i in range(len(elements))])
    return entropy_val


def info_gain(data, split_attribute_name, target_name="class"):
    total_entropy = entropy(data[target_name])
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)

    weighted_entropy = np.sum(
        [(counts[i] / np.sum(counts)) * entropy(data.where(data[split_attribute_name] == vals[i]).dropna()[target_name])
         for i in range(len(vals))])

    information_gain = total_entropy - weighted_entropy
    return information_gain


def id3(data, original_data, features, target_attribute_name="class", parent_node_class=None):
    # Warunek stopu 1: Jeśli wszystkie wartości docelowe są takie same, zwróć tę wartość
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]

    # Warunek stopu 2: Jeśli zbiór danych jest pusty, zwróć tryb (najczęstszą wartość) z oryginalnych danych
    elif len(data) == 0:
        return np.unique(original_data[target_attribute_name])[
            np.argmax(np.unique(original_data[target_attribute_name], return_counts=True)[1])]

    # Warunek stopu 3: Jeśli zbiór cech jest pusty, zwróć tryb z obecnych danych
    elif len(features) == 0:
        return parent_node_class

    else:
        parent_node_class = np.unique(data[target_attribute_name])[
            np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]

        # Wybierz cechę z największym przyrostem informacji
        item_values = [info_gain(data, feature, target_attribute_name) for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]

        tree = {best_feature: {}}
        features = [i for i in features if i != best_feature]

        # Buduj gałęzie
        for value in np.unique(data[best_feature]):
            sub_data = data.where(data[best_feature] == value).dropna()
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


def plot_tree(tree, title="Drzewo Decyzyjne"):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_axis_off()
    plt.title(title)

    def get_width(node):
        if not isinstance(node, dict): return 1
        width = 0
        for attr in node:
            for val in node[attr]:
                width += get_width(node[attr][val])
        return width

    def draw_node(node, x, y, dx):
        if not isinstance(node, dict):
            ax.text(x, y, f"\n{node}", bbox=dict(facecolor='lightgreen', boxstyle='round,pad=0.5'), ha='center',
                    va='center', fontsize=10)
            return

        attr = list(node.keys())[0]
        ax.text(x, y, f"[{attr}]", bbox=dict(facecolor='lightblue', boxstyle='square,pad=0.3'), ha='center',
                va='center', fontweight='bold')

        branches = node[attr]
        n = len(branches)
        total_w = get_width(node)

        current_x = x - dx / 2
        for val, subtree in branches.items():
            child_w = get_width(subtree)
            # Obliczanie pozycji dziecka proporcjonalnie do jego szerokości
            w_ratio = child_w / total_w
            child_x = current_x + (dx * w_ratio) / 2

            # Rysowanie linii (gałęzi)
            ax.annotate(val, xy=(child_x, y - 0.2), xytext=(x, y - 0.05),
                        arrowprops=dict(arrowstyle="->", color='gray'),
                        ha='center', va='center', fontsize=9, color='darkred')

            draw_node(subtree, child_x, y - 0.2, dx * w_ratio)
            current_x += dx * w_ratio

    draw_node(tree, 0.5, 1.0, 1.0)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Sciezka do pliku CSV")
    parser.add_argument("--target", required=True, help="Nazwa kolumny z etykieta (klasa)")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    features = df.columns.tolist()
    features.remove(args.target)

    print(f"Trenowanie ID3 na danych: {args.data}")
    tree = id3(df, df, features, args.target)

    print("\nZbudowane drzewo decyzyjne ID3 (tekstowo):")
    # Zakładam, że Twoja funkcja print_tree jest zdefiniowana powyżej
    print_tree(tree)

    # Rysowanie graficzne
    plot_tree(tree, title=f"Drzewo ID3 - {args.data}")


if __name__ == "__main__":
    main()