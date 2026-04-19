import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt


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


def calculate_pep_error(errors, total, z=2):
    """
    Oblicza górną granicę przedziału ufności dla błędu (Pessimistic Error) wg formuły Wilsona.
    z=0.69 odpowiada domyślnemu poziomowi ufności C4.5 (CF=0.25).
    """
    if total == 0:
        return 0
    p = errors / total
    # Formuła górnej granicy Wilson Score Interval
    numerator = p + (z ** 2) / (2 * total) + z * np.sqrt((p * (1 - p)) / total + (z ** 2) / (4 * total ** 2))
    denominator = 1 + (z ** 2) / total
    upper_bound_rate = numerator / denominator
    return upper_bound_rate * total  # Zwraca przewidywaną LICZBĘ błędów


def get_majority_class(data, target_name):
    if len(data) == 0: return None
    vals, counts = np.unique(data[target_name], return_counts=True)
    return vals[np.argmax(counts)]


def post_prune(tree, data, target_name):
    """Przycinanie drzewa metodą bottom-up."""
    if not isinstance(tree, dict) or len(data) == 0:
        return tree  # To już jest liść

    feature = list(tree.keys())[0]
    branches = tree[feature]

    # 1. KROK REKURENCYJNY: Najpierw przytnij dzieci (idziemy na sam dół drzewa)
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

    # 2. OCENA WĘZŁA: Czy zwinięcie go w liść zmniejszy błąd pesymistyczny?
    majority_class = get_majority_class(data, target_name)
    leaf_errors = len(data[data[target_name] != majority_class])
    leaf_pep_error = calculate_pep_error(leaf_errors, len(data))

    split_pep_error = 0
    is_all_leaves = True

    for key, subtree in branches.items():
        if isinstance(subtree, dict):
            is_all_leaves = False  # Jeśli pod spodem jest jeszcze drzewo, nie przycinamy tutaj
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

    # KRYTERIUM PRUNINGU
    if is_all_leaves and leaf_pep_error <= split_pep_error:
        # Odcinamy gałąź! Zastępujemy słownik wartością klasy większościowej.
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

    print(f"Trenowanie pełnego C4.5 na danych: {args.data}")

    # 1. Budowa drzewa
    raw_tree = build_tree(df, df, features, args.target)
    print("\n--- Drzewo PRZED przycięciem ---")
    print_tree(raw_tree) # usunąłem width=1, bo Twoja funkcja tego nie przyjmuje w Twoim kodzie

    # 2. Pruning
    pruned_tree = post_prune(raw_tree, df, args.target)
    print("\n--- Drzewo PO przycięciu ---")
    print_tree(pruned_tree)

    # 3. Rysowanie (Wyświetli dwa osobne okna jedno po drugim)
    print("\nGenerowanie wykresów...")
    plot_tree(raw_tree, title="C4.5 - Przed przycięciem (Overfitted)")
    plot_tree(pruned_tree, title="C4.5 - Po przycięciu (Pessimistic Error Pruning)")

if __name__ == "__main__":
    main()