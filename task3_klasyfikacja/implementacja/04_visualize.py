"""
Presentation of decision trees built by 01_id3.py and 02_c45.py:
text listing, tree statistics and a matplotlib drawing saved as PNG.

Node format: see the docstring of 01_id3.py / 02_c45.py.
"""

import os

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def _num(x):
    return f"{x:g}" if isinstance(x, float) else str(x)


def edge_label(node, key):
    """Text of the test on the edge from node to its branch key."""
    if node["threshold"] is None:
        return str(key)
    return f"{key} {_num(node['threshold'])}"


def leaf_label(leaf):
    """Class with C4.5-style statistics: (cases) or (cases/errors)."""
    n = sum(leaf["counts"].values())
    errors = n - leaf["counts"].get(leaf["label"], 0)
    stats = f"{_num(round(n, 2))}"
    if errors > 1e-9:
        stats += f"/{_num(round(errors, 2))}"
    return f"{leaf['label']} ({stats})"


# ---------------------------------------------------------------------------
# Text and statistics
# ---------------------------------------------------------------------------

def tree_to_text(tree, indent=""):
    """Listing in the style of Quinlan's C4.5 program, e.g.
        Outlook = Overcast: Yes (4)
        Outlook = Rain:
        |   Wind = Strong: No (2)
    """
    if tree["type"] == "leaf":
        return f"{indent}{leaf_label(tree)}\n"

    lines = []
    for key, child in tree["branches"].items():
        if tree["threshold"] is None:
            test = f"{tree['feature']} = {key}"
        else:
            test = f"{tree['feature']} {edge_label(tree, key)}"
        if child["type"] == "leaf":
            lines.append(f"{indent}{test}: {leaf_label(child)}\n")
        else:
            lines.append(f"{indent}{test}:\n")
            lines.append(tree_to_text(child, indent + "|   "))
    return "".join(lines)


def print_tree(tree):
    print(tree_to_text(tree), end="")


def count_leaves(tree):
    if tree["type"] == "leaf":
        return 1
    return sum(count_leaves(child) for child in tree["branches"].values())


def count_internal(tree):
    if tree["type"] == "leaf":
        return 0
    return 1 + sum(count_internal(child) for child in tree["branches"].values())


def depth(tree):
    """Maximum root-to-leaf path length in edges."""
    if tree["type"] == "leaf":
        return 0
    return 1 + max(depth(child) for child in tree["branches"].values())


def tree_stats(tree):
    return {"internal": count_internal(tree), "leaves": count_leaves(tree), "depth": depth(tree)}


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _draw(ax, node, level, x_left):
    """Draw node so that its leaves occupy the slots x_left .. x_left + width."""
    width = count_leaves(node)
    x, y = x_left + width / 2, -level

    if node["type"] == "leaf":
        ax.text(x, y, leaf_label(node), ha="center", va="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgreen"))
        return

    ax.text(x, y, node["feature"], ha="center", va="center", fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="square,pad=0.4", facecolor="lightblue"))

    child_left = x_left
    for key, child in node["branches"].items():
        child_x = child_left + count_leaves(child) / 2
        ax.annotate("", xy=(child_x, y - 1 + 0.15), xytext=(x, y - 0.15),
                    arrowprops=dict(arrowstyle="->", color="gray"))
        ax.text((x + child_x) / 2, y - 0.5, edge_label(node, key),
                ha="center", va="center", fontsize=8, color="darkred",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none"))
        _draw(ax, child, level + 1, child_left)
        child_left += count_leaves(child)


def plot_tree(tree, title="Decision tree", path=None, show=False):
    """Draw the tree. Saves a PNG when path is given; opens a window when show=True."""
    leaves, levels = count_leaves(tree), depth(tree) + 1
    fig, ax = plt.subplots(figsize=(max(6, 1.8 * leaves), max(3, 1.6 * levels)))
    ax.set_axis_off()
    ax.set_title(title)
    ax.set_xlim(0, leaves)
    ax.set_ylim(-levels + 0.5, 0.5)
    _draw(ax, tree, 0, 0)
    fig.tight_layout()

    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=120)
    if show:
        plt.show()
    plt.close(fig)