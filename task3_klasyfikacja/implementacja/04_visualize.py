import matplotlib.pyplot as plt

def get_width(node):
    if not isinstance(node, dict): return 1
    width = 0
    for attr in node:
        for val in node[attr]:
            width += get_width(node[attr][val])
    return width

def draw_node(ax, node, x, y, dx):
    if not isinstance(node, dict):
        ax.text(x, y, f"\n{node}", bbox=dict(facecolor='lightgreen', boxstyle='round,pad=0.5'), ha='center',
                va='center', fontsize=10)
        return

    attr = list(node.keys())[0]
    ax.text(x, y, f"[{attr}]", bbox=dict(facecolor='lightblue', boxstyle='square,pad=0.3'), ha='center',
            va='center', fontweight='bold')

    branches = node[attr]
    total_w = get_width(node)
    current_x = x - dx / 2

    for val, subtree in branches.items():
        child_w = get_width(subtree)
        w_ratio = child_w / total_w
        child_x = current_x + (dx * w_ratio) / 2

        ax.annotate(val, xy=(child_x, y - 0.2), xytext=(x, y - 0.05),
                    arrowprops=dict(arrowstyle="->", color='gray'),
                    ha='center', va='center', fontsize=9, color='darkred')

        draw_node(ax, subtree, child_x, y - 0.2, dx * w_ratio)
        current_x += dx * w_ratio

def plot_tree(tree, title="Decision Tree"):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_axis_off()
    plt.title(title)
    draw_node(ax, tree, 0.5, 1.0, 1.0)
    plt.show()