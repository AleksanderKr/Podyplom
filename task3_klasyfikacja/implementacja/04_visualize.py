import csv
import argparse
from collections import defaultdict

import matplotlib.pyplot as plt

try:
    import networkx as nx
except ImportError as e:
    raise SystemExit("Brak biblioteki networkx. Zainstaluj: pip install networkx") from e


def read_rules(path: str):
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)

    # wykryj czy to plik human czy nie
    if "antecedent_human" in rows[0]:
        a_key, c_key = "antecedent_human", "consequent_human"
    else:
        a_key, c_key = "antecedent", "consequent"

    rules = []
    for row in rows:
        rules.append({
            "A": row[a_key].strip(),
            "C": row[c_key].strip(),
            "support": float(row["support"]),
            "confidence": float(row["confidence"]),
            "lift": float(row["lift"]),
            "sc_xy": int(row["support_count_xy"]),
        })
    return rules


def build_graph(rules, top_n: int, min_conf: float, min_lift: float):
    # filtr + sort
    filtered = [r for r in rules if r["confidence"] >= min_conf and r["lift"] >= min_lift]
    filtered.sort(key=lambda x: x["lift"], reverse=True)
    top = filtered[:top_n]

    G = nx.DiGraph()

    # dodaj krawędzie; jeśli ta sama para pojawi się kilka razy, bierzemy najlepszą (lift)
    best = {}
    for r in top:
        key = (r["A"], r["C"])
        if key not in best or r["lift"] > best[key]["lift"]:
            best[key] = r

    for (a, c), r in best.items():
        G.add_node(a)
        G.add_node(c)
        G.add_edge(a, c, confidence=r["confidence"], lift=r["lift"], support=r["support"], sc_xy=r["sc_xy"])

    return G


def draw_graph(G, title: str, label_mode: str):
    import matplotlib.pyplot as plt
    import networkx as nx

    left_nodes = set(u for u, _ in G.edges())
    right_nodes = set(v for _, v in G.edges())

    antecedent_only = sorted(left_nodes - right_nodes)
    consequent_only = sorted(right_nodes - left_nodes)
    both = sorted(left_nodes & right_nodes)

    max_col = max(len(antecedent_only), len(both), len(consequent_only), 1)
    fig_h = max(8.0, min(24.0, 0.55 * max_col + 4.0))
    plt.figure(figsize=(20, fig_h))

    pos = {}

    def place_column(nodes, x, top, bottom):
        n = len(nodes)
        if n == 0:
            return
        if n == 1:
            ys = [(top + bottom) / 2.0]
        else:
            step = (top - bottom) / (n - 1)
            ys = [top - i * step for i in range(n)]
        for node, y in zip(nodes, ys):
            pos[node] = (x, y)

    top = 1.0
    bottom = -1.0

    place_column(antecedent_only, 0.0, top, bottom)
    place_column(both, 1.2, top, bottom)
    place_column(consequent_only, 2.4, top, bottom)

    for n in G.nodes():
        if n not in pos:
            pos[n] = (1.2, 0.0)

    confs = [G[u][v]["confidence"] for u, v in G.edges()]
    mn, mx = (min(confs), max(confs)) if confs else (0.0, 1.0)
    widths = [0.8 + 4.5 * ((c - mn) / (mx - mn) if mx > mn else 1.0) for c in confs]

    node_size = 1100 if max_col <= 18 else 850
    font_size = 9 if max_col <= 18 else 8

    nx.draw_networkx_nodes(G, pos, nodelist=antecedent_only, node_size=node_size, node_color="#6FA8DC")
    nx.draw_networkx_nodes(G, pos, nodelist=both, node_size=node_size, node_color="#FFD966")
    nx.draw_networkx_nodes(G, pos, nodelist=consequent_only, node_size=node_size, node_color="#93C47D")

    nx.draw_networkx_labels(G, pos, font_size=font_size)

    nx.draw_networkx_edges(
        G,
        pos,
        width=widths,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=16,
        min_source_margin=12,
        min_target_margin=12,
        connectionstyle="arc3,rad=0.10"
    )

    edge_labels = {}
    for u, v in G.edges():
        data = G[u][v]
        if label_mode == "lift":
            edge_labels[(u, v)] = f"{data['lift']:.2f}"
        elif label_mode == "conf":
            edge_labels[(u, v)] = f"{data['confidence']:.2f}"
        else:
            edge_labels[(u, v)] = f"{data['lift']:.2f}/{data['confidence']:.2f}"

    if len(G.edges()) <= 25:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, label_pos=0.5)

    ax = plt.gca()
    ax.set_xlim(-0.35, 2.75)
    ax.set_ylim(-1.10, 1.10)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout(pad=1.2)
    plt.show()




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", default="out/rules_human.csv", help="CSV z regułami (preferowane rules_human.csv)")
    ap.add_argument("--top", type=int, default=25, help="Ile reguł narysować (TOP po lifcie)")
    ap.add_argument("--min-conf", type=float, default=0.35, help="Minimalny confidence")
    ap.add_argument("--min-lift", type=float, default=1.2, help="Minimalny lift")
    ap.add_argument("--label", choices=["lift", "conf", "both"], default="lift", help="Co pokazać na krawędziach")
    args = ap.parse_args()

    rules = read_rules(args.rules)
    G = build_graph(rules, top_n=args.top, min_conf=args.min_conf, min_lift=args.min_lift)

    title = f"Association Rules Graph (top={args.top}, min_conf={args.min_conf}, min_lift={args.min_lift})"
    draw_graph(G, title=title, label_mode=args.label)


if __name__ == "__main__":
    main()


r"""
python .\04_visualize.py --rules .\out\rules_human.csv --top 20 --min-conf 0.4 --min-lift 1.5 --label lift
"""