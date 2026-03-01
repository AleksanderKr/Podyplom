import csv
import os
import argparse
from collections import defaultdict
from itertools import combinations


class SpadeMiner:
    def __init__(self, min_sup_count):
        self.min_sup_count = min_sup_count
        self.frequent_sequences = {}  # (itemset_tuple, ...) -> count
        self.id_lists = {}  # item -> {sid: set(times)}

    def read_data(self, path):
        raw_data = defaultdict(lambda: defaultdict(set))

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row['sequence_id']

                time = int(row.get('time') or row.get('event_idx') or row.get('pos') or 0)
                item = row['item'].strip()
                if item:
                    raw_data[sid][time].add(item)

        self.n_sequences = len(raw_data)
        return raw_data

    def build_initial_idlists(self, data):
        """Tworzy pionową bazę danych (Vertical Database) dla pojedynczych elementów."""
        temp_idlists = defaultdict(lambda: defaultdict(set))
        for sid, events in data.items():
            for time, items in events.items():
                for item in items:
                    temp_idlists[item][sid].add(time)

        # Filtrowanie przez min_sup
        for item, sid_map in temp_idlists.items():
            if len(sid_map) >= self.min_sup_count:
                self.id_lists[item] = {sid: sorted(list(times)) for sid, times in sid_map.items()}

    def temporal_join(self, idlist1, idlist2):
        """
        Łączy dwie listy ID tworząc relację sekwencji (e1 następuje po e2).
        Zwraca nową id-listę.
        """
        new_idlist = {}
        for sid in idlist1:
            if sid in idlist2:
                times1 = idlist1[sid]
                times2 = idlist2[sid]

                # Dla każdego zdarzenia w t1, szukamy zdarzeń w t2, które wystąpiły PÓŹNIEJ
                # (W SPADE to jest klucz do budowania sekwencji atomowych)
                min_t1 = times1[0]
                valid_times2 = [t for t in times2 if t > min_t1]

                if valid_times2:
                    new_idlist[sid] = valid_times2
        return new_idlist

    def mine(self):
        # 1-elementowe sekwencje
        items = sorted(self.id_lists.keys())
        for item in items:
            self.frequent_sequences[((item,),)] = len(self.id_lists[item])

        # DFS dla sekwencji (uproszczony na potrzeby sekwencji zdarzeń pojedynczych)
        # Dla pełnej obsługi złożeń wewnątrz koszyków (np. {A,B} -> C)
        # wymagana byłaby dodatkowa funkcja łączenia równoległego.
        for item in items:
            self._grow_sequence(((item,),), self.id_lists[item], items)

    def _grow_sequence(self, prefix, prefix_idlist, items):
        for item in items:
            new_idlist = self.temporal_join(prefix_idlist, self.id_lists[item])
            sup = len(new_idlist)

            if sup >= self.min_sup_count:
                new_prefix = prefix + ((item,),)
                self.frequent_sequences[new_prefix] = sup
                self._grow_sequence(new_prefix, new_idlist, items)

    def save_results(self, out_path, mapping=None):
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["sequence", "support_count", "support"])
            for seq, count in self.frequent_sequences.items():
                formatted_itemsets = []
                for iset in seq:
                    sorted_items = sorted([str(mapping.get(it, it)) for it in iset])
                    formatted_itemsets.append("{" + ",".join(sorted_items) + "}")
                readable_seq = f"<{','.join(formatted_itemsets)}>"

                writer.writerow([readable_seq, count, count / self.n_sequences])

def main():
    # Przykładowe użycie zgodne z Twoim pipeline
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", default="data/sequences.csv")
    parser.add_argument("--min-sup-count", type=int, default=8)
    parser.add_argument("--out", default="out/frequent_sequences.csv")
    args = parser.parse_args()

    miner = SpadeMiner(args.min_sup_count)
    data = miner.read_data(args.sequences)
    miner.build_initial_idlists(data)
    miner.mine()
    miner.save_results(args.out, mapping={})
    print(f"Zakończono. Znaleziono {len(miner.frequent_sequences)} sekwencji.")


if __name__ == "__main__":
    main()