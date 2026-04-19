import csv
import os
import argparse
from collections import defaultdict

class SpadeMiner:
    def __init__(self, min_sup_count):
        self.min_sup_count = min_sup_count
        self.frequent_sequences = {}
        self.id_lists = {}

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
        temp_idlists = defaultdict(lambda: defaultdict(set))
        for sid, events in data.items():
            for time, items in events.items():
                for item in items:
                    temp_idlists[item][sid].add(time)
        for item, sid_map in temp_idlists.items():
            if len(sid_map) >= self.min_sup_count:
                self.id_lists[item] = {sid: sorted(list(times)) for sid, times in sid_map.items()}

    def temporal_join(self, idlist1, idlist2):
        new_idlist = {}
        for sid in idlist1:
            if sid in idlist2:
                min_t1 = idlist1[sid][0]
                valid_times2 = [t for t in idlist2[sid] if t > min_t1]
                if valid_times2:
                    new_idlist[sid] = valid_times2
        return new_idlist

    def simultaneous_join(self, idlist1, idlist2):
        new_idlist = {}
        for sid in idlist1:
            if sid in idlist2:
                common_times = sorted(list(set(idlist1[sid]) & set(idlist2[sid])))
                if common_times:
                    new_idlist[sid] = common_times
        return new_idlist

    def mine(self):
        items = sorted(self.id_lists.keys())
        for item in items:
            self.frequent_sequences[((item,),)] = len(self.id_lists[item])
        for item in items:
            self._grow_sequence(((item,),), self.id_lists[item], items)

    def _grow_sequence(self, prefix, prefix_idlist, items):
        for item in items:
            new_idlist_temp = self.temporal_join(prefix_idlist, self.id_lists[item])
            if len(new_idlist_temp) >= self.min_sup_count:
                new_prefix = prefix + ((item,),)
                if new_prefix not in self.frequent_sequences:
                    self.frequent_sequences[new_prefix] = len(new_idlist_temp)
                    self._grow_sequence(new_prefix, new_idlist_temp, items)

            last_itemset = prefix[-1]
            if item > last_itemset[-1]:
                new_idlist_sim = self.simultaneous_join(prefix_idlist, self.id_lists[item])
                if len(new_idlist_sim) >= self.min_sup_count:
                    new_itemset = tuple(list(last_itemset) + [item])
                    new_prefix = prefix[:-1] + (new_itemset,)
                    if new_prefix not in self.frequent_sequences:
                        self.frequent_sequences[new_prefix] = len(new_idlist_sim)
                        self._grow_sequence(new_prefix, new_idlist_sim, items)

    def save_results(self, out_path, mapping=None):
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["sequence", "support_count", "support"])
            for seq, count in self.frequent_sequences.items():
                formatted_itemsets = []
                for iset in seq:
                    sorted_items = sorted([str(mapping.get(it, it)) if mapping else str(it) for it in iset])
                    formatted_itemsets.append("{" + ",".join(sorted_items) + "}")
                readable_seq = f"<{','.join(formatted_itemsets)}>"
                writer.writerow([readable_seq, count, count / self.n_sequences])

def main():
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