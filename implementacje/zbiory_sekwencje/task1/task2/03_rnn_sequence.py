import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import argparse


class MultiLabelSequenceDataset(Dataset):
    def __init__(self, sequences, vocab, window_size=1):
        self.data = []
        self.vocab_size = len(vocab)
        for seq in sequences:
            if len(seq) > window_size:
                for i in range(len(seq) - window_size):
                    inp_sets = seq[i:i + window_size]
                    target_set = seq[i + window_size]
                    target_vector = np.zeros(self.vocab_size, dtype=np.float32)
                    for item in target_set:
                        target_vector[vocab.get(item, 0)] = 1.0
                    self.data.append((inp_sets, torch.tensor(target_vector)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class SetLSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(SetLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, dropout=0.2, num_layers=2)
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x_lists, vocab, device):
        batch_size = len(x_lists)
        window_size = len(x_lists[0])
        final_input = torch.zeros((batch_size, window_size, self.embedding.embedding_dim)).to(device)
        for b in range(batch_size):
            for w in range(window_size):
                item_ids = torch.tensor([vocab.get(it, 0) for it in x_lists[b][w]]).to(device)
                embeds = self.embedding(item_ids)
                final_input[b, w, :] = embeds.mean(dim=0)

        out, _ = self.lstm(final_input)
        out = self.ln(out[:, -1, :])
        return self.fc(out)


def extract_frequent_sequences(model, vocab, inv_vocab, device, threshold, max_len=3):
    frequent_sequences = []
    initial_items = [k for k in vocab.keys() if k != '<PAD>']

    for start_item in initial_items:
        current_seq = [[start_item]]
        explore_queue = [current_seq]

        while explore_queue:
            seq = explore_queue.pop(0)

            if len(seq) > 1:
                frequent_sequences.append(seq)

            if len(seq) >= max_len:
                continue

            test_tensor = [seq]
            logits = model(test_tensor, vocab, device)
            probs = torch.sigmoid(logits)

            next_items = []
            for i in range(len(inv_vocab)):
                prob = probs[0][i].item()
                if prob >= threshold and inv_vocab[i] != '<PAD>':
                    next_items.append(inv_vocab[i])

            if next_items:
                next_items.sort()
                new_seq = seq.copy()
                new_seq.append(next_items)
                explore_queue.append(new_seq)

    return frequent_sequences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", default="data/sequences_test.csv")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--window", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--max_len", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.sequences)
    grouped_df = df.groupby(['sequence_id', 'pos'])['item'].apply(list).reset_index()
    sequences = grouped_df.groupby('sequence_id')['item'].apply(list).values

    unique_items = sorted(df['item'].unique())
    vocab = {item: i + 1 for i, item in enumerate(unique_items)}
    vocab['<PAD>'] = 0
    inv_vocab = {i: item for item, i in vocab.items()}

    dataset = MultiLabelSequenceDataset(sequences, vocab, window_size=args.window)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, collate_fn=lambda x: zip(*x))

    model = SetLSTMModel(len(vocab), 64, 128).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0
        for x_batch, y_batch in loader:
            y_batch = torch.stack(y_batch).to(device)
            optimizer.zero_grad()
            y_pred = model(x_batch, vocab, device)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

    model.eval()
    with torch.no_grad():
        print(f"\n--- Wyekstrahowane częste sekwencje (Próg prawdopodobieństwa >= {args.threshold}) ---")
        extracted_seqs = extract_frequent_sequences(model, vocab, inv_vocab, device, args.threshold, args.max_len)

        if not extracted_seqs:
            print("Brak sekwencji spełniających podany próg.")
        else:
            for seq in extracted_seqs:
                formatted_seq = " -> ".join(
                    [str(itemset) if isinstance(itemset, list) else f"[{itemset}]" for itemset in seq])
                print(f"Sekwencja: {formatted_seq}")


if __name__ == "__main__":
    main()