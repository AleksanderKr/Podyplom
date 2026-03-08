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
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x_lists, vocab, device):
        batch_size = len(x_lists)
        window_size = len(x_lists[0])
        final_input = torch.zeros((batch_size, window_size, self.embedding.embedding_dim)).to(device)
        for b in range(batch_size):
            for w in range(window_size):
                item_ids = torch.tensor([vocab.get(it, 0) for it in x_lists[b][w]]).to(device)
                embeds = self.embedding(item_ids)
                final_input[b, w, :] = embeds.sum(dim=0)
        out, _ = self.lstm(final_input)
        return self.fc(out[:, -1, :])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", default="data/sequences_test.csv")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--window", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.3)
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

    model = SetLSTMModel(len(vocab), 32, 64).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

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
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {epoch_loss / len(loader):.4f}")

    model.eval()
    with torch.no_grad():
        test_input = [[30]]
        test_tensor = [test_input]
        logits = model(test_tensor, vocab, device)
        probs = torch.sigmoid(logits)

        print(f"\nWejście: {test_input}")
        results = []
        for i in range(len(inv_vocab)):
            prob = probs[0][i].item()
            if prob > args.threshold and inv_vocab[i] != '<PAD>':
                results.append((inv_vocab[i], prob))

        results.sort(key=lambda x: x[1], reverse=True)
        print(f"Przewidziany zbiór (P > {args.threshold}):")
        for item, p in results:
            print(f"  - Lek ID: {item} (P: {p:.4f})")


if __name__ == "__main__":
    main()