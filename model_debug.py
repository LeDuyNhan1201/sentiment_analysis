import torch.nn as nn
import torchtext.vocab as vocab

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pretrained=False):
        super().__init__()
        print(f"[INFO] Initializing RNNModel | Pretrained: {pretrained}")
        print(f"[INFO] Vocab size: {vocab_size}, Embedding dim: {embedding_dim}, Hidden dim: {hidden_dim}")

        if pretrained:
            glove = vocab.GloVe(name='6B', dim=embedding_dim)
            print(f"[INFO] GloVe loaded with {len(glove.itos)} tokens")
            self.embedding = nn.Embedding.from_pretrained(glove.vectors[:vocab_size], freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            print(f"[INFO] Random Embedding initialized")

        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        print(f"\n[DEBUG] Input shape: {text.shape}")  # [batch_size, seq_len]

        embedded = self.embedding(text)
        print(f"[DEBUG] Embedded shape: {embedded.shape}")  # [batch_size, seq_len, embedding_dim]

        output, hidden = self.rnn(embedded)
        print(f"[DEBUG] RNN output shape: {output.shape}")    # [batch_size, seq_len, hidden_dim]
        print(f"[DEBUG] Hidden shape: {hidden.shape}")        # [1, batch_size, hidden_dim]

        out = self.fc(hidden.squeeze(0))  # [batch_size, output_dim]
        print(f"[DEBUG] Output logits shape: {out.shape}")

        return out
