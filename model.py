import torch.nn as nn
import torchtext.vocab as vocab

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pretrained=False):
        super().__init__()

        # Khởi tạo embedding layer
        if pretrained:
            # Dùng GloVe (100d)
            glove = vocab.GloVe(name='6B', dim=embedding_dim)
            self.embedding = nn.Embedding.from_pretrained(glove.vectors[:vocab_size], freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Khởi tạo RNN layer
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)

        # Khởi tạo Dense layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        # text: [batch_size, seq_len]

        # Bước 1: embedding
        embedded = self.embedding(text)  # [batch_size, seq_len, embedding_dim]

        # Bước 2: RNN
        output, hidden = self.rnn(embedded)  # hidden: [1, batch_size, hidden_dim]

        # Bước 3: Lấy hidden state cuối cùng và đưa qua Dense
        out = self.fc(hidden.squeeze(0))  # [batch_size, output_dim]

        return out
