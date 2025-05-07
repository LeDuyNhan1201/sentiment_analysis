from collections import Counter

import nltk
import pandas as pd
import torch
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

nltk.download('punkt_tab')

data = pd.read_csv('sentiment_data.csv').dropna()
texts = data['text'].tolist()
labels = data['label'].map({'Positive': 0, 'Negative': 1, 'Neutral': 2}).tolist()

tokenized_texts = [word_tokenize(t.lower()) for t in texts]
all_words = [w for txt in tokenized_texts for w in txt]
most_common = Counter(all_words).most_common(4998)

vocab = {'<PAD>': 0, '<UNK>': 1}
for i, (w, _) in enumerate(most_common, 2):
    vocab[w] = i
vocab_size = len(vocab)

def to_indices(tokens, max_len):
    idxs = [vocab.get(t, 1) for t in tokens][:max_len]
    return idxs + [0] * (max_len - len(idxs))

max_len_text = 50
text_indices = [to_indices(t, max_len_text) for t in tokenized_texts]

class SentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = torch.tensor(texts)
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

train_texts, test_texts, train_labels, test_labels = train_test_split(
    text_indices, labels, test_size=0.2, random_state=42
)

train_dataset = SentimentDataset(train_texts, train_labels)
test_dataset = SentimentDataset(test_texts, test_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)