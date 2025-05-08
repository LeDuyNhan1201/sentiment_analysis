import json

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize
import nltk
from sklearn.model_selection import train_test_split
from collections import Counter

nltk.download('punkt')

# Đọc dữ liệu
data = pd.read_csv('sentiment_data_debug.csv').dropna()
print(f"\n[INFO] Loaded {len(data)} samples")

# Xử lý text và label
texts = data['text'].tolist()
labels = data['label'].map({'Positive': 0, 'Negative': 1, 'Neutral': 2}).tolist()

print("\n[INFO] Sample raw texts and labels:")
for i in range(2):
    print(f"Text {i}: {texts[i]}")
    print(f"Label {i}: {labels[i]}")

# Token hóa
tokenized_texts = [word_tokenize(t.lower()) for t in texts]
print(f"\n[INFO] First tokenized text:\n{tokenized_texts[0]}")

# Tạo từ điển vocab
all_words = [w for txt in tokenized_texts for w in txt]
print(f"\n[INFO] Total tokens (including repeats): {len(all_words)}")

most_common = Counter(all_words).most_common(4998)
print(f"\n[INFO] Most common 5 words: {most_common[:5]}")

vocab = {'<PAD>': 0, '<UNK>': 1}
for i, (w, _) in enumerate(most_common, 2):
    vocab[w] = i

print(f"\n[INFO] Vocabulary size: {len(vocab)}")
sample_words = tokenized_texts[0][:10]  # lấy 10 token đầu của câu đầu tiên
print("\n[INFO] Sample vocab mappings from first sentence:")
for w in sample_words:
    print(f"  {w} → {vocab.get(w, 1)}")

# Hàm chuyển từ → số
def to_indices(tokens, max_len):
    idxs = [vocab.get(t, 1) for t in tokens][:max_len]
    return idxs + [0] * (max_len - len(idxs))

def to_indices_debug(tokens, max_len):
    idxs = []
    for t in tokens:
        index = vocab.get(t, 1)  # 1 là <UNK>
        idxs.append(index)
        print(f"  Token '{t}' → index {index}")
    if len(idxs) < max_len:
        print(f"  Padding with {max_len - len(idxs)} zeros")
    return idxs[:max_len] + [0] * (max_len - len(idxs))

# Test thử 1 câu
print("\n[DEBUG] Convert first sentence to index:")
indexed_example = to_indices_debug(tokenized_texts[0], max_len=50)
print(f"Indexed: {indexed_example}")

# Chuyển toàn bộ dataset
max_len_text = 50
text_indices = [to_indices(t, max_len_text) for t in tokenized_texts]

# Dataset class
class SentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = torch.tensor(texts)
        self.labels = torch.tensor(labels)
        print(f"\n[INFO] Dataset initialized: texts shape = {self.texts.shape}, labels shape = {self.labels.shape}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

# Chia train/test
train_texts, test_texts, train_labels, test_labels = train_test_split(
    text_indices, labels, test_size=0.2, random_state=42
)
print(f"\n[INFO] Train size: {len(train_texts)} | Test size: {len(test_texts)}")

train_dataset = SentimentDataset(train_texts, train_labels)
test_dataset = SentimentDataset(test_texts, test_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Lấy 1 batch để xem
print("\n[DEBUG] Sample batch from DataLoader:")
sample_batch = next(iter(train_loader))
print("Batch texts shape:", sample_batch[0].shape)
print("Batch labels shape:", sample_batch[1].shape)
print("First sample indices:\n", sample_batch[0][0])
print("First sample label:\n", sample_batch[1][0])

# Sau khi xây vocab
with open("vocab_debug.json", "w") as f:
    json.dump(vocab, f)

