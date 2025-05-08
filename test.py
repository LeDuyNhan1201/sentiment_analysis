import torch
from nltk.tokenize import word_tokenize
import json
from model import RNNModel
import torchtext

torchtext.disable_torchtext_deprecation_warning()

with open("vocab.json", "r") as f:
    vocab = json.load(f)

# Khởi tạo lại mô hình với đúng cấu hình như lúc train
model = RNNModel(vocab_size=len(vocab), embedding_dim=100, hidden_dim=128, output_dim=3, pretrained=True)
model.load_state_dict(torch.load('model/rnn_model.pt'))
model.eval()

def to_indices(tokens, vocab, max_len=50):
    idxs = [vocab.get(t.lower(), 1) for t in tokens][:max_len]
    return idxs + [0] * (max_len - len(idxs))

def predict_sentiment(text):
    tokens = word_tokenize(text)
    indices = to_indices(tokens, vocab)
    input_tensor = torch.tensor([indices])  # Batch size = 1
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).item()
        label_map = {0: "Positive", 1: "Negative", 2: "Neutral"}
        return label_map[pred]


print(predict_sentiment("I really love this product!"))
print(predict_sentiment("I hate you."))
print(predict_sentiment("It's okay, not too bad."))
print(predict_sentiment("I love you"))

