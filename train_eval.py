import json

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from prettytable import PrettyTable
from data import train_loader, test_loader, vocab
from model import RNNModel

def train_and_evaluate(model, train_loader, test_loader, epochs=10, lr=0.01, name="Pretrained"):
    # Loss function và Optimizer (dùng SGD, không dùng Adam)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Vòng lặp huấn luyện
    for epoch in range(epochs):
        model.train()
        for text, labels in train_loader:
            mask = ~torch.isnan(labels)
            text = text[mask]
            labels = labels[mask].long()

            optimizer.zero_grad()
            outputs = model(text)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # Đánh giá mô hình
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for text, labels in test_loader:
            mask = ~torch.isnan(labels)
            text = text[mask]
            labels = labels[mask].long()

            outputs = model(text)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')

    # Lưu vào results.json
    result = {"experiment": name, "accuracy": acc, "f1_score": f1}

    try:
        with open("results.json", "r") as f:
            results = json.load(f)
            if not isinstance(results, list):
                print("[WARNING] results.json không phải list. Ghi đè lại.")
                results = []
    except:
        results = []

    results.append(result)

    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)

    return acc, f1

# Thử nghiệm Pretrained vs Scratch
# results = {}
# for pretrained in [True, False]:
#     model = RNNModel(
#         vocab_size=len(vocab),
#         embedding_dim=100,
#         hidden_dim=128,
#         output_dim=3,
#         pretrained=pretrained
#     )
#     key = f"RNN_Pretrained={pretrained}"
#     acc, f1 = train_and_evaluate(model, train_loader, test_loader)
#     results[key] = {"Accuracy": acc, "F1-score": f1}
#     print(f"{key} - Accuracy: {acc:.4f}, F1-score: {f1:.4f}")
#
# with open("results.json", "w") as f:
#     json.dump(results, f, indent=4)
#
# torch.save(model.state_dict(), 'model/rnn_model.pt')

def print_results_table():
    try:
        with open("results.json", "r") as f:
            results = json.load(f)
    except:
        print("No results found.")
        return

    table = PrettyTable()
    table.field_names = ["Thử nghiệm", "Accuracy", "F1-score", "Ghi chú"]

    for r in results:
        table.add_row([r["experiment"], f"{r['accuracy']:.4f}", f"{r['f1_score']:.4f}", ""])

    print("\nBảng tổng hợp từ results.json:")
    print(table)

model_pretrained = RNNModel(
    vocab_size=len(vocab),
    embedding_dim=100,
    hidden_dim=128,
    output_dim=3,
    pretrained=True
)
model_scratch = RNNModel(
    vocab_size=len(vocab),
    embedding_dim=100,
    hidden_dim=128,
    output_dim=3,
    pretrained=False
)
# Gọi hai lần với pretrained=True và False
acc1, f1_1 = train_and_evaluate(model_pretrained, train_loader, test_loader, name="Pretrained")
acc2, f1_2 = train_and_evaluate(model_scratch, train_loader, test_loader, name="Scratch")
torch.save(model_pretrained.state_dict(), 'model/rnn_model.pt')
# In bảng kết quả
print_results_table()