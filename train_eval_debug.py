from data_debug import train_loader, test_loader, vocab
from model_debug  import RNNModel
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from prettytable import PrettyTable
import json
import torchtext

torchtext.disable_torchtext_deprecation_warning()

def train_and_evaluate(model, train_loader, test_loader, epochs=10, lr=0.01, name="Pretrained"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    print("\n[INFO] Starting training...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_count = 0

        for batch_idx, (text, labels) in enumerate(train_loader):
            mask = ~torch.isnan(labels)
            text = text[mask]
            labels = labels[mask].long()

            optimizer.zero_grad()
            outputs = model(text)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            if batch_idx % 10 == 0:
                print(f"  [Batch {batch_idx}] Loss: {loss.item():.4f}")

        avg_loss = total_loss / batch_count
        print(f"[Epoch {epoch+1}] Avg Loss: {avg_loss:.4f}")

    print("\n[INFO] Evaluating model...")
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
    print(f"[RESULT] Accuracy: {acc:.4f} | F1-score: {f1:.4f}")

    # Lưu vào results.json
    result = {"experiment": name, "accuracy": acc, "f1_score": f1}

    try:
        with open("results_debug.json", "r") as f:
            results = json.load(f)
            if not isinstance(results, list):
                print("[WARNING] results.json không phải list. Ghi đè lại.")
                results = []
    except:
        results = []

    results.append(result)

    with open("results_debug.json", "w") as f:
        json.dump(results, f, indent=4)

    return acc, f1

# Thử nghiệm Pretrained vs Scratch
# results = {}
# for pretrained in [True, False]:
#     print(f"\n============================")
#     print(f"[EXPERIMENT] RNN Pretrained = {pretrained}")
#     print(f"============================\n")
#
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
#     print(f"[SUMMARY] {key} - Accuracy: {acc:.4f}, F1-score: {f1:.4f}")
#
# with open("results_debug.json", "w") as f:
#     json.dump(results, f, indent=4)

def print_results_table():
    try:
        with open("results_debug.json", "r") as f:
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

# In bảng kết quả
print_results_table()

