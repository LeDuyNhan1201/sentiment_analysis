from data import train_loader, test_loader, vocab
from model import RNNModel
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
import json
import torchtext

torchtext.disable_torchtext_deprecation_warning()

def train_and_evaluate(model, train_loader, test_loader, epochs=10, lr=0.01):
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
    return acc, f1

# Thử nghiệm Pretrained vs Scratch
results = {}
for pretrained in [True, False]:
    print(f"\n============================")
    print(f"[EXPERIMENT] RNN Pretrained = {pretrained}")
    print(f"============================\n")

    model = RNNModel(
        vocab_size=len(vocab),
        embedding_dim=100,
        hidden_dim=128,
        output_dim=3,
        pretrained=pretrained
    )
    key = f"RNN_Pretrained={pretrained}"
    acc, f1 = train_and_evaluate(model, train_loader, test_loader)
    results[key] = {"Accuracy": acc, "F1-score": f1}
    print(f"[SUMMARY] {key} - Accuracy: {acc:.4f}, F1-score: {f1:.4f}")

with open("results.json", "w") as f:
    json.dump(results, f, indent=4)

torch.save(model.state_dict(), 'model/rnn_model.pt')
