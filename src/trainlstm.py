import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset, random_split


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_size1=128, hidden_size2=64, dropout=0.2):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.lstm1 = nn.LSTM(embedding_dim, hidden_size1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.fc = nn.Linear(hidden_size2, 1)

    def forward(self, x):
        out = self.embedding(x)
        out = self.dropout(out)
        out, _ = self.lstm1(out)
        out, _ = self.lstm2(out)
        out = self.fc(out[:, -1, :])
        return out.squeeze(1)


def trainmodel(X_train, y_train, X_test, y_test):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_train = np.asarray(X_train, dtype=np.int64)
    X_test = np.asarray(X_test, dtype=np.int64)
    y_train = np.asarray(y_train).reshape(-1).astype(np.float32)
    y_test = np.asarray(y_test).reshape(-1).astype(np.float32)

    X_train_t = torch.tensor(X_train, dtype=torch.long)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.long).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)

    # Train/validation split (90/10)
    dataset = TensorDataset(X_train_t, y_train_t)
    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    vocab_size = int(max(np.max(X_train), np.max(X_test)) + 1)
    model = LSTMClassifier(vocab_size=vocab_size).to(device)
    print(model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())
    model_path = Path('/home/rian/python_project/myvenv/nlp_in_sarcasm_detection/model/LSTMmodel.pt')
    output_dir = model_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop with early stopping
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    best_weights = copy.deepcopy(model.state_dict())

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(100):
        model.train()
        epoch_loss, correct, total = 0.0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(y_batch)
            preds = (torch.sigmoid(output) > 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)
        train_losses.append(epoch_loss / total)
        train_accs.append(correct / total)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                loss = criterion(output, y_batch)
                val_loss += loss.item() * len(y_batch)
                preds = (torch.sigmoid(output) > 0.5).float()
                val_correct += (preds == y_batch).sum().item()
                val_total += len(y_batch)
        val_losses.append(val_loss / val_total)
        val_accs.append(val_correct / val_total)

        print(f"Epoch {epoch+1}: loss={train_losses[-1]:.4f}, acc={train_accs[-1]:.4f}, "
              f"val_loss={val_losses[-1]:.4f}, val_acc={val_accs[-1]:.4f}")

        # Early stopping
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Restore best weights
    model.load_state_dict(best_weights)

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_t)
        y_pred_prob = torch.sigmoid(test_logits).cpu().numpy()
    test_loss = criterion(test_logits, y_test_t).item()
    y_pred = (y_pred_prob > 0.5).astype(int).reshape(-1)
    y_true = y_test.astype(int).reshape(-1)
    acc = (y_pred == y_true).mean()
    print(f'Test score: {test_loss:.4f}')
    print(f'Test accuracy: {acc:.4f}')

    # Classification report
    print(classification_report(y_true, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.tight_layout()
    plt.savefig(output_dir / 'lstm_confusion_matrix.png')
    plt.close()

    # Plot loss and accuracy
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Loss (training data)')
    plt.plot(val_losses, label='Loss (validation data)')
    plt.title('Loss for LSTM Model')
    plt.ylabel('Loss value')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Accuracy (training data)')
    plt.plot(val_accs, label='Accuracy (validation data)')
    plt.title('Accuracy for LSTM Model')
    plt.ylabel('Accuracy value')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / 'lstm_training_history.png')
    plt.close()

    return model



