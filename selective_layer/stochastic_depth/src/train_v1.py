import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# 1. Exit Gate モジュール (初期は極力退出させない)
# ---------------------------------------------------------
class ExitGate(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc = nn.Linear(d_model, 2)
        # Continue (index 1) のバイアスを大きくして初期の早期退出をガード
        nn.init.constant_(self.fc.bias[0], -8.0)
        nn.init.constant_(self.fc.bias[1], 4.0)

    def forward(self, x, tau=1.0, hard=False):
        logits = self.fc(x)
        if self.training:
            gate_decision = F.gumbel_softmax(logits, tau=tau, hard=hard)
        else:
            preds = torch.argmax(logits, dim=-1)
            gate_decision = F.one_hot(preds, num_classes=2).float()
        return gate_decision

# ---------------------------------------------------------
# 2. Depth-Adaptive Transformer Layer
# ---------------------------------------------------------
class DepthAdaptiveBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.classifier = nn.Linear(d_model, 2)
        self.gate = ExitGate(d_model)

    def forward(self, src, tau=1.0, hard=False):
        src2 = self.norm1(src)
        attn_out, _ = self.self_attn(src2, src2, src2)
        src = src + attn_out

        src2 = self.norm2(src)
        ff_out = self.linear2(self.dropout(F.relu(self.linear1(src2))))
        src = src + ff_out

        cls_rep = src[:, 0, :]
        logits = self.classifier(cls_rep)
        gate_decision = self.gate(cls_rep, tau=tau, hard=hard)

        return src, logits, gate_decision

# ---------------------------------------------------------
# 3. Depth-Adaptive Transformer 全体モデル
# ---------------------------------------------------------
class DepthAdaptiveTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=8, dim_ff=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            DepthAdaptiveBlock(d_model, nhead, dim_ff) for _ in range(num_layers)
        ])

    def forward(self, x, tau=1.0, hard=True):
        batch_size = x.size(0)
        h = self.embedding(x)

        cum_continue_prob = torch.ones(batch_size, 1, device=x.device)
        layer_logits = []
        exit_weights = []

        for layer_idx, layer in enumerate(self.layers):
            h, logits, gate_decision = layer(h, tau=tau, hard=hard)
            layer_logits.append(logits)

            continue_prob = gate_decision[:, 1:2]
            exit_prob = gate_decision[:, 0:1]

            if layer_idx < self.num_layers - 1:
                current_exit_weight = cum_continue_prob * exit_prob
                cum_continue_prob = cum_continue_prob * continue_prob
            else:
                current_exit_weight = cum_continue_prob

            exit_weights.append(current_exit_weight)

        exit_weights = torch.cat(exit_weights, dim=1)
        stacked_logits = torch.stack(layer_logits, dim=1)
        final_logits = torch.sum(stacked_logits * exit_weights.unsqueeze(-1), dim=1)

        return final_logits, exit_weights, layer_logits

# ---------------------------------------------------------
# 4. アルゴリズム的・多段階依存データセット
# ---------------------------------------------------------
np.random.seed(42)
torch.manual_seed(42)

def generate_algorithmic_data(num_samples=5000, seq_len=16):
    X = np.random.randint(1, 50, size=(num_samples, seq_len))
    y = np.zeros(num_samples, dtype=int)
    
    for i in range(num_samples):
        task_type = X[i, 0] % 3
        
        if task_type == 0:
            # 【Tier 1: Easy】単一の要素閾値 (1〜2層)
            y[i] = 1 if X[i, 1] > 25 else 0
            
        elif task_type == 1:
            # 【Tier 2: Medium】前半と後半の間接的な比較 (3〜4層)
            mid = seq_len // 2
            val1 = np.max(X[i, 1:mid])
            val2 = np.min(X[i, mid:])
            y[i] = 1 if val1 > val2 else 0
            
        else:
            # 【Tier 3: Ultra-Hard】長距離状態走査 (Parity / Modulo Scan)
            # 各ステップの数値が直前の状態を変化させる非線形ループ
            # 単層のAttentionでは一発で解けず、層を重ねて状態更新が必要 (7〜8層必須)
            state = 1
            for j in range(1, seq_len):
                v = X[i, j]
                if v % 2 == 0:
                    state = (state + v) % 7
                else:
                    state = (state * v + 3) % 7
            y[i] = 1 if state >= 3 else 0

    return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)

X_data, y_data = generate_algorithmic_data()

X_train, X_val = X_data[:4000], X_data[4000:]
y_train, y_val = y_data[:4000], y_data[4000:]

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ---------------------------------------------------------
# 5. 学習ループ（Depth Penalty Warmup 導入）
# ---------------------------------------------------------
model = DepthAdaptiveTransformer(vocab_size=50, num_layers=8).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
ce_criterion = nn.CrossEntropyLoss()

epochs = 30
warmup_epochs = 12  # 最初12エポックはペナルティ0で深層までしっかり学習させる

history = {"loss": [], "acc": [], "avg_depth": []}

print("=== Warmup付き Depth-Adaptive Transformer 学習開始 ===")
for epoch in range(epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0
    epoch_depths = []
    
    # ペナルティのスケジューリング（Warmup期間終了後に段階的に付与）
    if epoch < warmup_epochs:
        current_lambda = 0.0
    else:
        # 徐々にペナルティを有効化 (0.0001 -> 0.001)
        current_lambda = min(0.001, 0.0001 * (1.2 ** (epoch - warmup_epochs)))

    tau = max(0.5, 3.0 * (0.92 ** epoch))

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()

        final_logits, exit_weights, _ = model(X_batch, tau=tau, hard=True)

        task_loss = ce_criterion(final_logits, y_batch)

        layer_indices = torch.arange(1, model.num_layers + 1, device=device).float()
        expected_depth = torch.sum(exit_weights * layer_indices, dim=1).mean()
        depth_loss = current_lambda * expected_depth

        loss = task_loss + depth_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(final_logits, dim=-1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)
        epoch_depths.append(expected_depth.item())

    epoch_loss = total_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    epoch_avg_depth = np.mean(epoch_depths)

    history["loss"].append(epoch_loss)
    history["acc"].append(epoch_acc)
    history["avg_depth"].append(epoch_avg_depth)

    print(f"Epoch {epoch+1:02d}/{epochs:02d} | Loss: {epoch_loss:.4f} | "
          f"Train Acc: {epoch_acc:.2f}% | Avg Executed Depth: {epoch_avg_depth:.2f} layers | Penalty Lambda: {current_lambda:.6f}")

# ---------------------------------------------------------
# 6. 学習経過の可視化
# ---------------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
epoch_range = range(1, epochs + 1)

ax1.plot(epoch_range, history["loss"], marker='o', color='crimson')
ax1.set_title("Training Loss")
ax1.set_xlabel("Epoch")
ax1.grid(True, alpha=0.3)

ax2.plot(epoch_range, history["acc"], marker='o', color='forestgreen')
ax2.set_title("Training Accuracy (%)")
ax2.set_xlabel("Epoch")
ax2.grid(True, alpha=0.3)

ax3.plot(epoch_range, history["avg_depth"], marker='o', color='darkorange')
ax3.set_title("Average Executed Depth (Layers)")
ax3.set_xlabel("Epoch")
ax3.set_ylim(1, model.num_layers + 0.5)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 7. 評価とレイヤー退出分布
# ---------------------------------------------------------
model.eval()
executed_layers = []
val_correct = 0

with torch.no_grad():
    for X_batch, y_batch in val_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        final_logits, exit_weights, _ = model(X_batch, tau=0.1, hard=True)

        exited_layer = torch.argmax(exit_weights, dim=1) + 1
        executed_layers.extend(exited_layer.cpu().numpy())

        preds = torch.argmax(final_logits, dim=-1)
        val_correct += (preds == y_batch).sum().item()

print("\n=== 評価結果 ===")
print(f"Validation Accuracy: {100 * val_correct / len(val_dataset):.2f}%")
print(f"Average Executed Layers: {np.mean(executed_layers):.2f} / {model.num_layers} layers")

plt.figure(figsize=(7, 4))
plt.hist(executed_layers, bins=range(1, model.num_layers + 2), align='left', rwidth=0.8, color='teal')
plt.xticks(range(1, model.num_layers + 1))
plt.xlabel("Executed Exit Layer")
plt.ylabel("Sample Count")
plt.title("Sample Exit Layer Distribution (Validation Set)")
plt.grid(True, alpha=0.3)
plt.show()