from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# モデルを評価モードにして、全ノードの埋め込みを取得
model.eval()
with torch.no_grad():
    # 1層目の出力（隠れ表現）を取得
    h1 = model.conv1(data.x, data.edge_index)
    h1 = F.relu(h1)

    # 最終層の出力（クラススコア）を取得
    out = model(data.x, data.edge_index)

# t-SNE で 2次元に圧縮
tsne = TSNE(n_components=2, random_state=42)
h1_2d = tsne.fit_transform(h1.numpy())
out_2d = tsne.fit_transform(out.numpy())

# 可視化
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 1層目の埋め込み
scatter1 = ax1.scatter(h1_2d[:, 0], h1_2d[:, 1], c=data.y, cmap='Set1', s=5)
ax1.set_title('GCN Layer 1 Embeddings (t-SNE)')
ax1.set_xlabel('t-SNE 1')
ax1.set_ylabel('t-SNE 2')
plt.colorbar(scatter1, ax=ax1)

# 最終層の埋め込み
scatter2 = ax2.scatter(out_2d[:, 0], out_2d[:, 1], c=data.y, cmap='Set1', s=5)
ax2.set_title('GCN Final Output Embeddings (t-SNE)')
ax2.set_xlabel('t-SNE 1')
ax2.set_ylabel('t-SNE 2')
plt.colorbar(scatter2, ax=ax2)

plt.tight_layout()
plt.show()

# モデルを評価モードに
model.eval()

# 全ノードの出力を取得（log_softmax 済み）
with torch.no_grad():
    out = model(data.x, data.edge_index)  # shape: [num_nodes, num_classes]

# ノード 0 の予測結果を確認
node_id = 0

# 各クラスの対数確率 → 確率に変換
log_probs = out[node_id]
probs = torch.exp(log_probs)  # log_softmax の逆変換

# 予測ラベル（最大確率のクラス）
pred_label = out[node_id].argmax().item()
true_label = data.y[node_id].item()

# クラス名の対応（Cora の 7 クラス）
class_names = [
    "Case-Based",
    "Genetic Algorithms",
    "Neural Networks",
    "Probabilistic Methods",
    "Reinforcement Learning",
    "Rule Learning",
    "Theory"
]

# いくつかのノードについて一覧表示
node_ids = [0, 10, 100, 500, 1000]

print("=== 複数ノードの予測結果一覧 ===")
for node_id in node_ids:
    pred_label = out[node_id].argmax().item()
    true_label = data.y[node_id].item()
    match = "✓" if pred_label == true_label else "✗"
    print(f"ノード {node_id:4d}: 予測 {pred_label} ({class_names[pred_label]:20s}) | "
          f"正解 {true_label} ({class_names[true_label]:20s}) | {match}")
    
