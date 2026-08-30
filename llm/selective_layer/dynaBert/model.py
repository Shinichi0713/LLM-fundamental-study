import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. Exit Head (各層に付加する中間分類器) ---
class DeeBertExitHead(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [CLS] トークンのベクトル (batch_size, hidden_size) を抽出して分類
        cls_output = hidden_states[:, 0, :]
        logits = self.classifier(cls_output)
        return logits

    def calculate_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """予測結果のエントロピー（不確実性）を計算"""
        probs = F.softmax(logits, dim=-1)
        # エントロピー: -sum(p * log(p))
        entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
        return entropy


# --- 2. DeeBERT トランスフォーマーブロック (簡易モデル) ---
class DeeBertEncoderLayer(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int):
        super().__init__()
        # 簡易的なトランスフォーマー層の構造（本番はTransformerEncoderLayer等）
        self.layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=4, batch_first=True
        )
        # 各層に配置するExit Head
        self.exit_head = DeeBertExitHead(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.layer(x)
        logits = self.exit_head(x)
        return x, logits


# --- 3. DeeBERT メインモデル ---
class DeeBERT(nn.Module):
    def __init__(self, num_layers: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.embedding = nn.Embedding(1000, hidden_size)
        self.layers = nn.ModuleList([
            DeeBertEncoderLayer(hidden_size, num_classes) for _ in range(num_layers)
        ])

    def forward_eval_early_exit(
        self, input_ids: torch.Tensor, entropy_threshold: float
    ) -> tuple[torch.Tensor, int]:
        """
        推論時 (Inference): エントロピーが閾値以下になったら途中でExitする
        """
        x = self.embedding(input_ids)
        
        for layer_idx, encoder_layer in enumerate(self.layers):
            x, logits = encoder_layer(x)
            entropy = encoder_layer.exit_head.calculate_entropy(logits)
            
            # エントロピーが低い（＝モデルの予測に確信がある）場合は早期退出
            if entropy.item() < entropy_threshold:
                return logits, layer_idx + 1  # 予測結果と使用した層数

        # 最終層まで到達した場合
        return logits, len(self.layers)


# --- 4. 動作検証 ---
if __name__ == "__main__":
    torch.manual_seed(42)
    
    # 6層のDeeBERTモデルを定義
    num_layers = 6
    hidden_size = 64
    num_classes = 2
    model = DeeBERT(num_layers=num_layers, hidden_size=hidden_size, num_classes=num_classes)
    model.eval()

    # ダミーデータ (batch_size=1)
    input_ids = torch.randint(0, 1000, (1, 16))

    # エントロピー閾値を変更して推論層数の変化を確認
    print("--- Early Exit 推論デモ ---")
    for threshold in [0.8, 0.3, 0.01]:
        logits, exited_layer = model.forward_eval_early_exit(input_ids, entropy_threshold=threshold)
        print(f"閾値: {threshold:<4} | 終了層: Layer {exited_layer}/{num_layers} | 出力: {logits.detach().numpy()}")