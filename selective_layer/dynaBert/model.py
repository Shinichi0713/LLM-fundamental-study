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



import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification

# --- 1. モデルとトークナイザーの準備 ---
# ここでは感情分析用に調整されたDeeBERTモデルを例として使用します
model_name = "deebert-base-uncased-sst2" # 例：SST-2タスク用DeeBERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(model_name)

# 早期終了を利用するための設定
# 各層のExit Head（中間分類器）を有効にし、終了判定の閾値（エントロピー）を設定します
# モデルがこの設定に対応している必要があります
if hasattr(model.bert, "enable_early_exit"):
    # Exit Headを有効化
    model.bert.enable_early_exit(True)
    # 早期終了の閾値を設定 (例: エントロピー 0.1)
    # 値を大きくすると速度重視、小さくすると精度重視になります
    model.bert.set_early_exit_threshold(0.1) 
    print(f"DeeBERT early exit enabled with threshold: {model.bert.early_exit_threshold}\n")
else:
    print("This model may not support DeeBERT-style early exit in the standard transformers library.")
    print("Running standard BERT inference.\n")

model.eval() # 推論モードに設定

# --- 2. 推論関数の定義 ---
def run_deebert_inference(text, label_map):
    """入力テキストに対してDeeBERT推論を実行する"""
    print(f"--- 推論開始 ---")
    print(f"Input Text: \"{text}\"")

    # テキストをトークン化し、テンソルに変換
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)

    # 推論実行（勾配計算を無効化）
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True) # 隠れ状態も出力させる

    # 標準のBERT出力
    final_logits = outputs.logits
    predictions = torch.argmax(final_logits, dim=-1)
    
    # --- 3. 早期終了（DeeBERT）情報の抽出 ---
    # Hugging Faceの標準モデルでは、早期終了の情報はoutputs内に特定のキーで格納される場合があります
    # モデルの実装（プルリクエストやカスタムクラス）によります
    # ここでは概念的な抽出例を示します

    exit_information = {}
    
    # 実際の実装でのキーの例 (モデルに依存します)
    # outputs.exited_layer_index (終了した層のインデックス)
    # outputs.exit_logits (終了層の出力)
    # outputs.exit_entropy (終了層のエントロピー)
    
    if hasattr(outputs, "exited_layer_index"):
        # 早期終了した場合の情報
        exited_layer = outputs.exited_layer_index.item()
        exit_logits = outputs.exit_logits
        exit_probs = F.softmax(exit_logits, dim=-1)
        exit_predictions = torch.argmax(exit_probs, dim=-1)
        exit_entropy = outputs.exit_entropy.item()
        
        exit_information['exited'] = True
        exit_information['layer'] = exited_layer + 1 # 1-based index
        exit_information['prediction'] = label_map[exit_predictions.item()]
        exit_information['confidence'] = exit_probs.max().item()
        exit_information['entropy'] = exit_entropy
    else:
        # 早期終了せず最終層まで到達した場合
        exit_information['exited'] = False
        exit_information['layer'] = model.config.num_hidden_layers
        
    # --- 4. 結果の出力 ---
    final_pred_label = label_map[predictions.item()]
    final_probs = F.softmax(final_logits, dim=-1)
    
    print(f"Final Prediction: {final_pred_label} (Confidence: {final_probs.max().item():.4f})")
    print(f"Using standard output.\n")
    
    if exit_information['exited']:
        print(f"** DeeBERT Early Exit triggered! **")
        print(f"  Exited at Layer: {exit_information['layer']} / {model.config.num_hidden_layers}")
        print(f"  Prediction at Exit: {exit_information['prediction']}")
        print(f"  Confidence at Exit: {exit_information['confidence']:.4f}")
        print(f"  Entropy at Exit: {exit_information['entropy']:.4f}")
    else:
        print(f"DeeBERT did not exit early. Processed all {exit_information['layer']} layers.")
    
    print(f"--- 推論終了 ---\n")

# --- 5. 実行例 ---
# 感情分析用のラベルマップ
label_map = {0: "NEGATIVE", 1: "POSITIVE"}

# 簡単なテキスト（浅い層で終了しやすい）
text_easy = "This is the best movie I have ever seen!"
run_deebert_inference(text_easy, label_map)

# 複雑なテキスト（深い層まで処理が必要になりやすい）
text_hard = "While the plot was intricate and the acting was superb, the overall execution felt somewhat disjointed and the ending was predictable."
run_deebert_inference(text_hard, label_map)