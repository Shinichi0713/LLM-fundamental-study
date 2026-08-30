import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. 幅を可変制御できる Multi-Head Attention ---
class DynamicSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # 全体のパラメータを保持 (Supernet)
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, width_mult: float) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.shape
        
        # 1. 幅 (Head数) のスライスサイズを計算
        active_heads = int(self.num_heads * width_mult)
        active_hidden = active_heads * self.head_dim

        # 2. QKVの射影とスライス (最初の active_hidden 次元のみ使用)
        qkv = self.qkv_proj(x)  # (batch_size, seq_len, 3 * hidden_size)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # アクティブな Head 次元のみを切り出し
        q = q[:, :, :active_hidden].view(batch_size, seq_len, active_heads, self.head_dim).transpose(1, 2)
        k = k[:, :, :active_hidden].view(batch_size, seq_len, active_heads, self.head_dim).transpose(1, 2)
        v = v[:, :, :active_hidden].view(batch_size, seq_len, active_heads, self.head_dim).transpose(1, 2)

        # 3. Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v)  # (batch_size, active_heads, seq_len, head_dim)

        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, active_hidden)

        # 4. Output Projection のスライス対応 (入力: active_hidden, 出力: hidden_size)
        out = F.linear(
            attn_out, 
            weight=self.out_proj.weight[:, :active_hidden], 
            bias=self.out_proj.bias
        )
        return out


# --- 2. 幅を可変制御できる Feed-Forward Network (FFN) ---
class DynamicFFN(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        self.dense2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor, width_mult: float) -> torch.Tensor:
        # FFNの動的幅スライス
        active_intermediate = int(self.intermediate_size * width_mult)

        # Dense 1 (hidden -> active_intermediate)
        h = F.linear(
            x, 
            weight=self.dense1.weight[:active_intermediate, :], 
            bias=self.dense1.bias[:active_intermediate]
        )
        h = F.gelu(h)

        # Dense 2 (active_intermediate -> hidden)
        out = F.linear(
            h, 
            weight=self.dense2.weight[:, :active_intermediate], 
            bias=self.dense2.bias
        )
        return out


# --- 3. Dynamic Transformer Encoder Block ---
class DynamicTransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int):
        super().__init__()
        self.attn = DynamicSelfAttention(hidden_size, num_heads)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ffn = DynamicFFN(hidden_size, intermediate_size)
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, width_mult: float) -> torch.Tensor:
        # 残差接続 + Multi-Head Attention
        x = x + self.attn(self.ln1(x), width_mult=width_mult)
        # 残差接続 + FFN
        x = x + self.ffn(self.ln2(x), width_mult=width_mult)
        return x


# --- 4. DynaBERT (幅と深さを両方制御するメインモデル) ---
class DynaBERT(nn.Module):
    def __init__(self, num_layers: int, hidden_size: int, num_heads: int, intermediate_size: int, num_classes: int):
        super().__init__()
        self.num_layers = num_layers
        self.embedding = nn.Embedding(1000, hidden_size)
        
        self.layers = nn.ModuleList([
            DynamicTransformerBlock(hidden_size, num_heads, intermediate_size)
            for _ in range(num_layers)
        ])
        
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor, width_mult: float = 1.0, depth_mult: float = 1.0) -> torch.Tensor:
        x = self.embedding(input_ids)

        # 深さ（使用する層の選択）の決定
        active_num_layers = int(self.num_layers * depth_mult)
        
        # 均等間隔（Every-Other Strategy）で層を選択するインデックス計算
        if active_num_layers < self.num_layers:
            step = self.num_layers / active_num_layers
            selected_layer_indices = [int(i * step) for i in range(active_num_layers)]
        else:
            selected_layer_indices = list(range(self.num_layers))

        # 選択された層のみを推論実行
        for idx in selected_layer_indices:
            x = self.layers[idx](x, width_mult=width_mult)

        # [CLS] トークンで最終分類
        cls_output = x[:, 0, :]
        logits = self.classifier(cls_output)
        return logits


# --- 5. 動作・推論テスト ---
if __name__ == "__main__":
    torch.manual_seed(42)

    # 12層、768次元、12ヘッド（BERT-Base相当）のDynaBERTを定義
    model = DynaBERT(
        num_layers=12,
        hidden_size=768,
        num_heads=12,
        intermediate_size=3072,
        num_classes=2
    )
    model.eval()

    dummy_input = torch.randint(0, 1000, (1, 128)) # Batch size 1, Seq length 128

    print("--- DynaBERT 動的推論 (Width & Depth Multipliers) ---")
    
    # 異なるサイズ指定（サブネットワーク）でそのまま推論
    configs = [
        (1.0, 1.0),   # Full Model (100% 幅, 100% 深さ)
        (0.75, 1.0),  # 幅75%, 深さ100%
        (0.5, 0.75),  # 幅50%, 深さ75% (9層)
        (0.25, 0.5),  # 幅25%, 深さ50% (6層)
    ]

    for w_mult, d_mult in configs:
        with torch.no_grad():
            output = model(dummy_input, width_mult=w_mult, depth_mult=d_mult)
        
        active_layers = int(12 * d_mult)
        active_heads = int(12 * w_mult)
        print(f"Config [Width: {w_mult:>4.2f} ({active_heads:>2}/12 heads) | Depth: {d_mult:>4.2f} ({active_layers:>2}/12 layers)] -> Output shape: {list(output.shape)}")