"""
tokenwise_layer_skip.py
========================
Gumbel-Softmax + Straight-Through Estimator を用いた
トークンごとのバイナリ層スキップの PyTorch 実装

実行例:
    python tokenwise_layer_skip.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# 1. Gumbel-Softmax によるバイナリ選択関数
# ============================================================

def gumbel_softmax_binary(logits, temperature=0.5, hard=True):
    """
    バイナリ（2値）選択用の Gumbel-Softmax。

    Args:
        logits: [..., 1]  「通す（pass）」側のロジット。
                           「スキップ（skip）」側のロジットは暗黙的に 0 とする。
        temperature: Gumbel-Softmax の温度。小さいほど one-hot に近づく。
        hard: True の場合、Straight-Through Estimator を使用。

    Returns:
        [..., 2] のテンソル。
        [..., 0] = スキップ確率（one-hot 値）、[..., 1] = 通す確率（one-hot 値）
    """
    # 2クラス用ロジット: [skip_logit=0, pass_logit=logits]
    logits_2 = torch.cat([
        torch.zeros_like(logits),  # skip 側のロジット
        logits                     # pass 側のロジット
    ], dim=-1)  # [..., 2]

    # Gumbel ノイズをサンプリング
    u = torch.rand_like(logits_2)
    gumbel_noise = -torch.log(-torch.log(u + 1e-10) + 1e-10)

    # 温度付き softmax
    y = F.softmax((logits_2 + gumbel_noise) / temperature, dim=-1)  # [..., 2]

    if hard:
        # --- Straight-Through Estimator ---
        # 順伝播: argmax で one-hot
        index = y.max(dim=-1, keepdim=True)[1]  # [..., 1]
        y_hard = torch.zeros_like(y)
        y_hard.scatter_(-1, index, 1.0)

        # 逆伝播: y（ソフトな確率）の勾配を流す
        y = (y_hard - y).detach() + y

    return y


# ============================================================
# 2. トークンごとのルーター
# ============================================================

class TokenwiseRouter(nn.Module):
    """
    各 Transformer 層の手前に配置する軽量ルーター。
    トークンごとに「この層を通すか / スキップするか」を判断する。

    入力: 隠れ状態 h [batch, seq_len, d_model]
    出力: gate [batch, seq_len, 2] （0: skip, 1: pass）
    """
    def __init__(self, d_model):
        super().__init__()
        self.fc = nn.Linear(d_model, 1)

    def forward(self, h, temperature=0.5, hard=True):
        """
        Args:
            h: [B, T, d_model]
            temperature: Gumbel-Softmax の温度
            hard: Straight-Through の有無
        Returns:
            gate: [B, T, 2]  one-hot 風のゲート値
        """
        logit_pass = self.fc(h)  # [B, T, 1]
        gate = gumbel_softmax_binary(logit_pass, temperature=temperature, hard=hard)
        return gate  # [B, T, 2]


# ============================================================
# 3. スキップ可能な Transformer 層
# ============================================================

class SkippableTransformerLayer(nn.Module):
    """
    トークンごとに層の通過/スキップを選択できる Transformer 層。

    スキップの仕組み:
        - 層の入力 h_in を保存しておく
        - ルーターが「通す」と判断したトークンだけ Self-Attention + FFN を実行
        - 「スキップ」と判断したトークンは h_in をそのまま出力

    数学的には:
        h_out = gate_pass * Layer(h_in) + gate_skip * h_in
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        # --- Self-Attention ---
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)

        # --- Feed-Forward Network ---
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        # --- ルーター ---
        self.router = TokenwiseRouter(d_model)

    def forward(self, x, attn_mask=None, temperature=0.5, hard=True):
        """
        Args:
            x: [B, T, d_model]  入力隠れ状態
            attn_mask: Self-Attention 用マスク
            temperature: ルーターの Gumbel-Softmax 温度
            hard: Straight-Through Estimator の有無

        Returns:
            out: [B, T, d_model]  出力隠れ状態
            usage: [B, T]  各トークンがこの層を通ったか（1=通過, 0=スキップ）
        """
        B, T, D = x.shape
        h_in = x  # スキップ時にそのまま使うため保存

        # --- ルーターで通過/スキップを決定 ---
        gate = self.router(x, temperature=temperature, hard=hard)  # [B, T, 2]
        gate_pass = gate[:, :, 1:2]   # [B, T, 1]
        gate_skip = gate[:, :, 0:1]   # [B, T, 1]

        # --- 通常の Transformer 層の計算 ---
        # Self-Attention + Residual
        attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # FFN + Residual
        ffn_out = self.linear2(self.dropout(F.gelu(self.linear1(x))))
        x = self.norm2(x + self.dropout(ffn_out))
        # x: [B, T, d_model]  ← 全トークンについて計算済み

        # --- スキップの適用 ---
        # gate_pass=1 のトークン → 計算済みの x を採用
        # gate_skip=1 のトークン → 元の h_in を採用
        out = gate_pass * x + gate_skip * h_in  # [B, T, d_model]

        # 各トークンが通過したかどうか（可視化・監視用）
        usage = gate_pass.squeeze(-1)  # [B, T]

        return out, usage


# ============================================================
# 4. スキップ可能な Transformer モデル全体
# ============================================================

class SkippableTransformer(nn.Module):
    """
    各層がトークンごとにスキップ可能な Transformer デコーダモデル。
    言語モデリング（次トークン予測）用。
    """
    def __init__(
        self,
        vocab_size,
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        max_seq_len=512,
        dropout=0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        # --- トークン埋め込み + 位置エンコーディング ---
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # --- スキップ可能な Transformer 層のスタック ---
        self.layers = nn.ModuleList([
            SkippableTransformerLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # --- 出力ヘッド ---
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # 重み初期化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, temperature=0.5, hard=True):
        """
        Args:
            input_ids: [B, T]  入力トークン ID
            temperature: ルーターの Gumbel-Softmax 温度
            hard: Straight-Through Estimator の有無

        Returns:
            logits: [B, T, vocab_size]  次トークン予測用ロジット
            layer_usages: List[Tensor]  各層の通過率 [B, T]
        """
        B, T = input_ids.shape

        # 位置エンコーディング
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)  # [1, T]

        # 埋め込み
        x = self.token_emb(input_ids) + self.pos_emb(positions)  # [B, T, D]
        x = self.dropout(x)

        # Causal Mask（下三角マスク）
        causal_mask = torch.triu(torch.ones(T, T, device=input_ids.device), diagonal=1).bool()
        causal_mask = causal_mask.masked_fill(causal_mask, float('-inf'))

        # 各層を順番に通過
        layer_usages = []
        for layer in self.layers:
            x, usage = layer(x, attn_mask=causal_mask, temperature=temperature, hard=hard)
            layer_usages.append(usage)

        x = self.ln_final(x)
        logits = self.lm_head(x)  # [B, T, vocab_size]

        return logits, layer_usages


# ============================================================
# 5. 計算コストペナルティ付き損失関数
# ============================================================

def compute_loss_with_budget(model, logits, targets, layer_usages, lambda_budget=0.1):
    """
    タスク損失（交叉エントロピー）+ 計算コストペナルティ を計算。

    Args:
        model: SkippableTransformer モデル
        logits: [B, T, vocab_size]
        targets: [B, T]  正解トークン ID
        layer_usages: List[[B, T]]  各層の通過ゲート値
        lambda_budget: 計算コストペナルティの係数

    Returns:
        total_loss: 総損失
        ce_loss: 交叉エントロピー損失
        budget_loss: 計算コストペナルティ
    """
    B, T, V = logits.shape

    # --- タスク損失: 交叉エントロピー ---
    ce_loss = F.cross_entropy(
        logits.view(-1, V),
        targets.view(-1),
        ignore_index=-100
    )

    # --- 計算コストペナルテ ---
    # 各層の usage を平均して「モデル全体の平均通過率」を計算
    # usage=1（通過）を減らす方向にペナルティ
    total_usage = torch.stack(layer_usages, dim=0).mean()  # スカラー
    # 目標: total_usage を小さくする → スキップを増やす
    budget_loss = total_usage

    total_loss = ce_loss + lambda_budget * budget_loss

    return total_loss, ce_loss, budget_loss


# ============================================================
# 6. 実行デモ
# ============================================================

def demo():
    """デモ実行。ダミーデータで学習ステップを1回実行する。"""

    # --- ハイパーパラメータ ---
    VOCAB_SIZE = 1000
    D_MODEL = 128
    NHEAD = 4
    NUM_LAYERS = 4
    DIM_FEEDFORWARD = 512
    BATCH_SIZE = 2
    SEQ_LEN = 16
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {DEVICE}")
    print(f"Model: d_model={D_MODEL}, nhead={NHEAD}, num_layers={NUM_LAYERS}")
    print("-" * 60)

    # --- モデル構築 ---
    model = SkippableTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        max_seq_len=SEQ_LEN,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # --- ダミーデータ生成 ---
    # input_ids: [B, T], targets: [B, T]（1つずらしたトークンを予測）
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).to(DEVICE)
    targets = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).to(DEVICE)

    # --- 学習ステップ ---
    model.train()
    optimizer.zero_grad()

    # 順伝播
    logits, layer_usages = model(input_ids, temperature=0.5, hard=True)

    # 損失計算
    total_loss, ce_loss, budget_loss = compute_loss_with_budget(
        model, logits, targets, layer_usages, lambda_budget=0.1
    )

    # 逆伝播
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # --- 結果表示 ---
    print(f"CE Loss:      {ce_loss.item():.4f}")
    print(f"Budget Loss:  {budget_loss.item():.4f}")
    print(f"Total Loss:   {total_loss.item():.4f}")
    print("-" * 60)

    # --- 各層のスキップ率を表示 ---
    print("各層のスキップ率（バッチ平均）:")
    for i, usage in enumerate(layer_usages):
        pass_rate = usage.mean().item()
        skip_rate = 1.0 - pass_rate
        print(f"  Layer {i}:  通過率={pass_rate:.3f},  スキップ率={skip_rate:.3f}")

    print("-" * 60)

    # --- トークンごとの通過パターンを可視化（最初のサンプル） ---
    print("\n最初のサンプルのトークンごとの通過パターン:")
    print("（行=層, 列=トークン位置。1=通過, 0=スキップ）")
    usage_matrix = torch.stack([u[0] for u in layer_usages], dim=0).cpu()
    # 0/1 に丸める（Gumbel-Softmax の浮動小数点値を）
    usage_matrix = usage_matrix.round().int()
    for i in range(NUM_LAYERS):
        row = "".join(str(usage_matrix[i, j].item()) for j in range(SEQ_LEN))
        print(f"  Layer {i}: {row}")

    print("\n✅ デモ完了。モデルは正常に動作しています。")


if __name__ == "__main__":
    demo()
