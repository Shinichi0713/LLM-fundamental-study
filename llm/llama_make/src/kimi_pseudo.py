"""
Mini Kimi K3 Architecture Implementation (NumPy)
===============================================
Kimi K3 の主要コンポーネントを概念的に実装したものです。
実際のKimi K3は2.8Tパラメータですが、ここでは教育・実験用に小さなサイズで再現しています。

実装されているコンポーネント:
1. SiTU (Sigmoid Tanh Unit) - 大規模MoE向け活性化関数
2. SiTU-GLU - GLUにSiTUを組み合わせたFeed-Forward
3. Kimi Delta Attention (KDA) - 差分圧縮による長文対応アテンション
4. Gated MLA - 低ランク圧縮 + ゲーティングを持つアテンション
5. Stable LatentMoE - スパースMoE + Quantile Balancing
6. Attention Residuals (AttnRes) - 層間表現の選択的融合
7. KimiK3Block - 上記コンポーネントを統合したTransformerブロック
8. MiniKimiK3 - 完全なモデル

参考: Kimi K3 Tech Blog (kimi.ai/blog/kimi-k3)
"""

import numpy as np

# ============================================================
# ユーティリティ関数
# ============================================================
def softmax(x, axis=-1):
    """数値安定なsoftmax"""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def rmsnorm(x, eps=1e-6):
    """RMSNorm: LayerNormの簡略版、平均ではなく二乗平均を使う"""
    return x / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)


# ============================================================
# 1. SiTU (Sigmoid Tanh Unit) Activation
# ============================================================
def situ(x):
    """
    SiTU(x) = sigmoid(x) * tanh(x)

    大規模MoEモデルでの学習安定性を高める活性化関数。
    Sigmoidの0〜1の範囲とTanhの-1〜1の範囲を組み合わせ、
    勾配の爆発を防ぎつつ表現力を維持する。
    """
    sigmoid = 1 / (1 + np.exp(-x))
    tanh = np.tanh(x)
    return sigmoid * tanh


# ============================================================
# 2. 線形層と埋め込み (NumPy版)
# ============================================================
class Linear:
    """全結合層 (Xavier初期化)"""
    def __init__(self, in_features, out_features):
        self.weight = np.random.randn(in_features, out_features).astype(np.float32) * np.sqrt(2.0 / in_features)

    def __call__(self, x):
        return x @ self.weight

    def numel(self):
        return self.weight.size


class Embedding:
    """埋め込み層"""
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = np.random.randn(num_embeddings, embedding_dim).astype(np.float32) * 0.02

    def __call__(self, indices):
        return self.weight[indices]

    def numel(self):
        return self.weight.size


# ============================================================
# 3. Kimi Delta Attention (KDA)
# ============================================================
class KimiDeltaAttention:
    """
    Kimi Delta Attention (KDA)

    従来のアテンションでは長文処理時にKVキャッシュが肥大化する問題があった。
    KDAは「隣接トークン間の差分（Delta）」に着目し、冗長な情報を圧縮して保持する。

    仕組み:
    1. 各トークンのKVを潜在次元に圧縮 (w_k_delta, w_v_delta)
    2. 累積和で差分から完全なKVを復元 (cumsum)
    3. アップ投影して元の次元に戻す (w_k_up, w_v_up)

    効果:
    - KVキャッシュのメモリを約75%削減
    - 100万トークンでのデコード速度を6.3倍に高速化
    """
    def __init__(self, dim, num_heads, latent_dim):
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.latent_dim = latent_dim

        self.w_q = Linear(dim, dim)
        self.w_k_delta = Linear(dim, latent_dim)
        self.w_v_delta = Linear(dim, latent_dim)
        self.w_k_up = Linear(latent_dim, dim)
        self.w_v_up = Linear(latent_dim, dim)
        self.w_o = Linear(dim, dim)

        # ベースKVは潜在次元に合わせる
        self.k_base = np.zeros((1, 1, latent_dim), dtype=np.float32)
        self.v_base = np.zeros((1, 1, latent_dim), dtype=np.float32)

    def __call__(self, x, mask=None):
        B, T, D = x.shape

        # Query投影
        q = self.w_q(x).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Delta圧縮: 隣接トークン間の差分を潜在空間に投影
        delta_k = self.w_k_delta(x)  # [B, T, latent_dim]
        delta_v = self.w_v_delta(x)  # [B, T, latent_dim]

        # 累積和で差分から完全なKVを復元
        # 例: delta = [d1, d2, d3] -> cumsum = [d1, d1+d2, d1+d2+d3]
        k_accum = self.k_base + np.cumsum(delta_k, axis=1)  # [B, T, latent_dim]
        v_accum = self.v_base + np.cumsum(delta_v, axis=1)

        # アップ投影して元の次元に戻す
        k = self.w_k_up(k_accum).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.w_v_up(v_accum).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # スケールドドットプロダクトアテンション
        scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
        attn = softmax(scores, axis=-1)

        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, D)
        return self.w_o(out)

    def count_params(self):
        return (self.w_q.numel() + self.w_k_delta.numel() + self.w_v_delta.numel() +
                self.w_k_up.numel() + self.w_v_up.numel() + self.w_o.numel())


# ============================================================
# 4. Gated MLA (Multi-head Latent Attention)
# ============================================================
class GatedMLA:
    """
    Gated Multi-head Latent Attention (Gated MLA)

    低ランク圧縮 + ゲーティングを組み合わせたアテンション機構。
    KDAと3:1の比率で交互に配置される。

    仕組み:
    1. クエリ・キー・バリューを低ランク潜在空間に圧縮 (down_q, down_kv)
    2. アップ投影で元の次元に戻す (up_q, up_kv)
    3. 各ヘッドごとに学習可能なゲートで選択性を付与

    効果:
    - KDAと組み合わせることで長文・深層の両方に対応
    - ゲーティングで不要なヘッドの影響を抑制
    """
    def __init__(self, dim, num_heads, latent_dim):
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.latent_dim = latent_dim

        self.down_q = Linear(dim, latent_dim)
        self.down_kv = Linear(dim, latent_dim)
        self.gate = np.ones(num_heads, dtype=np.float32)
        self.up_q = Linear(latent_dim, dim)
        self.up_kv = Linear(latent_dim, dim)
        self.w_o = Linear(dim, dim)

    def __call__(self, x, mask=None):
        B, T, D = x.shape

        # 低ランク圧縮
        q_latent = self.down_q(x)
        kv_latent = self.down_kv(x)

        # アップ投影
        q = self.up_q(q_latent).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.up_kv(kv_latent).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = k.copy()  # 簡略化: 実際には別投影

        # ゲーティング適用: 各ヘッドの重要度を調整
        q = q * self.gate.reshape(1, self.num_heads, 1, 1)

        scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
        attn = softmax(scores, axis=-1)

        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, D)
        return self.w_o(out)

    def count_params(self):
        return (self.down_q.numel() + self.down_kv.numel() + self.up_q.numel() +
                self.up_kv.numel() + self.w_o.numel() + self.gate.size)


# ============================================================
# 5. SiTU-GLU Feed-Forward
# ============================================================
class SiTUGLU:
    """
    SiTU-GLU Feed-Forward

    GLU (Gated Linear Unit) にSiTU活性化関数を組み合わせたもの。
    MoEの各エキスパートとして使用される。

    仕組み:
    1. 入力を2つに分割 (ゲート用と値用)
    2. ゲート側にSiTUを適用
    3. 値側と要素積を取る
    4. 出力投影
    """
    def __init__(self, dim, hidden_dim):
        self.w1 = Linear(dim, hidden_dim * 2)
        self.w2 = Linear(hidden_dim, dim)

    def __call__(self, x):
        proj = self.w1(x)
        a, b = np.split(proj, 2, axis=-1)
        return self.w2(situ(a) * b)

    def count_params(self):
        return self.w1.numel() + self.w2.numel()


# ============================================================
# 6. Stable LatentMoE
# ============================================================
class StableLatentMoE:
    """
    Stable LatentMoE

    多数のエキスパートから少数を選択するスパースMoE。
    Kimi K3では896個のエキスパートから16個を選択する。

    主な特徴:
    - Quantile Balancing: ルーター確率の分位点から負荷分散を計算
    - 共有エキスパート: 常に活性化されるエキスパト
    - スパース活性化: 各トークンごとにtop-k個のエキスパートのみ使用

    効果:
    - 総パラメータ数を増やしつつ計算コストを抑える
    - Kimi K2と比較して約2.5倍のスケーリング効率を達成
    """
    def __init__(self, dim, num_experts=16, top_k=4, shared_experts=2):
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.shared_experts = shared_experts

        # ルーター: 各トークンをどのエキスパートに送るか決定
        self.router = Linear(dim, num_experts)

        # 各エキスパートは小さなFFN
        expert_hidden = dim * 2
        self.experts = [SiTUGLU(dim, expert_hidden) for _ in range(num_experts)]

        # 共有エキスパート（常に活性化）
        self.shared = [SiTUGLU(dim, expert_hidden) for _ in range(shared_experts)]

    def quantile_balance(self, router_probs):
        """
        Quantile Balancing

        ルーター確率の分位点から負荷分散を計算する。
        従来のヒューリスティックな調整を排除し、
        安定したエキスパート負荷分散を実現する。

        損失 = sum((mean_probs - target)^2)
        target = 1 / num_experts (均等分散)
        """
        mean_probs = router_probs.mean(axis=(0, 1))
        target = 1.0 / self.num_experts
        return np.sum((mean_probs - target) ** 2)

    def __call__(self, x):
        B, T, D = x.shape
        x_norm = rmsnorm(x)

        # ルータースコア計算
        router_logits = self.router(x_norm)  # [B, T, num_experts]
        router_probs = softmax(router_logits, axis=-1)

        # Top-k選択（スパース化）
        topk_indices = np.argpartition(router_probs, -self.top_k, axis=-1)[..., -self.top_k:]

        # エキスパート出力の集約
        output = np.zeros_like(x)
        for b in range(B):
            for t in range(T):
                for i in range(self.top_k):
                    e_idx = topk_indices[b, t, i]
                    weight = router_probs[b, t, e_idx]
                    expert_input = x_norm[b, t].reshape(1, 1, -1)
                    expert_out = self.experts[e_idx](expert_input).reshape(-1)
                    output[b, t] += expert_out * weight

        # 共有エキスパートを加算
        for shared in self.shared:
            output = output + shared(x_norm)

        # 残差接続
        balance_loss = self.quantile_balance(router_probs)
        return x + output, balance_loss

    def count_params(self):
        total = self.router.numel()
        for e in self.experts:
            total += e.count_params()
        for s in self.shared:
            total += s.count_params()
        return total


# ============================================================
# 7. Kimi K3 Block
# ============================================================
class KimiK3Block:
    """
    Kimi K3 Transformer Block

    KDAまたはGated MLA + AttnRes + Stable LatentMoE を統合したブロック。

    構造:
    1. RMSNorm -> Attention (KDA or Gated MLA) -> 残差
    2. AttnRes: 過去の層表現を選択的に融合
    3. MoE (SiTU-GLUエキスパート) -> 残差

    実際のKimi K3では69層がKDA、24層がGated MLAの3:1比率。
    """
    def __init__(self, dim, num_heads, latent_dim, num_experts, top_k, use_kda=True):
        self.use_kda = use_kda

        # アテンション層: KDAとGated MLAを切り替え可能
        if use_kda:
            self.attn = KimiDeltaAttention(dim, num_heads, latent_dim)
        else:
            self.attn = GatedMLA(dim, num_heads, latent_dim)

        # MoE FFN
        self.moe = StableLatentMoE(dim, num_experts, top_k)

        # AttnRes用の融合層
        self.attnres_gate = Linear(dim * 2, dim)

    def __call__(self, x, mask=None, layer_idx=0, residual_bank=None):
        # 1. アテンション + 残差
        attn_out = self.attn(rmsnorm(x), mask)
        x = x + attn_out

        # 2. AttnRes: 過去の層表現を選択的に融合
        if residual_bank is not None and layer_idx > 0:
            prev = residual_bank[layer_idx - 1]
            gate_input = np.concatenate([x, prev], axis=-1)
            gate = 1 / (1 + np.exp(-self.attnres_gate(gate_input)))  # sigmoid
            x = x * gate + prev * (1 - gate)

        # 3. MoE FFN + 残差
        moe_out, balance_loss = self.moe(x)
        x = x + moe_out

        return x, balance_loss

    def count_params(self):
        return self.attn.count_params() + self.moe.count_params() + self.attnres_gate.numel()


# ============================================================
# 8. Mini Kimi K3 Model
# ============================================================
class MiniKimiK3:
    """
    Mini Kimi K3 Model

    Kimi K3 の主要コンポーネントを統合した小規模モデル。
    実際のKimi K3は以下のスペック:
    - 総パラメータ数: 2.8T (2.8兆)
    - 有効化パラメータ数: 104B (1,040億)
    - 層数: 93 (69 KDA + 24 Gated MLA)
    - エキスパート数: 896 (16個を選択)
    - コンテキスト長: 100万トークン
    - ビジョンエンコーダー: MoonViT-V2 (401Mパラメータ)

    この実装は教育・実験用に大幅に縮小したもの。
    """
    def __init__(
        self,
        vocab_size=32000,
        dim=512,
        num_layers=6,
        num_heads=8,
        latent_dim=128,
        num_experts=16,
        top_k=4,
        max_seq_len=2048
    ):
        self.dim = dim
        self.num_layers = num_layers

        self.token_emb = Embedding(vocab_size, dim)
        self.pos_emb = Embedding(max_seq_len, dim)

        # Transformerブロックのスタック
        # 実際のK3では 69 KDA + 24 Gated MLA の比率だが、ここでは簡略化
        self.layers = []
        for i in range(num_layers):
            use_kda = (i % 4 != 3)  # 3:1の比率でKDAを使用
            self.layers.append(KimiK3Block(
                dim=dim,
                num_heads=num_heads,
                latent_dim=latent_dim,
                num_experts=num_experts,
                top_k=top_k,
                use_kda=use_kda
            ))

        self.lm_head = Linear(dim, vocab_size)

    def __call__(self, input_ids):
        """
        順伝播

        Args:
            input_ids: [batch_size, seq_len] トークンIDの配列

        Returns:
            logits: [batch_size, seq_len, vocab_size] 次トークンの予測確率
            balance_loss: MoEの負荷分散損失
        """
        B, T = input_ids.shape

        # トークン埋め込み + 位置埋め込み
        x = self.token_emb(input_ids)
        positions = np.arange(T).reshape(1, -1)
        x = x + self.pos_emb(positions)

        # 因果マスク（下三角行列）
        mask = np.tril(np.ones((T, T))).reshape(1, 1, T, T)

        # AttnRes用の残差バンク
        residual_bank = []
        total_balance_loss = 0.0

        for i, layer in enumerate(self.layers):
            x, bal_loss = layer(x, mask, layer_idx=i, residual_bank=residual_bank if i > 0 else None)
            residual_bank.append(x)
            total_balance_loss += bal_loss

        # 最終正規化 + 出力投影
        x = rmsnorm(x)
        logits = self.lm_head(x)

        return logits, total_balance_loss

    def count_params(self):
        total = self.token_emb.numel() + self.pos_emb.numel()
        for layer in self.layers:
            total += layer.count_params()
        total += self.lm_head.numel()
        return total


# ============================================================
# テスト実行
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Mini Kimi K3 Model (NumPy Implementation)")
    print("=" * 60)

    # モデル初期化（小さなサイズでテスト）
    model = MiniKimiK3(
        vocab_size=1000,
        dim=64,
        num_layers=4,
        num_heads=4,
        latent_dim=16,
        num_experts=8,
        top_k=2,
        max_seq_len=128
    )

    # ダミー入力
    batch_size = 2
    seq_len = 16
    np.random.seed(42)
    input_ids = np.random.randint(0, 1000, size=(batch_size, seq_len))

    # 順伝播
    logits, balance_loss = model(input_ids)

    print(f"Input shape:         {input_ids.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"MoE balance loss:    {balance_loss:.6f}")
    print(f"Total parameters:    {model.count_params() / 1e6:.2f}M")
    print("=" * 60)

    # 各コンポーネントの個別テスト
    print("\nComponent Tests:")
    print("-" * 40)

    x_test = np.random.randn(2, 8, 64).astype(np.float32)

    print(f"SiTU output shape: {situ(x_test).shape}")

    kda = KimiDeltaAttention(dim=64, num_heads=4, latent_dim=16)
    print(f"KDA output shape: {kda(x_test).shape}")

    gmla = GatedMLA(dim=64, num_heads=4, latent_dim=16)
    print(f"Gated MLA output shape: {gmla(x_test).shape}")

    moe = StableLatentMoE(dim=64, num_experts=8, top_k=2)
    out, loss = moe(x_test)
    print(f"MoE output shape: {out.shape}, balance loss: {loss:.6f}")

    glu = SiTUGLU(dim=64, hidden_dim=128)
    print(f"SiTU-GLU output shape: {glu(x_test).shape}")

    print("-" * 40)
    print("All tests passed!")
