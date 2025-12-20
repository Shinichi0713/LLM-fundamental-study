# **Global Attention**

**グローバルアテンション（Global Attention）** は、スパースアテンションの一種として、長距離の依存関係を効率的に捕捉するために非常に重要な役割を果たします。

その特徴、メリット、そして採用している主要なAIモデルについて解説します。

---

## 🌟 グローバルアテンションの特徴

グローバルアテンションは、ローカルアテンション（近隣トークンのみに注目）の弱点である「長距離の文脈の欠落」を補うために、**特定のトークンを優遇する**仕組みを採用しています。

### 1. 特徴：情報ハブの設置

グローバルアテンションの核となるのは、シーケンス内のごく少数のトークンを**「グローバルトークン」**として指定することです。

* **グローバルトークン** ：
* **全てのトークンに注目する（Attends to all tokens）** 。これにより、シーケンス全体の情報を収集し、文脈の要約を行います。
* **全てのトークンから注目される（Is attended by all tokens）** 。これにより、他の全てのトークンがこのグローバルトークンを通じて、間接的に長距離の情報を得ることができます。
* **ローカルトークン（残りの大多数のトークン）** ：
* 通常は、自身を中心とした**ローカルな窓**内のトークンに注目します。
* さらに、このローカルな窓に加えて、**すべてのグローバルトークン**にも注目します。

### 2. スパース化のパターン

この混合型のパターンにより、Attentionマップは以下のような構造になります。

* **帯状（バンド）の接続** ：ローカルトークン同士の接続（Local Attention）。
* **行・列の接続** ：グローバルトークンと、残りの全てのトークンとの接続（Global Attention）。

これにより、計算量は **$O(N^2)$** ではなく、例えば **$O(N \cdot W + N \cdot G)$** （**$W$** は窓サイズ、**$G$** はグローバルトークン数）のように、シーケンス長 **$N$** に対して線形に近い形で抑えられます。

---

## 📈 グローバルアテンションのメリット

グローバルアテンションを導入するメリットは、主に以下の3点です。

1. 長距離依存関係の捕捉:
   ローカルアテンションでは失われる、遠く離れたトークン間の重要な依存関係を、グローバルトークンが情報の中継役となることで効率的に捕捉できます。これは、文書要約や質問応答など、文書全体を理解する必要があるタスクで特に重要です。
2. 計算効率の維持:
   全てのトークン間の接続を維持するわけではないため、Transformerのボトルネックである $O(N^2)$ の計算コストを回避し、長大な入力シーケンスの処理を可能にします。
3. 柔軟な文脈要約:
   グローバルトークンは、シーケンスの全体的な主題や最も重要な情報を集約する「サマリーノード」として機能するため、モデルがタスクに必要な文脈を柔軟に構築できます。

---

## 🤖 採用されているAIモデルの例

グローバルアテンション、またはこの概念を組み込んだ混合型のスパースアテンションは、特に長文処理に特化したモデルで採用されています。

### 1. Longformer

* **概要** : Facebook AIによって開発されたTransformerモデルで、非常に長い文書（最大4096トークン以上）を処理できるように設計されています。
* **採用パターン** : **ローカルアテンション**を基本としつつ、特定の事前定義されたトークン（例：`[CLS]`トークンや、特定のタスクに特化したトークン）に**グローバルアテンション**を適用することで、長距離の文脈を効果的に保持しています。

### 2. BigBird

* **概要** : Googleによって開発されたTransformerモデルで、Longformerと同様に長いシーケンスを処理できます。
* **採用パターン** : **ローカルアテンション**に加えて、 **ランダムなアテンション** 、そして**グローバルアテンション**の3種類のスパースアテンションを組み合わせることで、理論上、線形の計算量 **$O(N)$** を実現しています。

このように、グローバルアテンションは、限られた計算資源の中でTransformerモデルの「長距離記憶」能力を向上させるための、実用的で効果的な解決策となっています。


# **ハイブリッド疎結合アテンション（Hybrid Sparse Attention）**メカニズム

このメカニズムは、通常のTransformerが抱える$O(T^2)$（シーケンス長の2乗）の計算負荷を、$O(T)$（線形）に抑えつつ、長距離の依存関係を捉える能力を維持することを目的としています。

---

## コードの主要な機能と構造

このコードは、主に2つの部分から構成されています。

### 1. `unfold_kv` 関数 (ローカルウィンドウ抽出)

これは、ローカルアテンション（近傍トークンへの注意）に必要な**K（Key）**と**V（Value）**を効率的に抽出するためのユーティリティ関数です。

* **機能:** 入力シーケンス `x`（KまたはV）の各トークンに対し、そのトークンを中心とした固定長の**ウィンドウ（近傍）**に含まれるトークンを抜き出します。
* **技術:** PyTorchの`F.unfold`関数（通常は畳み込みネットワークのプーリングなどで使われる）を1次元シーケンスに応用することで、ループを使わずに高速にウィンドウ抽出を行っています。
* **出力形状:** `(B, H, T, window_len, D)`。これは、「バッチサイズ、ヘッド数、トークン数、**ウィンドウ内のトークン数**、ヘッドの次元」を意味します。

### 2. `HybridSparseAttention` クラス (ハイブリッドアテンションの実装)

このクラスは、**ローカルアテンション**と**グローバルアテンション**を組み合わせて最終的な出力を計算します。

#### A. ローカルアテンションの計算

1.  `unfold_kv` を使用して、KとVのローカルウィンドウ `K_windows` と `V_windows` を抽出します。
2.  各クエリ `Q` とローカルウィンドウ内のキー `K_windows` との内積を計算し、**ローカルスコア**を算出します。

#### B. グローバルアテンションの計算

1.  入力された `global_mask`（グローバルトークンを示すマスク）に基づき、シーケンス全体からグローバルトークンに対応するKとV（`K_global` および `V_global`）を収集します。
2.  このとき、バッチごとにグローバルトークンの数が異なる可能性があるため、最大長 `maxG` に合わせてパディングを行い、**マスク**を適用します。
3.  各クエリ `Q` とグローバルキー `K_global` との内積を計算し、**グローバルスコア**を算出します。

#### C. スコアの結合と出力

1.  ローカルスコアとグローバルスコアを最後の次元で**結合（`torch.cat`）**します。
2.  結合されたスコア全体に対して **Softmax** を適用することで、各トークンが「ローカルの近傍」と「すべてのグローバルトークン」のどちらにどれだけ注意を払うかを決定します。
3.  この重み（`w_local` と `w_global`）を使って、対応するV（`V_windows` と `V_global`）を重み付けし、**ローカルコンテキスト**と**グローバルコンテキスト**を計算し、足し合わせます。
4.  ヘッドを統合（`merge heads`）し、最終的な出力射影（`out_proj`）を通します。

#### D. 可視化のためのフルアテンションマップ

* コードの最後では、計算された疎結合な重みを、全結合アテンションと同様の形状 `(B, H, T, T)` の**`full_attn`**テンソルにマッピングし直す処理が含まれています。
* これは、実際の計算には使用されませんが、**どのトークンがどのトークンに注意を払っているか**を可視化（デバッグや解析）するために役立ちます。

### Big Birdにおける疎結合アテンションの仕組み

Big Birdの疎結合アテンションは、このコードで実装されている「ローカルアテンション」と「グローバルアテンション」に加え、**「ランダムアテンション」**の3つを組み合わせることで、**全結合アテンションの特性を維持**しながら、計算負荷を線形に抑えます。



この実装では、ローカルとグローバルのハイブリッド構造に焦点を当てており、疎結合アテンションの中核となる仕組みを効果的に再現しています。



以下では **Sparse Attention（特に BigBird / Longformer 系）の Global Attention の最小実装例** を示します。
✔ **特殊トークン（global tokens）を使う**
✔ **global token は全トークンと相互注意できる**
✔ **その他のトークンは local / sliding-window attention のみ**

という仕組みです。

---

# ✅ **Sparse Attention の Global Attention の最小実装（PyTorch）**

以下は **local + global attention** を備えた簡易版の実装コードです。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseGlobalAttention(nn.Module):
    def __init__(self, dim, num_heads=8, window=4):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window = window
        self.kernel_size = 2*window + 1

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, global_mask):
        """
        x: (B,T,D)
        global_mask: (B,T) bool — True の位置がグローバルトークン
        """
        B,T,D = x.shape

        # ---- 1) Project to multi-head ----
        q = self.q_proj(x).view(B,T,self.num_heads,self.head_dim).transpose(1,2)  # (B,H,T,Dh)
        k = self.k_proj(x).view(B,T,self.num_heads,self.head_dim).transpose(1,2)
        v = self.v_proj(x).view(B,T,self.num_heads,self.head_dim).transpose(1,2)

        # ============================================================
        # 2) Local Attention: sliding window (各クエリは近傍のみ参照)
        # ============================================================
        local_scores = torch.zeros(B,self.num_heads,T,self.kernel_size, device=x.device)

        for t in range(T):
            L = max(0, t-self.window)
            R = min(T, t+self.window+1)
            k_local = k[:,:,L:R,:]          # (B,H,win,Dh)
            q_t = q[:,:,t:t+1,:]            # (B,H,1,Dh)
            score = torch.einsum("bhid,bhjd->bhij", q_t, k_local) / (self.head_dim**0.5)
            local_scores[:,:,t,:R-L] = score.squeeze(2)

        # ============================================================
        # 3) Global Attention 部分
        # ============================================================
        # global_mask = True のトークンだけ全部に対して attention！
        global_scores = []

        for b in range(B):
            idx = torch.nonzero(global_mask[b], as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                global_scores.append(None)
                continue

            k_g = k[b,:,idx,:]  # (H, G, Dh)
            v_g = v[b,:,idx,:]

            # Q 全部のトークン → global tokens のスコア
            score = torch.einsum("htd,hgd->htg", q[b], k_g) / (self.head_dim**0.5)
            global_scores.append(score)     # (H,T,G)

        # ============================================================
        # 4) Local + Global を結合して softmax
        # ============================================================
        out_heads = torch.zeros(B,self.num_heads,T,self.head_dim, device=x.device)

        for b in range(B):
            for h in range(self.num_heads):
                for t in range(T):
                    # local
                    L = max(0, t-self.window)
                    R = min(T, t+self.window+1)
                    local_s = local_scores[b,h,t,:R-L]

                    if global_scores[b] is None:
                        # local only
                        attn = F.softmax(local_s, dim=-1)
                        v_loc = v[b,h,L:R,:]
                        ctx = torch.sum(attn.unsqueeze(-1) * v_loc, dim=-2)
                    else:
                        # local + global
                        g_score = global_scores[b][h,t]   # (G)
                        s = torch.cat([local_s, g_score], dim=-1)   # (win+G)
                        attn = F.softmax(s, dim=-1)

                        # 分割
                        attn_loc = attn[:R-L]
                        attn_g   = attn[R-L:]

                        v_loc = v[b,h,L:R,:]
                        v_g   = v[b,h, global_mask[b], :]

                        ctx = (
                            torch.sum(attn_loc.unsqueeze(-1)*v_loc, dim=-2) +
                            torch.sum(attn_g.unsqueeze(-1)*v_g, dim=-2)
                        )

                    out_heads[b,h,t] = ctx

        # ---- merge heads ----
        out = out_heads.transpose(1,2).reshape(B,T,D)
        out = self.out_proj(out)
        return out
```

---

# ✅ **この Sparse Global Attention の実装が再現している内容**

## **① ローカルアテンション（Local / Sliding Window）**

* 各トークンは
  **近傍の 2×window + 1 個の token にのみ attention**
  → 計算量を **O(n * window)** に削減

---

## **② グローバルトークン（Global Token）**

global_mask=True の位置（例：
CLS, 見出し文、特別なキーワードなど）は…

### **✔ 全トークンと attention できる（Global → All）**

### **✔ 全トークンから attention を受けられる（All → Global）**

これは Longformer / BigBird などが採用。

---

# ✅ **特殊トークンはどんな動作をする？**

### **📌 Global token の特徴**

* 全てのトークンと相互注意
* つまり **密な attention** を行う
* 文書全体の情報を集約・伝搬する役割
* Global token の例：

  * **[CLS]**
  * 要約タスクでの特別トークン
  * セクションタイトル
  * 質問応答での「質問トークン」

### **📌 実装上の動作**

* global_mask=True の token は k / v を特別に抽出し
  全 Query がそこに注意できるようにする。

コードの該当箇所：

```python
idx = torch.nonzero(global_mask[b], as_tuple=False).squeeze(-1)
k_g = k[b,:,idx,:]
v_g = v[b,:,idx,:]
score = torch.einsum("htd,hgd->htg", q[b], k_g)
```

---

# まとめ

| 機能                   | 実装での処理                                               |
| -------------------- | ---------------------------------------------------- |
| **Local Attention**  | sliding window の k_local / v_local を使用               |
| **Global Attention** | global_mask=True の k,v を抽出し、全 token の Q から attention |
| **特殊トークン**           | global_mask=True の token。全トークンと双方向 attention が可能     |
| 計算量                  | O(n × window + n × G)（G は global token の数）           |



### 実装法

Sparse Attention（スパース・アテンション）における **Global Attention（グローバル・アテンション）** の実装は、計算量を節約しつつ「文脈全体を俯瞰するハブ」を作るための非常に賢い仕組みです。

代表的な手法である **Longformer** や **BigBird** を例に、その実装の仕組みをステップバイステップで解説します。


__1. 全体像：混合アテンション・パターン__

通常の（Denseな）Attentionは  の全組み合わせを計算しますが、実装上は以下の2つの「窓」を組み合わせることで効率化します。

* **Local Attention (Sliding Window):** 各トークンは「自分の前後  個」だけを見る。
* **Global Attention:** 特定の重要なトークン（または場所）だけは「全トークン」を見る。

__2. 具体的な実装の仕組み__

実装の核となるのは、**「アテンション・マスク（Attention Mask）」のカスタマイズ**です。

__① グローバル・トークンの選定__

まず、どの位置を「グローバル」にするかを決めます。

* **特殊トークン:** `[CLS]` トークン（文章全体の意味を要約するトークン）など。
* **ユーザー指定:** 質問応答タスクなら「質問部分」のトークンすべて。

__② マスク行列の書き換え__

計算コードレベルでは、Attention計算に使う  のマスク行列を以下のように構成します。

1. **ベース:** すべてを `0` (注目しない) で初期化。
2. **Local:** 対角線付近（自分と周辺  個）を `1` (注目する) に書き換える。
3. **Global (行方向):** グローバルに設定されたトークン（行）について、**その行のすべてを `1` に書き換える。**（＝そのトークンは全員を見る）
4. **Global (列方向):** グローバルに設定されたトークン（列）について、**その列のすべてを `1` に書き換える。**（＝全員がそのトークンを見る）

__3. PyTorchでの簡易的な実装イメージ__

実際の大規模モデルでは専用のCUDAカーネルを使いますが、概念的な実装は以下のようになります。

```python
import torch
import torch.nn.functional as F

def sparse_global_attention(q, k, v, global_mask, window_size=5):
    # q, k, v: [batch, seq_len, dim]
    seq_len = q.size(1)
    
    # 1. 通常のドット積スコア計算
    scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1)**0.5)
    
    # 2. 基本のマスク（スライディングウィンドウ）を作成
    # 対角線付近以外を -inf に飛ばす
    mask = torch.ones((seq_len, seq_len), device=q.device)
    mask = torch.triu(mask, diagonal=window_size) + torch.tril(mask, diagonal=-window_size)
    
    # 3. GLOBALアテンションの適用
    # global_mask は [seq_len] の 0/1 ベクトル（1がグローバルトークン）
    # グローバルトークンに対応する「行」と「列」のマスクを解除(0にする)
    for i in range(seq_len):
        if global_mask[i] == 1:
            mask[i, :] = 0 # このトークンは全員を見る (行)
            mask[:, i] = 0 # 全員がこのトークンを見る (列)
            
    # 4. マスク適用とSoftmax
    scores = scores.masked_fill(mask.bool(), float('-inf'))
    probs = F.softmax(scores, dim=-1)
    
    return torch.matmul(probs, v)

```

__4. なぜこれで長距離依存が解けるのか？__

この実装のポイントは、グローバルトークンが **「情報のハブ（中継地点）」** になることです。

* トークンAとトークンBが数万単語離れていても、両者が「グローバルトークン（例：[CLS]）」に注目していれば、**A → [CLS] → B** というわずか2ステップで情報が伝達されます。
* これにより、計算量を （線形）に抑えつつ、理論上は文章の端から端までの関係性を保持できるようになります。

### まとめ

1. **Local窓**で近くの細かい文脈を拾う。
2. **Globalトークン**を「情報の交差点」として設定する。
3. **マスク操作**によって、その交差点だけは全方向への通行を許可する。

これが、最新の長文処理モデルで使われているGlobal Attentionの実装の本質です。



