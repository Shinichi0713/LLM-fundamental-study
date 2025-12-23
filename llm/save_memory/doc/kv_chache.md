 **KVキャッシュ（Key-Value Cache）** は、LLMが文章を生成する際の「計算の無駄」を省き、生成速度を劇的に向上させるための最も重要な推論最適化技術の一つです。

なぜこの技術が必要なのか、LLMの「生成の仕組み」に潜む課題から解説します。


### 1. 解決しようとした課題：再計算の無駄

LLM（デコーダオンリーモデル）は、**「自己回帰（Autoregressive）」** という性質を持っています。これは、次の1単語を予測するために、過去のすべての単語を読み直す必要があるという仕組みです。

#### KVキャッシュがない場合の問題点

例えば、「私は / リンゴ / を」の後に続く言葉（食べた）を予測する場合を考えます。

1. 「私は」から「リンゴ」を予測する際、**「私は」**の K（Key）と V（Value）を計算します。
2. 「私は リンゴ」から「を」を予測する際、再び**「私は」**と**「リンゴ」**の両方の K と V を計算し直します。
3. 「私は リンゴ を」から「食べた」を予測する際、また最初から**「私は」「リンゴ」「を」**の K と V をすべて計算し直します。

このように、文章が長くなればなるほど、**過去に一度計算したはずの K と V を何度も何度も計算し直す**ことになり、計算量が雪だるま式に増えてしまいます。これが原因で、生成が非常に遅くなってしまうのです。


### 2. KVキャッシュの概要：計算結果の「保存」

KVキャッシュは、この **「二度手間」を解消するメモ帳** のような役割を果たします。

#### 仕組み

1. **保存**: あるトークン（単語）を処理した際に生成された **K（Key）ベクトル** と **V（Value）ベクトル** を、GPUのメモリ上に保存しておきます。
2. **再利用**: 次のトークンを生成する際は、 **新しく追加されたトークンの K と V だけ** を計算します。
3. **結合**: 保存しておいた過去の K, V キャッシュと、新しい K, V をガッチャンコ（結合）してアテンション計算を行います。


### 3. KVキャッシュ導入の効果

| 項目 | キャッシュなし | キャッシュあり |
| --- | --- | --- |
| **計算量** | トークンが増えるたびに爆発的に増加 | **常に一定（最新の1語分のみ）** |
| **生成速度** | 文章が長くなるほど遅くなる | **最後まで高速なまま** |
| **GPUの役割** | 演算ユニット（演算）を酷使する | **メモリ帯域（読み出し）が主役になる** |

### 4. 新たな課題：メモリ（VRAM）の圧迫

KVキャッシュによって「計算速度」の課題は解決されましたが、代わりに従うのが**「メモリ消費量」**という課題です。

* **VRAMを大量に喰う**: 過去の情報をすべてメモリに載せ続ける必要があるため、長い文章（ロングコンテキスト）を扱うほど、GPUのメモリを圧迫します。
* **共有化技術への繋がり**: この「メモリが足りない！」という問題を解決するために、最初にお話しした **MQA** や **GQA**（KeyとValueの数を減らしてキャッシュを節約する技術）が誕生したのです。

### まとめ：推論エンジンの心臓部

KVキャッシュは、**「計算時間をメモリで買う」**というトレードオフの技術です。これがあるおかげで、ChatGPTのようなスムーズな文字生成が実現できています。


## KVキャッシュの効果

KVキャッシュの効果を定量的に評価した研究は数多くあり、その結論は一貫して　**「計算量を劇的に削減する一方で、メモリ帯域（読み出し速度）と容量が新たなボトルネックになる」**　というものです。

主要な論文や技術レポートから、いくつかの具体的な指標を挙げて解説します。

### 1. 理論的な計算量の削減

最も基本的な評価は、計算の計算量（アルゴリズムの複雑性）の変化です。

* **キャッシュなし**: 1トークン生成するごとに過去の全トークンを再処理するため、生成ステップごとの計算量はシーケンス長  に対して **** で増加します（個のトークンを生成する総計算量は ****）。
* **キャッシュあり**: 過去の情報を再利用するため、1ステップあたりの計算量は **** に抑えられます（総計算量は ****）。

ある解析（Mandeep Singh, 2024）では、7Bクラスのモデルにおいて、文脈が長くなるほどキャッシュによる計算効率の向上は**4,000倍〜16,000倍**に達すると試算されています。


### 2. レイテンシ（遅延）の劇的な低下

「Keyformer（MLSys 2024）」などの論文では、キャッシュなしの状態との比較が示されています。

* **推論遅延の増加**: キャッシュを使用しない場合、文脈が長くなるにつれて推論レイテンシは**最大で50倍以上**に膨れ上がることが報告されています。
* **データ移動のコスト**: 一方で、キャッシュを使用すると「計算」自体は速くなりますが、推論時間の**約40%**が「過去のKVキャッシュをメモリから読み出すためのデータ移動」に費やされるようになります。これが、現代のLLMが「演算性能」よりも「メモリ帯域（HBM）」を重視する最大の理由です。


### 3. スループット（処理能力）の向上

システムレベルでの評価としては、**vLLM**（PagedAttention）の論文（Kwon et al., 2023）が有名です。

* **スループットの改善**: 効率的なKVキャッシュ管理（PagedAttention）を導入することで、従来の標準的な推論エンジンと比較して、**2倍〜4倍のスループット向上**を達成したことが定量的に示されています。
* **メモリの有効活用**: 従来の固定的なキャッシュ確保では、メモリの **60%〜80% が未使用（断片化）** でしたが、これを改善することでより大きなバッチサイズでの処理を可能にしました。


### 4. メモリ消費のトレードオフ

計算速度と引き換えに、メモリ消費量は以下のように定量化されます。

> **計算例（FP16精度の場合）**:
> 1トークンあたりのKVキャッシュサイズ =  (bytes)
> * **Llama-7B (32層, 4096次元)** の場合：
> 1トークンごとに約 **0.5MB** のメモリを消費します。
> * **1000トークン保持する場合**：
> 約 **500MB** がキャッシュだけで占有されます。
> 
> 




## 実装
KVキャッシュを自作モデルやPyTorchで実装する場合、主な変更点は **「Attention（注意機構）層」** と **「推論ループ（生成ループ）」** の2箇所です。

実装の肝は、各レイヤーで計算した  と  をリスト（またはタプル）として保持し、次のステップの入力に結合することです。


### 1. Attention層での実装（PyTorch）

通常のAttentionに、過去の  を受け取り、新しい  を返す機能を追加します。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttentionWithCache(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head = n_head
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, past_key_value=None):
        # x: [batch, 1, d_model]  (推論時は通常、最新の1トークンのみ入力)
        batch, seq_len, d_model = x.shape
        head_dim = d_model // self.n_head

        # 1. Q, K, V の計算
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        
        # [batch, n_head, seq_len, head_dim] に変換
        q = q.view(batch, seq_len, self.n_head, head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_head, head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_head, head_dim).transpose(1, 2)

        # 2. KVキャッシュの適用
        if past_key_value is not None:
            prev_k, prev_v = past_key_value
            # 過去のK, Vと現在のK, Vを結合 (seq_len方向に結合)
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)
        
        # 今回のステップのK, Vを保存用として出力
        present_key_value = (k, v)

        # 3. Attention計算 (Qは最新の1つ、K, Vは過去すべて)
        # q: [b, h, 1, d], k: [b, h, total_seq, d]
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        attn_probs = F.softmax(attn_weights, dim=-1)
        
        out = torch.matmul(attn_probs, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        
        return self.out_proj(out), present_key_value

```


### 2. 推論ループでの実装

モデル全体で各レイヤーのキャッシュを管理する必要があります。

```python
def generate_with_cache(model, tokenizer, prompt, max_new_tokens=20):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # 全レイヤー分のキャッシュを保持するリスト (最初はNone)
    past_key_values = None
    
    generated_ids = input_ids
    next_token_id = input_ids

    for _ in range(max_new_tokens):
        # モデルのforwardにキャッシュを渡す
        # 最初のステップ以外は、最新の1トークン(next_token_id)だけ入力すれば良い！
        outputs = model(next_token_id, past_key_values=past_key_values)
        
        logits = outputs.logits  # [batch, 1, vocab_size]
        past_key_values = outputs.past_key_values # 更新されたキャッシュを受け取る
        
        # 次のトークンを選択
        next_token_id = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
        
        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
        
        if next_token_id == tokenizer.eos_token_id:
            break
            
    return tokenizer.decode(generated_ids[0])

```


### 3. 実装のポイントと注意点

#### ① 入力サイズの削減

KVキャッシュを使う最大のメリットは、2ステップ目以降の入力 `x` のサイズを `[batch, 1, d_model]`（つまり **1トークンだけ**）にできることです。キャッシュがない場合は常に `[batch, 現在の全長さ, d_model]` を入力し、全計算をやり直す必要があります。

#### ② 位置エンコーディング（RoPEなど）の扱い

位置エンコーディング（Rotary Positional Embeddings）を使用している場合、**「今のトークンが全体の何番目か」**という絶対的な位置情報を正しく渡す必要があります。

* キャッシュを使う場合、`input_ids` は1つですが、その位置インデックス（`position_ids`）は `5` や `10` といった進んだ値にする必要があります。

#### ③ メモリ管理

`torch.cat` は計算のたびに新しいメモリを確保するため、非常に長い文章では非効率になることがあります。商用の高速推論エンジン（vLLMなど）では、**PagedAttention** という、OSの仮想メモリのようにキャッシュを飛び飛びのメモリブロックで管理する、より高度な実装が使われています。


### まとめ

1. **保存**: `(k, v)` を各層で計算して返す。
2. **再利用**: 次の入力時に前回の `(k, v)` を `past_key_value` として受け取る。
3. **結合**: `torch.cat` で過去分と今回分をくっつける。


