先日うまくいかなかった自作VLMを再チャレンジしようと思います。
タスクはImage Captionで、Image Captionをうまく理解できるようにすることを目的として検討します。

## 目的
目的を整理します。

> **Google Colab 上で動く範囲で、先ほどの最小VLM（CLIP + Q-Former + GPT-2）よりVQA性能を明確に向上させたい**

この条件では、単にモデルを大きくするよりも、

* **Image Captionに適したアーキテクチャ**
* **事前学習済みマルチモーダル表現の活用**
* **Colabで学習・推論可能な計算量**

のバランスが重要です。

## 原因考察
先日の構成（**CLIP + Q-Former + GPT-2**）で **Image Caption が生成されなかった（`tensor([[50256]])` → 空出力）件**は、単一の不具合というより、**構成的に起きやすい失敗要因が重なった典型例**です。

### 結論（最も本質的な理由）

**GPT-2は「条件付き生成（conditioning）」に弱く、Q-Former経由の視覚情報をほぼ利用できていない可能性が高い**

その結果：

* 視覚特徴を無視
* 開始直後に `<eos>` を出力
* → 空キャプション

`50256` は GPT-2 の **eos token** です。


### 失敗の主因（構造レベル）

#### ① GPT-2はEncoder-Decoderではない

GPT-2は：

* **Decoder-only**
* 本来の用途：次トークン予測（テキスト続き生成）

今回の構成：

```
Image → CLIP → Q-Former → GPT-2
```

ここで問題：

* GPT-2には「外部特徴を強制的に使う仕組み」がない
* Cross-Attentionが標準では存在しない

結果：

**条件無視問題**

学習中も：

```
P(token | previous tokens)
```

だけで最適化されるため、

視覚特徴を使うよりも

```
即EOS
```

の方が損失が小さくなるケースが多い。
→GPT-2は損失最小のため、EOSをすぐに出してダメージ最小化を狙ってた可能性があります。

#### ② 学習データ量が少なすぎる

Flickr8k：

* 6,000〜8,000画像
* GPT-2: 1億パラメータ

これは実質：

> 視覚条件付き言語生成をゼロから学習させている状態

結果：

* 条件無視
* モード崩壊
* EOS出力


#### ③ Q-Former→GPT-2接続の情報量不足

典型的な実装：

```
Q-Former出力 → Linear → GPT-2 embedding次元
```

問題：

* GPT-2はprefix conditioningに弱い
* 数個のトークンでは影響が消える

BLIP2がOPT/T5を使う理由はここです。

#### ④ 学習安定性の問題（実務的に多い）

もし以下条件なら、ほぼ失敗します：

* GPT-2フル学習
* batch小
* learning rate大きい
* LoRAなし

結果：

* Language prior崩壊
* 生成不能
* EOS固定

#### ⑤ Attention mask警告（ログの内容）

先日でていたログ：

> pad token is same as eos token

これは：

* GPT-2のpad設定不適切
* 学習時にEOSが過学習される

これもEOS出力を助長します。


### 今回構成
文章生成するLLMをGPT-2からFlan-T5に変更しようと思います。
なぜFlan-T5では改善するのかですが。

|                 | GPT-2        | Flan-T5         |
| --------------- | ------------ | --------------- |
| 構造              | Decoder-only | Encoder-Decoder |
| 外部条件            | 弱い           | 強い              |
| Cross-Attention | なし           | あり              |
| 少量データ           | 弱い           | 比較的強い           |
| VLM用途           | 不向き          | 向いている           |

### 今回の症状の典型パターン

```
tensor([[50256]])
Generated caption:
```

これはVLM実験では：

> 条件無視 + 即EOS = Conditioning failure

条件を考えるのをやめてしまってるとされる状態です。

今回うまくいかなかった主因をまとめると次だと考えています。

1. GPT-2が外部特徴を活用できない構造
2. データ量不足
3. Prefix conditioningの弱さ
4. EOSへの損失バイアス

---

# 研究的な示唆（ここが重要）

この結果はむしろ**正しい挙動**です。

なぜなら：

> GPT-2 + Q-Former は、小規模VLMでは成立しにくい

実際のVLM：

* BLIP2 → OPT / T5
* LLaVA → LLaMA
* Flamingo → gated cross-attention

すべて：

**Cross-Attention型**

---

# もし改善するなら（優先度順）

① GPT-2にCross-Attention追加
② LoRAのみ学習
③ EOS weightを下げる
④ Caption長を制約
⑤ データ増量

ただし、実験効率を考えると：

**Flan-T5-smallに切り替える方が圧倒的に良い**

---

必要なら次に、

**なぜFlan-T5 + Q-Formerが小規模実験で最も成功率が高いか（理論 + 実験的理由）**

を、VLM設計の観点で整理できます。これはモデル選定の判断基準としてかなり重要です。



# 結論（推奨構成）

## ◎ 推奨：**BLIP-2構成（Q-Former継続）＋ Flan-T5 + Instruction形式**

```
Image → CLIP ViT-L
       ↓
   Q-Former (trainable)
       ↓
   Projection
       ↓
Flan-T5-base (LoRA)
       ↓
Answer text
```

**これがColabで動く中では、VQA性能の伸びが最も大きい構成です。**

---

# なぜGPT-2構成より強いのか

先ほどの構成の弱点：

| 問題            | 影響          |
| ------------- | ----------- |
| GPT-2は指示理解が弱い | 質問条件を無視しやすい |
| Caption向け学習   | QA形式に弱い     |
| Decoder-only  | 条件理解能力が低い   |

VQAは実際には：

> **Image + Question → 条件付き生成**

つまり必要なのは：

* 条件理解
* instruction-following
* reasoning

Flan-T5はここが圧倒的に強いです。

---

# 推奨構成の詳細

## 1. Vision Encoder

**CLIP ViT-L/14**

理由：

* ViT-BよりVQA性能が明確に上
* Colabでもfreezeなら余裕

```python
openai/clip-vit-large-patch14
```

freeze推奨。

---

## 2. Q-Former

そのまま使用（BLIP-2形式）

設定：

* query tokens: **32〜64**
* trainable

VQAでは**query数を増やすと効きやすい**

---

## 3. LLM

### Flan-T5-base（推奨）

約 250M パラメータ

Colab可：

* LoRA使用
* fp16

代替：

* flan-t5-small（さらに軽量）
* flan-t5-large（Colab Pro）

---

# 入力フォーマット（重要）

VQAでは形式が性能に直結します。

### 入力テンプレート

```
Question: What color is the car?
Answer:
```

または

```
Based on the image, answer the question.
Question: ...
Answer:
```

Flanはinstruction形式で性能が上がります。

---

# 学習対象（Colab向け）

| 部分             | 設定     |
| -------------- | ------ |
| Vision Encoder | freeze |
| Q-Former       | train  |
| Projection     | train  |
| Flan-T5        | LoRAのみ |

これでVRAM 12GB前後で収まります。

---

# データセット（Colab向け）

性能を上げるなら：

## VQAv2（小サブセット）

例：

* trainから 20k〜50k サンプル抽出

理由：

* 多様な質問
* Captionより難易度高い
* 性能差が出やすい

---

# さらに性能を上げる重要ポイント

## ① Query数を増やす

```
num_query_tokens = 64
```

Captionでは32で十分ですが、VQAでは効きます。

---

## ② Questionをテキストトークンとして入力

LLM入力：

```
[Image tokens][Question tokens]
```

※ Captionとの違い

---

## ③ Answer長制限

```
max_new_tokens = 5〜10
```

VQAは短文が多い。

---

# Colabで動く「性能順」構成比較

| 構成                              | VQA性能     |
| ------------------------------- | --------- |
| CLIP + QFormer + GPT-2          | ★         |
| CLIP + QFormer + GPT-2(LoRA)    | ★★        |
| CLIP + QFormer + Flan-T5-small  | ★★★       |
| CLIP-L + QFormer + Flan-T5-base | ★★★★（推奨）  |
| BLIP-2そのまま                      | ★★★★★（重い） |

---

# さらに強い構成（余裕があれば）

## Option A: InstructBLIP（最強Colab枠）

```
Salesforce/instructblip-flan-t5-base
```

特徴：

* VQA特化済み
* Q-Former最適化済み
* 追加学習でさらに性能UP

---

## Option B: LLaVA-1.5-7B（Colab Pro）

ただし：

* VRAM要求高い
* 学習は厳しい

---

# なぜこの構成が今回の条件に最適か

今回の条件：

* Colabで動く
* 自作VLM延長
* 性能向上を確認したい

この構成は：

* Q-Formerの研究性を維持
* Instruction能力を導入
* 計算量は現実的
* 性能差が明確に出る

---

# 実験としての意義

この構成にすると、

以下が評価できます：

* GPT-2 vs Flan-T5 の比較
* Query数の影響
* Instructionの効果
* Captionモデル → VQA適応

研究的にも意味があります。

---

# まとめ

ColabでVQA性能を上げるなら：

**最もバランスの良い構成**

```
CLIP ViT-L (freeze)
+ Q-Former (64 queries)
+ Linear projection
+ Flan-T5-base (LoRA)
+ Instruction形式入力
+ VQAv2サブセット
```

---

必要なら次に、

* Colabでそのまま動く **最小VQA実装コード**
* VRAM使用量目安
* 学習ステップ数の現実ライン
* GPT-2構成との性能差の目安

まで具体的に出せます。
