# VLMの構造

VLM（Vision-Language Model）の「内部構造の基本」を理解することは、BLIP や CLIP、MiniGPT-4 などのモデルを深く理解する第一歩です。

以下では、**VLMの共通構造**をわかりやすく、**構成要素・データフロー・学習方式**の3段階で説明します。

---

# 🎯 VLM（Vision-Language Model）とは

VLMとは：

> 「画像とテキストを同じ意味空間で理解・生成するAIモデル」

を指します。

つまり「画像を見て文章を生成する」あるいは「テキストを読んで対応する画像を理解する」ための仕組みを持っています。

代表例：

* **CLIP（OpenAI）** ：画像とテキストを同じベクトル空間に埋め込む
* **BLIP / BLIP-2（Salesforce）** ：画像を見てキャプション生成・質問応答
* **LLaVA / MiniGPT-4** ：画像＋LLMによる対話

---

# 🧩 **VLMの内部構造の全体像**

VLMは、基本的に以下の3つの部分から構成されます👇

```
🖼️ Vision Encoder → 🔄 Cross-Modal Connector → 🗣️ Language Model
```

| パート                                   | 役割                       | 具体例                                  |
| ---------------------------------------- | -------------------------- | --------------------------------------- |
| **1️⃣ Vision Encoder**           | 画像をベクトル表現に変換   | CNN, ViT (Vision Transformer), CLIP-ViT |
| **2️⃣ Cross-Modal Connector**    | 画像とテキストを結びつける | Projection Layer, Q-Former              |
| **3️⃣ Language Model (Decoder)** | テキストを理解・生成       | BERT, T5, LLaMA, GPT など               |

---

## 1️⃣ Vision Encoder：画像 → 埋め込みベクトル

### 🔍 役割

画像を入力して、その内容を数値化（特徴量ベクトル）する部分です。

### 🧠 内部構造

多くの場合、**Vision Transformer (ViT)** が使われます。

```
画像 (224×224×3)
↓
パッチ分割 (16×16)
↓
埋め込み + 位置エンコーディング
↓
Transformer Encoder × N層
↓
出力ベクトル（例えば768次元）
```

### 💡 出力

* 各パッチごとの特徴量ベクトル（N×D）
* 画像全体を代表するCLSトークンベクトル（1×D）

### 📘 代表モデル

* **CLIP-ViT** （OpenAI）
* **BLIP-Vision Encoder**
* **ResNet** （古い構成）

---

## 2️⃣ Cross-Modal Connector（中間ブリッジ）

### 🔍 役割

Vision Encoderが出力した画像ベクトルを、言語モデルが理解できる形に変換します。

---

### 🧩 構成パターン

| 手法                                 | 仕組み                                       | 代表モデル       |
| ------------------------------------ | -------------------------------------------- | ---------------- |
| **投影層（Projection Layer）** | 線形変換で画像ベクトル→テキスト空間         | CLIP             |
| **Cross-Attention層**          | 画像トークンをテキストのAttentionに組み込む  | BLIP             |
| **Q-Former**                   | Queryベクトルで画像の要点を抽出し、LLMに渡す | BLIP-2           |
| **Adapter / LoRA**             | 軽量変換層でマルチモーダル拡張               | MiniGPT-4, LLaVA |

---

### 🧠 Q-Formerの例（BLIP-2）

```
Vision Encoder 出力: N個の画像トークン
↓
Q-Former（Transformer構造）
    Query Token × M個（学習可能）
    ↓
    Cross-Attentionで画像特徴と結合
↓
M個の Query 出力ベクトル（画像の意味的要約）
```

➡ これを **LLM（例：Flan-T5）** に入力として与えます。

---

## 3️⃣ Language Model（テキスト生成）

### 🔍 役割

画像情報をもとに自然言語（キャプション・回答）を生成します。

### 📘 構造

基本的には **Transformer Decoder** 型モデルです。

例：

* BLIP → BERT (Encoder-Decoder型)
* BLIP-2 → Flan-T5 / LLaMA など
* LLaVA → Vicuna（LLaMAベースLLM）

### 💡 入力形式

```
[画像特徴ベクトル] + [質問テキストトークン]
```

### 💬 出力例

```
"An orange cat sitting on a chair."
```

---

# 🔄 **VLMのデータフロー（例：BLIP-2）**

```
📸 画像 → Vision Encoder (ViT)
          ↓
      画像特徴トークン
          ↓
      Q-Former（Queryで要約）
          ↓
      要約ベクトル
          ↓
🧠 LLM（Flan-T5など）
          ↓
      "This is a cat sitting on a sofa."
```

---

# 🧠 **VLMの学習方法**

| 学習方式                                   | 内容                                         | 目的                        |
| ------------------------------------------ | -------------------------------------------- | --------------------------- |
| **Contrastive Learning（対照学習）** | 画像と対応するテキストを近づける（CLIP方式） | 意味空間の統一              |
| **Image-Text Matching（ITM）**       | 画像と文章が一致しているかを分類             | 概念の関連性理解            |
| **Captioning（生成学習）**           | 画像から文章を生成                           | 言語生成能力                |
| **Instruction Tuning**               | 「画像＋指示」に対する出力を学習             | 対話的画像理解（LLaVAなど） |

---

# 🧩 **VLMの代表アーキテクチャ比較**

| モデル              | Vision Encoder | Connector         | Language Model  | 主な用途           |
| ------------------- | -------------- | ----------------- | --------------- | ------------------ |
| **CLIP**      | ViT / ResNet   | Linear Projection | Text Encoder    | 画像検索・分類     |
| **BLIP**      | ViT            | Cross-Attention   | BERT-like       | 画像キャプション   |
| **BLIP-2**    | ViT            | Q-Former          | Flan-T5 / LLaMA | QA・説明生成       |
| **LLaVA**     | CLIP-ViT       | Linear Adapter    | Vicuna          | マルチモーダル対話 |
| **MiniGPT-4** | BLIP-2構造     | Q-Former改良版    | Vicuna          | ChatGPT的画像対話  |

---

# 📊 **図式まとめ（構造イメージ）**

```
        ┌────────────────────────────┐
        │        VISION ENCODER       │
        │ (ViT, CNN, CLIP-ViT etc.)   │
        └────────────┬───────────────┘
                     │ 画像特徴
        ┌────────────┴───────────────┐
        │     CROSS-MODAL CONNECTOR    │
        │ (Projection, Q-Former etc.)  │
        └────────────┬───────────────┘
                     │ 画像の意味表現
        ┌────────────┴───────────────┐
        │       LANGUAGE MODEL         │
        │ (T5, LLaMA, Vicuna, GPT etc.)│
        └────────────┬───────────────┘
                     ↓
              📝 テキスト出力
```

---

# 🚀 **まとめ：VLM理解の3ステップ**

| ステップ | 学ぶ内容                                  | 例                   |
| -------- | ----------------------------------------- | -------------------- |
| ①       | Vision Encoderがどう特徴を抽出するか      | ViT, CNN             |
| ②       | Cross-Modal層でどう画像と言語を接続するか | Projection, Q-Former |
| ③       | LLMがどうテキストを生成するか             | T5, LLaMA            |


# VLMの学習

LLM（Large Language Model）の学習経験を前提として、その後に **VLM（Vision-Language Model）** を学習・開発する際に、特に重要となる学習事項を体系的に整理します。
LLMとの「共通点」と「決定的な差分」を意識して学ぶことが効率的です。


## 1. LLMとVLMの本質的な違い（最重要）

### 1.1 モダリティの違い

| 観点 | LLM     | VLM             |
| -- | ------- | --------------- |
| 入力 | テキストのみ  | 画像＋テキスト         |
| 表現 | 離散トークン  | 連続特徴（画像）＋離散トークン |
| 課題 | 言語理解・生成 | 視覚と言語の対応付け      |

**重要ポイント**

* 画像は「トークン化」されない（連続特徴）
* 画像特徴を**どの層で・どう融合するか**がVLMの核心


## 2. 画像エンコーダ（Vision Encoder）の理解

### 2.1 主流アーキテクチャ

必須で理解すべきもの：

* **CNN系**（ResNet, EfficientNet）
* **Vision Transformer (ViT)**

  * Patch Embedding
  * CLS token
  * Self-Attention

現在の主流は **ViT系** です。

### 2.2 画像特徴の性質

* 空間構造を持つ（H×W×C）
* セマンティック抽象度は層ごとに異なる
* 言語特徴と**分布が異なる**

→ この「分布差」をどう吸収するかが設計上の重要点です。


## 3. マルチモーダル融合（Fusion）の設計

### 3.1 代表的な融合方式

| 方式           | 説明      | 代表例      |
| ------------ | ------- | -------- |
| Early Fusion | 入力段階で結合 | Rare     |
| Mid Fusion   | 中間層で結合  | Flamingo |
| Late Fusion  | 出力段階で結合 | CLIP系    |

### 3.2 現実的に重要な方式

**LLMベースVLMでは以下が主流**：

* 画像特徴 → **投影層（Projection Layer）**
* 擬似トークンとして LLM に入力

例：

```
[IMG_1][IMG_2][IMG_3] ... + Text Tokens
```

## 4. CLIP思想の理解（必須）

### 4.1 対照学習（Contrastive Learning）

* 画像とテキストを**同一埋め込み空間**に写像
* 正例：対応する画像・文
* 負例：対応しない組

### 4.2 なぜ重要か

* VLMの事前学習の基礎
* Zero-shot性能の源泉
* 多くのVLM（BLIP, LLaVA, Flamingo）がCLIPを基盤にしている


## 5. 学習フェーズの分離設計

### 5.1 一般的な学習ステップ

1. **Vision Encoder 事前学習**（ImageNet / CLIP）
2. **Vision-Language Alignment**
3. **Instruction Tuning（マルチモーダル）**

### 5.2 なぜ分けるのか

* 同時学習は不安定
* LLMの言語能力を破壊しやすい
* 学習コストが爆発する


## 6. マルチモーダルInstruction Tuning

### 6.1 特有の難しさ

* 「画像を見て答える」という指示理解
* 画像を無視するハルシネーション
* 視覚根拠の欠如

### 6.2 重要なデータ形式

```json
{
  "image": "...",
  "instruction": "この画像について説明してください",
  "output": "..."
}
```

* Visual grounding を意識したアノテーションが重要


## 7. データ設計と品質管理（LLM以上に重要）

### 7.1 よくある問題

* キャプションが浅い
* 画像とテキストが弱く対応
* バイアス（文字入り画像など）

### 7.2 勉強すべきデータセット

* COCO Captions
* Visual Genome
* LAION
* LLaVA Instruction data


## 8. 評価指標とベンチマーク

### 8.1 定量評価

* VQA accuracy
* Image Captioning（BLEU / CIDEr）
* Retrieval Recall@K

### 8.2 定性評価

* 視覚的根拠があるか
* 画像にない情報を捏造していないか


## 9. 実装面での重要事項（実務的）

### 9.1 計算資源

* Vision Encoderがメモリを食う
* Mixed Precision必須
* Gradient Checkpointing

### 9.2 Fine-tuning戦略

* Vision Encoder 凍結
* Projection Layer + LLM LoRA更新（QLoRAと相性良）


## 10. LLM経験者が特に意識すべきポイント

LLM経験者ほど注意すべき点：

* 「言語だけで解けてしまう」学習崩壊
* 画像を無視しても loss が下がる問題
* 視覚情報の**強制利用設計**


## 学習ロードマップ（簡易）

1. ViT / CLIP の理論と実装
2. 画像特徴とテキスト特徴の分布差
3. Projection + 擬似トークン設計
4. マルチモーダルInstruction Tuning
5. VQA・Captioning評価


## まとめ（要点）

* **VLMの本質は「アライメント」**
* 画像特徴をどうLLMに食わせるかが核心
* データ品質と学習フェーズ分離が成否を分ける
* LLM知識は強力だが「言語バイアス」に注意


VLM（Vision-Language Model）を学習・研究・実装する際は、
**「理論 → 構成要素 → 学習戦略 → 実装・デバッグ」** の順で理解を積み上げるのが最も効率的です。
LLMを既に学習されている前提で、**最短距離で実務・研究に到達する学習順序**を示します。

---

## 1. 最初に学ぶべきこと（必須の土台）

### 1.1 VLMが解いている本質的問題

最初に理解すべき問いは以下です。

* 「画像」と「言語」は**どう対応付けられるのか**
* なぜ単純な結合ではうまくいかないのか
* なぜ“画像を無視する”学習崩壊が起きるのか

ここを理解しないと、以降の設計判断がすべてブラックボックスになります。

---

### 1.2 CLIPと対照学習（最優先）

VLMのほぼすべては **CLIP思想の上に構築**されています。

学ぶべき点：

* Contrastive Loss（InfoNCE）
* 画像・テキストの共通埋め込み空間
* Zero-shot 推論が成立する理由

**理由**

* LLaVA / BLIP / Flamingo / Kosmos すべての前提知識
* VLMの「事前学習とは何か」を理解できる

---

## 2. Vision Encoderの理解（次に重要）

### 2.1 ViTを重点的に学ぶ

以下は最低限理解が必要です。

* Patch Embedding の意味
* 空間構造とSelf-Attention
* CLS token とパッチ特徴の違い

**理由**

* VLMでは CLS token だけでは情報不足
* パッチ列をどう使うかが設計差分になる

---

### 2.2 CNNは「比較対象」として学ぶ

* CNNは空間局所性に強い
* ViTはグローバル関係に強い

→ なぜVLMではViTが主流なのかを理解する。

---

## 3. モダリティ融合（Fusion）の設計原理

### 3.1 3つの融合戦略

まず理論的に理解すべき分類：

| 方式              | 特徴       | 学ぶ理由 |
| --------------- | -------- | ---- |
| Late Fusion     | CLIP型    | 基礎   |
| Cross-Attention | Flamingo | 発展   |
| Token Injection | LLaVA    | 実装容易 |

### 3.2 なぜLLaVA型から学ぶべきか

* 構造が最も単純
* LLM知識がそのまま使える
* デバッグしやすい

---

## 4. Projection Layerとアライメント

### 4.1 Projectionの役割

* 分布の異なる特徴空間を接続
* LLMが「意味のあるトークン」として扱えるようにする

### 4.2 勉強すべき観点

* 線形 vs MLP
* Projectionが弱すぎる／強すぎる問題
* 勾配の流れ

**ここはVLMの中核理解ポイントです。**

---

## 5. 学習ステージ設計（非常に重要）

### 5.1 なぜ段階学習が必須か

* 同時学習は不安定
* 言語能力の破壊を招く

### 5.2 標準的な流れ

1. Vision Encoder事前学習（CLIP）
2. Vision-Language Alignment
3. Multimodal Instruction Tuning

**LLM学習経験者ほど「一気に学習したくなる」点に注意が必要です。**

---

## 6. データ設計（性能の8割を決める）

### 6.1 最初に学ぶべきデータ

* COCO Captions
* Visual Genome
* VQA v2

### 6.2 勉強すべき視点

* 画像とテキストの依存度
* 曖昧な質問の扱い
* 言語バイアスの除去

---

## 7. 評価と失敗事例分析

### 7.1 定量評価

* VQA accuracy
* Captioning metrics（CIDEr等）
* Retrieval Recall

### 7.2 定性評価（重要）

* 画像を見ているか
* 捏造していないか
* 根拠を説明できるか

---

## 8. 実装フェーズで学ぶべきこと

### 8.1 最小構成の実装

最初にやるべき実装：

* CLIP + Linear Projection + LLM
* Vision Encoder Freeze
* 小規模データ

### 8.2 デバッグ観点

* Image tokenにAttentionが向いているか
* 画像を変えると出力が変わるか
* Projection層の重み分布

---

## 9. LLM経験者が陥りやすい罠

* 言語だけでlossが下がる
* Image tokenがノイズ扱いされる
* Instructionが抽象的すぎる

→ **「画像がなければ答えられない設計」** を常に意識する。

---

## 10. 推奨学習順ロードマップ（実践的）

### Step 1（理論）

* CLIP論文
* ViT論文
* LLaVA論文

### Step 2（実装）

* CLIP推論コード
* LLaVA最小構成実装

### Step 3（発展）

* Flamingo / BLIP比較
* Projection改良
* マルチ画像・動画対応

---

## まとめ（重要事項）

* **最初はCLIP思想から**
* Vision EncoderはViT中心
* ProjectionとAlignmentが核心
* データ設計が成功可否を決める
* 構造は単純なものから入る

---

次に進む候補としては、

* **「最小構成VLMのPyTorch実装」**
* **「VLM学習が失敗する具体例と対処」**
* **「LLaVA vs Flamingo 徹底比較」**

などが適しています。
どの方向を深掘りしますか。

