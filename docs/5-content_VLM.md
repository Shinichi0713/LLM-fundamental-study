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
