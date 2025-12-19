以下では **LLaVA（Large Language-and-Vision Assistant）** の構成を、
「なぜその設計になっているのか」「LLM経験者が理解すべき勘所はどこか」を重視して、体系的かつ実装視点で詳細に解説します。

---

## 1. LLaVAとは何か（設計思想）

LLaVAは一言で言うと、

> **既存の強力なLLMに、最小限の構造追加で視覚能力を付与するVLM**

です。

### 設計原則

* LLM本体は極力変更しない
* Vision Encoderは既存モデルを流用
* 両者の橋渡しは **単純な線形Projection**
* 学習は段階的に行う

この「割り切り」が、再現性と拡張性の高さにつながっています。

---

## 2. 全体アーキテクチャ

### 2.1 構成要素

```
[ Image ]
   ↓
Vision Encoder（CLIP ViT-L/14）
   ↓
Projection Layer（Linear / MLP）
   ↓
Image Tokens（擬似トークン列）
   ↓
LLM（Vicuna / LLaMA）
   ↓
Text Output
```

### 2.2 重要なポイント

* **クロスアテンションは使わない**
* 画像は「トークン列」としてLLMに直接挿入
* LLMは「画像を含む文章」を処理しているだけ

---

## 3. Vision Encoder（CLIP ViT-L/14）

### 3.1 なぜCLIPか

* 画像とテキストの対応をすでに学習済み
* Zero-shot性能が高い
* 大規模データで安定

### 3.2 出力形式

* 入力画像 → Patch分割（例：14×14）
* 各パッチ → 1024次元（ViT-L）

例：

```
Image → (N_patches=256, dim=1024)
```

CLS tokenは使わず、**パッチ特徴をすべて使う**のがLLaVAの特徴です。

---

## 4. Projection Layer（最重要）

### 4.1 役割

Vision Encoderの出力を
**LLMの埋め込み空間に写像**するための層。

```
1024 (Vision) → 4096 (LLM hidden)
```

### 4.2 なぜ単純な線形層なのか

* 学習が安定
* 過学習しにくい
* LLMの言語能力を壊しにくい

実装例（概念）：

```python
proj = nn.Linear(vision_dim, llm_hidden_dim)
image_embeds = proj(image_features)
```

### 4.3 擬似トークン化

Projection後のベクトルは：

* LLMの **token embedding と同次元**
* LLM内部では「普通のトークン」として扱われる

---

## 5. LLMへの入力構造

### 5.1 入力フォーマット

```
<image>
[IMG_1][IMG_2]...[IMG_N]
</image>
User: この画像について説明してください
Assistant:
```

* `[IMG_i]` は Projection後の埋め込み
* 特殊トークン `<image>` は境界識別用

### 5.2 Self-Attentionの働き

* 画像トークン同士
* 画像 ↔ テキスト
* テキスト同士

すべて **同一のSelf-Attention** で処理される。

---

## 6. 学習ステージの詳細

### Stage 1：Vision-Language Alignment

**目的**

* 画像特徴を「言語的に意味のある空間」に押し込む

**設定**

* Vision Encoder：Freeze
* LLM：Freeze
* Projection Layer：Train

**データ**

* 画像キャプション（COCOなど）

**損失**

* 通常の言語モデル損失（Next Token Prediction）

---

### Stage 2：Multimodal Instruction Tuning

**目的**

* 「画像を見て指示に従う」能力の獲得

**設定**

* Vision Encoder：Freeze
* LLM：LoRA or Partial Fine-tune
* Projection Layer：Train

**データ**

* GPT-4で生成した画像付きInstructionデータ

---

## 7. なぜこの学習設計が有効か

### 7.1 LLM破壊を防ぐ

* LLM全体をいきなり更新しない
* 言語能力を保持したまま視覚を追加

### 7.2 勾配の流れが明確

* 視覚情報は必ずProjectionを通る
* 不要な複雑化を避けている

---

## 8. よくある失敗とLLaVAの回避策

### 8.1 画像無視問題

**原因**

* テキストだけで答えられてしまう

**対策**

* 画像に依存したInstruction
* キャプションよりVQA比重を増やす

---

### 8.2 視覚ハルシネーション

**原因**

* 画像と関係ない言語知識の暴走

**対策**

* Alignmentフェーズを十分に行う
* 視覚根拠を要求するInstruction

---

## 9. 実装上の要点（LLM経験者向け）

### 9.1 メモリ

* Image tokens数が多い（256前後）
* Context length圧迫に注意

### 9.2 学習テクニック

* QLoRAとの相性が非常に良い
* Projection LayerはFP32推奨

---

## 10. LLaVAの本質的理解（まとめ）

* **LLaVAは「クロスアテンションなしVLM」**
* 画像を「トークン列」に偽装してLLMに渡す
* 成否は Projection とデータ設計で決まる
* 構造は単純、だが設計思想は極めて洗練されている

---

## 次に学ぶと理解が深まるテーマ

* Projection設計の改良（MLP / Perceiver）
* Flamingoとの比較（Cross-Attention型）
* LLaVA-1.5 / LLaVA-NeXTの差分
* 最小構成LLaVAのPyTorch実装
