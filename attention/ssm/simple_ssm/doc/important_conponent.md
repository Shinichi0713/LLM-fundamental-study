Mamba-2 の技術的な要素について、**論文「Mamba-2: Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality」**[arXiv](https://arxiv.org/abs/2405.21060) に基づき、要点を整理して説明します。

---

## 1. 状態空間モデル（SSM）の基本

Mamba-2 は、**状態空間モデル（State Space Model, SSM）**を中核に据えたモデルです。

### 連続時間の SSM
- 入力 \( x(t) \)、状態 \( h(t) \)、出力 \( y(t) \) に対して、
  \[
  \begin{aligned}
  h'(t) &= A h(t) + B x(t) \\
  y(t)  &= C h(t) + D x(t)
  \end{aligned}
  \]
- \( A, B, C, D \) はパラメータ行列です。

### 離散化（ゼロ次ホールド）
- サンプリング間隔 \( \Delta \) を用いて離散化：
  \[
  \begin{aligned}
  \bar{A} &= \exp(A \Delta) \\
  \bar{B} &= (\exp(A \Delta) - I) A^{-1} B \\
  h_t &= \bar{A} h_{t-1} + \bar{B} x_t \\
  y_t  &= C h_t + D x_t
  \end{aligned}
  \]
- Mamba-2 では、**入力に依存した \( \Delta_t \)（時間ステップ）** を導入し、**選択的（selective）SSM** として機能させます。

---

## 2. 選択的状態空間モデル（Selective SSM）

Mamba-1 から引き継がれた重要な要素が、**選択的 SSM** です。

### 入力依存のパラメータ
- \( B_t, C_t, \Delta_t \) を **入力 \( x_t \) から生成**します。
  - これにより、**重要な情報を状態に強く保持し、不要な情報を忘れる**ことができます。
- 例：言語モデルでは、**文脈上重要なトークン**を状態に長く保持し、**フィラー語**は早く忘れる、といった挙動が可能になります。

### 選択性の利点
- **長距離依存の扱いが柔軟**になり、Transformer の注意機構に近い表現力を持ちます。
- 一方で、**計算量は依然として線形 O(L)** のままです。

---

## 3. 状態空間双対性（State Space Duality, SSD）

Mamba-2 の最大の特徴は、**状態空間モデルと注意機構の双対性（SSD）**を利用したアルゴリズムです。

### 双対性の概要
- 特定の構造を持つ SSM（スカラー対角 \( A \)）と、**特定の形式の注意機構**が、**数学的に等価**であることを示しました。
- これにより、
  - SSM を **注意機構のように並列計算**できる
  - 注意機構を **SSM のように線形時間で計算**できる
  という両方の視点が得られます。

### SSD アルゴリズムの利点
- **訓練時の並列性**：SSM を注意機構風に並列計算できるため、**訓練速度が大幅に向上**します。
- **推論時の効率**：SSM としての線形時間計算を維持しつつ、**メモリ効率も改善**します。
- 実装では、**行列積（matmul）ベースの並列計算**で状態更新を行います。

---

## 4. アーキテクチャの設計要素

### Mamba-2 Block の構造
1. **LayerNorm**：入力の正規化
2. **in_proj**：入力から `[z0, x0, z, B, C, Δ]` を生成
   - `z0, x0`：MLP 的スキップ用
   - `z, B, C, Δ`：SSM 用
3. **causal_conv1d**：局所的な時間方向の混ぜ合わせ（因果性を保証）
4. **SSM（SSD）**：状態空間モデルによる長距離依存の処理
5. **out_proj**：SSM 出力と MLP スキップを統合
6. **残差接続＋ゲート**：安定した学習を実現

### グループ化された値注意（Grouped-Value Attention, GVA）
- Mamba-2 では、**複数の SSM ヘッドをグループ化**し、**値（value）を共有**する構造を導入しています。
- これにより、
  - **パラメータ効率**が向上
  - **テンソル並列性**が改善
  といった利点があります。

---

## 5. 実装上の最適化

### 高速な selective_state_update
- Triton や CUDA カーネルを用いた **高速な状態更新カーネル**を実装。
- SSD アルゴリズムにより、**行列積ベースの並列計算**が可能になり、**訓練速度が Mamba-1 比で 2–8 倍高速**と報告されています[HuggingFace Transformers Docs](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/mamba2.md)。

### メモリ効率の良いパス
- 長シーケンスではメモリ使用量を抑えるため、**メモリ効率パス（use_mem_eff_path=True）**を提供。
- 推論時は `step` 関数で状態を更新し、**O(1) の計算量**で次のトークンを生成できます。

### テンソル並列性の改善
- GVA 構造により、**テンソル並列（Tensor Parallelism）**が容易になり、大規模モデルの分散学習が可能です。

---

## 6. Mamba-2 の位置づけと利点

### Transformer との関係
- Mamba-2 は、**Transformer の代替モデル**として設計されています。
- 注意機構の二次計算量 O(L²) を避けつつ、**選択的 SSM と SSD により表現力を維持**しています。

### 主な利点
1. **線形計算量 O(L)**：長シーケンスでもスケーラブル
2. **選択的 SSM**：入力依存の状態遷移により、重要な情報を長く保持
3. **SSD アルゴリズム**：訓練時の並列計算により高速化
4. **メモリ効率**：推論時は状態を保持して O(1) 生成
5. **テンソル並列性**：大規模モデルへの拡張が容易

---

## 7. まとめ

Mamba-2 の技術的な要素を一言でまとめると、

> **「選択的状態空間モデル（Selective SSM）と状態空間双対性（SSD）を組み合わせ、Transformer の表現力を維持しつつ線形時間で計算できる新しいシーケンスモデル」**

です。

- **選択的 SSM**：入力依存の状態遷移で柔軟な長距離依存を実現
- **SSD**：SSM と注意機構の双対性を利用し、並列計算と線形時間計算を両立
- **GVA**：グループ化された値注意でパラメータ効率と並列性を向上
- **実装最適化**：Triton/CUDA カーネル、メモリ効率パス、step 関数などで実用性を高める

これらにより、Mamba-2 は **長文処理や大規模言語モデルにおいて、Transformer の有力な代替候補**として位置づけられています。

## 参考サイト

https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba2.py

https://github.com/Dao-AILab/causal-conv1d/blob/main/causal_conv1d/causal_conv1d_varlen.py

