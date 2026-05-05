Mamba（特に Mamba-1）の主要な技術要素は、selective SSM 以外では以下のようなものが挙げられます。

---

### 1. 因果1D畳み込み（Causal 1D Convolution）

- Mambaブロック内に**局所的な文脈モデリング**用の因果1D畳み込みが入っています。
- 入力シーケンスに対して、短い窓幅（例: kernel size 4）で畳み込みを行い、近傍トークン間の局所的な依存関係を先に捉えてからSSMに渡す構造です。
- 実装上は `causal-conv1d` パッケージを用いた効率的な実装になっています。[GitHub - state-spaces/mamba](https://github.com/state-spaces/mamba)

---

### 2. ハードウェアを意識した並列アルゴリズム（Hardware-aware parallel algorithm）

- selective SSM は**入力依存のパラメータ**を持つため、単純な畳み込みや再帰として実装するとメモリ・計算効率が悪くなります。
- Mambaでは、**kernel fusion・並列スキャン・再計算**などを組み合わせた「ハードウェアを意識した並列アルゴリズム」を設計し、GPU上で効率的に動くようにしています。[Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- これは FlashAttention に着想を得た「SSM版のハードウェア最適化」と位置づけられます。[GitHub - state-spaces/mamba](https://github.com/state-spaces/mamba)

---

### 3. ブロック拡張（Block expansion）と簡素化されたブロック構造

- Mambaブロックでは、入力次元を **expand 倍（典型的には 2 倍）**に拡張してからSSMを適用し、その後で元の次元に戻す構造になっています。
- これにより、チャネル方向の情報混合（TransformerのMLPに相当する役割）と、シーケンス方向の情報混合（Attention/SSMに相当）を**一つのブロックに統合**しています。
- 論文では「AttentionもMLPブロックも使わない、シンプルなエンドツーエンドのニューラルネットワーク」と表現されており、Transformerのような明示的なMLP層を持たない点が特徴です。[Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)

---

### 4. 混合精度訓練（Mixed Precision / AMP）による安定化

- SSMの再帰的なダイナミクスは数値的に不安定になりやすいため、Mamba実装では**PyTorch AMP（Automatic Mixed Precision）**を利用し、主要パラメータを float32 で保持しつつ、計算は低精度で行うことで安定性と速度を両立しています。[GitHub - state-spaces/mamba](https://github.com/state-spaces/mamba)

---

### 5. 時間依存（time-varying）な計算フレームワーク

- 従来のS4（Structured State Space Model）は**時間不変（time-invariant）**なパラメータ（A, B, C, Δ）を持っていました。
- Mambaでは、**B, C, Δ を入力に依存させて時間変化する**ようにし、これが「selective SSM」の核ですが、その結果として**SSM全体が時間依存のフレームワーク**として再設計されています。[Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- この時間依存性により、TransformerのAttentionに近い「入力に応じた情報の選択・忘却」が可能になっています。

---

### 6. Mamba-2 / Mamba-3 での追加要素（参考）

- **Mamba-2** では、SSMとAttentionの「双対性（duality）」に基づく**SSD（State Space Duality）**アルゴリズムが導入され、SSMヘッドをより効率的な行列積として計算できるようになっています。[State Space Duality (Mamba-2) Part I - The Model | Tri Dao](https://tridao.me/blog/2024/mamba2-part1-model/)
- **Mamba-3** では、MIMO（Multiple Input Multiple Output）モードや、`is_outproj_norm` による出力投影後の正規化、`chunk_size` によるチャンク処理など、さらなる効率化・安定化の仕組みが追加されています。[GitHub - state-spaces/mamba](https://github.com/state-spaces/mamba)

---

### まとめ

Mambaの「selective SSM 以外の主要な技術要素」としては、

- **因果1D畳み込み**（局所文脈のモデリング）
- **ハードウェアを意識した並列アルゴリズム**（kernel fusion・並列スキャン・再計算）
- **ブロック拡張と簡素化されたブロック構造**（expand によるチャネル混合とシーケンス混合の統合）
- **混合精度訓練**（SSMの再帰ダイナミクスの安定化）
- **時間依存なSSMフレームワーク**（入力依存パラメータによる選択的処理）

が特に重要です。[Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)[GitHub - state-spaces/mamba](https://github.com/state-spaces/mamba)[Mamba (deep learning architecture) - Wikipedia](https://en.wikipedia.org/wiki/Mamba_(deep_learning_architecture))


前回挙げた要素のうち、**精度（予測性能）に直接影響する**のは主に次の3つです。

---

### 1. 時間依存（time-varying）なSSMフレームワーク（＝selective SSMそのもの）

- Mambaの最大の特徴は、SSMのパラメータ（B, C, Δ）を**入力に依存させて時間変化させる**ことです。
- これにより、TransformerのAttentionのように「どの情報を保持し、どれを忘れるか」を**入力ごとに動的に選択**できるようになります。
- この「選択性」が、言語・音声・ゲノミクスなどでTransformer級の精度を達成する**最も直接的な要因**です。[Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)

---

### 2. 因果1D畳み込み（Causal 1D Convolution）

- 畳み込みは**局所的な文脈モデリング**を担い、近傍トークン間の依存関係を事前に捉えます。
- これにより、SSMに渡る前の表現が豊かになり、**長距離依存だけでなく短距離のパターンも効率的に学習**できるようになります。
- 実験的にも、この局所畳み込みを入れることで言語モデリング精度が向上することが報告されています。[GitHub - state-spaces/mamba](https://github.com/state-spaces/mamba)

---

### 3. ブロック拡張（Block expansion）と簡素化されたブロック構造

- expand による次元拡張は、**チャネル方向の情報混合（TransformerのMLPに相当）**と**シーケンス方向の混合（SSM）**を一つのブロックに統合する役割があります。
- これにより、表現力が向上し、Transformerのような明示的なMLP層がなくても十分な非線形変換能力を確保できます。
- 結果として、**モデルの表現力そのもの**に直接影響し、精度向上に寄与します。[Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)

---

### 補足：精度に「間接的」に効く要素

- **ハードウェアを意識した並列アルゴリズム**や**混合精度訓練**は、主に**計算効率・安定性**のための要素です。
  - これらがなければ、大規模モデルを現実的な時間で学習できなかったり、数値不安定で学習が破綻したりするため、**結果として精度に影響**しますが、表現力そのものを増やすわけではありません。
- したがって、「精度に直接影響」という意味では、上記3つ（時間依存SSM、因果畳み込み、ブロック拡張）が中心的な役割を果たします。[Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)[GitHub - state-spaces/mamba](https://github.com/state-spaces/mamba)
