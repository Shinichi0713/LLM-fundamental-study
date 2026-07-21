
ニューラルネットワークの演算は[テンソル演算](https://yoshishinnze.hatenablog.com/entry/2026/02/28/175306)により計算されます。
その計算量は非常に膨大で、計算をいかに高速にするかでニューラルネットワークをリアルタイムに使えるか否かが決定することになると思います。

本日はそんなテンソル演算の高速化について説明していきます。

## テンソル演算とベクトル演算

テンソル演算はベクトル演算によって構成される演算です。
まず、関係性について扱っていきます。

### 1. 数学的な定義の違い

__ベクトル（vector）__
- 1次元の配列として表される量です。
- 例：3次元空間の位置ベクトル  
  $ \mathbf{v} = (v_1, v_2, v_3) $
- 数学的には「階数1のテンソル」と見なせます。

__テンソル（tensor）__
- 複数のインデックスを持つ量で、**階数（rank）** で次元を表します。
  - スカラー：階数0（0次元テンソル）
  - ベクトル：階数1（1次元テンソル）
  - 行列：階数2（2次元テンソル）
  - 3次元以上の配列：階数3以上
- 例：
  - 行列（階数2テンソル）：$ A_{ij} $
  - 3階テンソル：$ T_{ijk} $

つまり、**ベクトルは「階数1のテンソル」に含まれる**ため、  
「ベクトル演算 ⊂ テンソル演算」という包含関係になります。

### 2. 計算機科学・GPU・機械学習での「テンソル」

GPUやディープラーニングフレームワーク（PyTorch, TensorFlowなど）では、**多次元配列（multi-dimensional array）のことを「テンソル」と呼ぶ**ことが多いです。

この文脈では：

- **ベクトル**：1次元配列（例：`[1, 2, 3]`）
- **行列**：2次元配列（例：`[[1,2],[3,4]]`）
- **テンソル**：3次元以上の配列（例：`(batch, channel, height, width)`）

ここでも、ベクトルは「1階テンソル」、行列は「2階テンソル」として扱われます。

### 3. 「テンソル演算」と「ベクトル演算」の関係

__共通点__
- どちらも**配列に対する要素ごとの演算**（加算、スカラー倍など）を指すことが多いです。
- GPUでは、どちらもSIMT（スレッド並列）で効率的に処理できます。

__違い__
- **ベクトル演算**：主に1次元配列に対する演算（例：`y[i] = a*x[i] + b`）
- **テンソル演算**：多次元配列に対する演算全般を指し、ベクトル演算も含みます。
  - 例：行列積、テンソル縮約（einsum）、畳み込み、バッチ処理など



## GPUフレンドリーなベクトル演算

GPUフレンドリーなベクトル演算は、**「GPUのアーキテクチャ特性を理解し、その強みを最大限引き出すようにアルゴリズム・データ構造・メモリアクセスを設計すること」** で実現されます。  
以下、主な観点ごとに整理します。

### 1. メモリアクセスの局所性と連続性（Coalesced Access）

GPUは「ワープ／ウェーブ」単位で多数のスレッドが同じ命令を実行するSIMT型のアーキテクチャです。  
このとき、**隣接スレッドが連続したメモリアドレスにアクセスする**と、DRAMへのアクセスがまとまり、帯域を効率よく使えます。

__実現方法の例__
- **配列の要素を「スレッドインデックス順」に並べる**
  - 例：`a[threadIdx.x + blockIdx.x * blockDim.x]` のように、連続したインデックスでアクセスする。
- **AoS（Array of Structs）ではなく SoA（Struct of Arrays）を優先**
  - AoS: `struct { float x, y, z; } points[N];`
  - SoA: `struct { float x[N], y[N], z[N]; }`
  - SoAにすると、同じフィールド（x, y, z）が連続領域に並び、ベクトルロードやキャッシュ利用が効率化されます。

### 2. ベクトル化を阻害する分岐の回避（Branch Divergence 最小化）

SIMTでは、ワープ内のスレッドが異なる分岐を取ると、**分岐した両パスを順次実行**するため、性能が低下します。

__実現方法の例__
- **条件分岐を「マスク＋選択」に置き換える**
  - 例：`if (x > 0) y = a; else y = b;`  
    → `mask = (x > 0); y = mask * a + (1 - mask) * b;`
- **データを事前にソートして分岐を減らす**
  - 例：正負で処理が変わる場合、事前に符号でソートしておくと、ワープ内で同じ分岐を取りやすくなります。

### 3. メモリ階層の活用（共有メモリ・レジスタ・キャッシュ）

GPUには、グローバルメモリ（DRAM）、L2/L1キャッシュ、共有メモリ（shared memory）、レジスタ（register）といった階層があります。  
**データ再利用が多い場合、共有メモリやレジスタに載せてから演算する**と、グローバルメモリ帯域のボトルネックを緩和できます。

__実現方法の例__
- **行列積（GEMM）のタイル化**
  - グローバルメモリからタイル（小行列）を共有メモリにロードし、そのタイル内で多くの演算を行う。
- **リダクション（総和・最大値など）の段階的集約**
  - 各スレッドブロック内で共有メモリを使って部分和を取り、最後にグローバルで統合する。

### 4. 演算強度（Arithmetic Intensity）の向上

「演算強度」＝（演算回数）÷（メモリアクセス量）  
GPUは演算能力が高い一方、メモリ帯域は相対的に有限です。  
**1回のメモリアクセスでより多くの演算を行う**と、GPUの演算ユニットを飽和させやすくなります。

__実現方法の例__
- **ループ展開（loop unrolling）**
  - 1回のロードで複数回の演算を行うようにする。
- **演算の融合（kernel fusion）**
  - 例：`y = a*x + b` と `z = exp(y)` を別カーネルにせず、1つのカーネルにまとめることで、中間結果をメモリに書き戻さずに済む。

### 5. ベクトル型・組み込み関数の活用（float4, half2 など）

多くのGPUでは、32ビット幅のメモリインターフェースに対して、**より広いベクトル型（float4, int4など）でアクセスする**と、帯域利用効率が上がります。

__実現方法の例__
- CUDA の `float4` 型や、ROCm/HIP の `float4` 等を使い、1スレッドが4要素をまとめてロード・ストアする。
- ただし、スレッドごとに独立した4要素を扱えるようにデータ配置を設計する必要があります。

### 6. ライブラリ・フレームワークの活用（cuBLAS, oneMKL, Thrust など）

「自前で最適化する」よりも、**GPUベンダーが提供する最適化済みライブラリ**を使うのが最も確実です。

__代表例__
- NVIDIA: cuBLAS（BLAS）、cuFFT（FFT）、Thrust（アルゴリズム集）
- AMD: rocBLAS、rocFFT
- Intel: oneMKL（GPU対応あり）

これらは、ハードウェアごとにチューニングされており、多くの場合、手書きカーネルより高速です。

### 7. 実装レベルでの具体的な工夫（簡易例）

例：ベクトル加算 `c[i] = a[i] + b[i]` をGPUフレンドリーに

- メモリレイアウト：SoAまたは連続配列
- アクセスパターン：スレッドID順に連続アクセス
- 分岐：基本的に不要（単純な加算のみ）
- 演算強度：1ロード＋1ストアに対して1演算と低めだが、メモリ帯域をほぼ上限まで使える

```cuda
__global__ void vec_add(float *c, const float *a, const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
```

このように、**「連続アクセス＋スレッド並列性の最大化」** が基本となります。

## 例題

実際に計算速度の違いを確かめてみましょう。
以下にGoogle ColabでGPUを使い、「GPUフレンドリーな実装」と「非効率な実装」の速度差を体感できる例題を用意します。  
Colab上ではCUDA C/C++のコンパイルが面倒なため、**PyTorch（GPUテンソル）** を使ったPythonコードで比較します。

### 例題：ベクトル加算の速度比較（Colab + PyTorch）

__環境準備（Colab上で実行）__

まずはGoogle ColabでGPUが実行可能であることを確認してください。

![1784602889406](image/vector_firendly/1784602889406.png)

念のため以下を実行して`True`と表示されるか確認しておきましょう。

```python
import torch
import time

# GPUが使えるか確認
print("CUDA available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
```


__問題設定__
- 配列サイズ：`N = 10**7`（1000万要素）
- データ型：`float32`
- 演算：`c[i] = a[i] + b[i]`
- 比較する2パターン：
  1. **GPUフレンドリー**：SoA（別配列）＋連続アクセス
  2. **非効率**：AoS（構造体配列）＋ランダムアクセス

__1. GPUフレンドリーな実装（SoA＋連続アクセス）__

この処理が連続アクセスとなるのは、PyTorchテンソルのメモリレイアウトが「C-contiguous（行優先）」であり、要素がメモリ上に連続して並ぶためです。
PyTorchのデフォルトのテンソルは「C-contiguous」です。1次元配列（ベクトル）の場合、要素はメモリ上に次のように並びます。



```python
N = 10**7

# SoA: a, b, c を別々の配列としてGPU上に確保
a_good = torch.randn(N, dtype=torch.float32, device=device)
b_good = torch.randn(N, dtype=torch.float32, device=device)
c_good = torch.empty(N, dtype=torch.float32, device=device)

# 計測
torch.cuda.synchronize()
start = time.time()
c_good = a_good + b_good  # 連続アクセス＋SoA
torch.cuda.synchronize()
elapsed_good = time.time() - start

print(f"[Good] SoA + contiguous: {elapsed_good:.4f} s")
```

結果は以下でした。

```
[Good] SoA + contiguous: 0.0469 s
```

__2. 非効率な実装（AoS＋ランダムアクセス）__

以下はわざと配列をランダムアクセスにしました。

- indices は [100, 5, 73, ...] のようなランダム順です。
- GPUカーネル内では、スレッド0が points[100, 0]、スレッド1が points[5, 0]、スレッド2が points[73, 0]… とアクセスします。

→つまり、隣接スレッドがアクセスするメモリアドレスがバラバラになります。

更に効率を阻害する要因が以下のように計算した配列をバラバラにする計算です。
- points[indices, 0] のような整数配列によるインデックス（advanced indexing） は、PyTorch内部で次のような処理になります。
- indices に基づいて要素を集めて新しいテンソルを作る（gather 的な操作）
そのテンソルに対して演算を行い、結果を points[indices, 2] に書き戻す（scatter 的な操作）

この「集めて→演算→散らす」の過程で、
- 一時テンソルの確保・コピー
- メモリアクセスの不連続性

が発生し、純粋な連続アクセス演算よりオーバーヘッドが増えます。

```python
# AoS: 構造体風に x, y, z を持つ配列を1本で表現
# ここでは単純化のため、x=a, y=b, z=c を1つの2次元テンソルで表現
# shape: (N, 3) → points[i, 0]=a[i], points[i, 1]=b[i], points[i, 2]=c[i]
points = torch.randn(N, 3, dtype=torch.float32, device=device)

# ランダムなインデックス順にアクセスするためのインデックス配列
indices = torch.randperm(N, device=device)  # 0..N-1 をランダムに並べ替え

torch.cuda.synchronize()
start = time.time()

# 非効率なアクセス: ランダム順に points[i, 2] = points[i, 0] + points[i, 1] を実行
points[indices, 2] = points[indices, 0] + points[indices, 1]

torch.cuda.synchronize()
elapsed_bad = time.time() - start

print(f"[Bad]  AoS + random access: {elapsed_bad:.4f} s")
print(f"Speedup: {elapsed_bad / elapsed_good:.2f}x")
```

実行結果は以下の通りです。
1の例に比べて1.5倍程度速度が低下しました。
これは場合によっては更に速度差が広がるケースが存在します。

```
[Bad]  AoS + random access: 0.0699 s
Speedup: 1.49x
```

GPU上での「分岐の有無」による速度差を体感できる例題として、**条件付きのベクトル演算**を2通りの実装で比較する課題を提案します。

- 条件：`if x[i] > 0 then y[i] = a[i] else y[i] = b[i]`
- 比較する2パターン：
  1. **分岐あり**：`torch.where` やループ内if相当の処理
  2. **分岐なし**：マスク演算（`mask * a + (1 - mask) * b`）

Colab + PyTorch でそのまま動くコード例を示します。

### 例題：条件付き代入の速度比較（分岐あり vs 分岐なし）

__問題設定__
- 配列サイズ：`N = 10**7`（1000万要素）
- データ型：`float32`
- 条件：`x[i] > 0` なら `y[i] = a[i]`、そうでなければ `y[i] = b[i]`

__1. 分岐なし（マスク演算）の実装__

GPUフレンドリーな「分岐なし」実装です。  
条件をマスク（0 or 1）に変換し、要素ごとの演算だけで表現します。

>__マスクがifの代用になる理由__  
>マスク演算が if の代用になるのは、**条件を 0/1 に変換し、その重みで a と b を線形結合する**ことで、if の挙動を数式的に再現しているからです。
>- `mask = (x>0).float()` → `x[i]>0` なら 1、そうでなければ 0
>- `y = mask*a + (1-mask)*b`  
>  - `mask=1` → `y=a`  
>  - `mask=0` → `y=b`
>GPUでは、if を使うとワープ内で分岐発散が起きやすい一方、マスク演算はすべてのスレッドが同じ命令を実行し続けるため、SIMT効率が高くなります。

```python
N = 10**7

x = torch.randn(N, dtype=torch.float32, device=device)
a = torch.randn(N, dtype=torch.float32, device=device)
b = torch.randn(N, dtype=torch.float32, device=device)
y_no_branch = torch.empty(N, dtype=torch.float32, device=device)

# マスクを作成（x > 0 なら 1, そうでなければ 0）
mask = (x > 0).float()

torch.cuda.synchronize()
start = time.time()

# 分岐なし：マスク演算で条件付き代入を実現
y_no_branch = mask * a + (1 - mask) * b

torch.cuda.synchronize()
elapsed_no_branch = time.time() - start

print(f"[No branch] Masked: {elapsed_no_branch:.4f} s")
```

結果は以下の通りです。

```
[No branch] Masked: 0.0284 s
```

__2. 分岐あり（torch.where）の実装__

PyTorchの `torch.where` は内部的に条件分岐を伴う操作です。  
GPU上では、条件がランダムな場合に「分岐発散（branch divergence）」が起こりやすく、非効率になりがちです。

```python
# 同じ入力を使う
x = torch.randn(N, dtype=torch.float32, device=device)
a = torch.randn(N, dtype=torch.float32, device=device)
b = torch.randn(N, dtype=torch.float32, device=device)
y_branch = torch.empty(N, dtype=torch.float32, device=device)

torch.cuda.synchronize()
start = time.time()

# 分岐あり：torch.where を使用
y_branch = torch.where(x > 0, a, b)

torch.cuda.synchronize()
elapsed_branch = time.time() - start

print(f"[Branch]   torch.where: {elapsed_branch:.4f} s")
print(f"Speedup: {elapsed_branch / elapsed_no_branch:.2f}x")
```

こちらの結果は以下です。
1.56倍のスピード低下。
同じ計算しているのに時間は多くかかってしまうということになります。

```
[Branch]   torch.where: 0.0443 s
Speedup: 1.56x
```

## 総括

テンソル演算を高速化するキーポイントは、主に次の4点です。

### 1. メモリアクセスを「連続・局所化」する
- 配列は**連続インデックスでアクセス**（`a[i], a[i+1], ...`）
- データ構造は**SoA（Struct of Arrays）**を優先  
  → 同じフィールドがメモリ上で連続し、GPUのcoalesced accessが効く

### 2. 分岐を減らし、「マスク演算」で置き換える
- if 分岐は**ワープ内で分岐発散**を起こしやすい
- 条件を `mask = (x>0).float()` のように0/1に変換し、  
  `y = mask*a + (1-mask)*b` で if を表現すると、すべてのスレッドが同じ命令を実行し続け、SIMT効率が上がる

### 3. 演算強度（演算回数÷メモリアクセス量）を高める
- 1回のメモリアクセスでより多くの演算を行う（ループ展開、カーネル融合）
- 共有メモリ・レジスタを使い、データ再利用を増やす（行列積のタイル化など）

### 4. 可能なら最適化済みライブラリを使う
- cuBLAS / rocBLAS / oneMKL など、GPUベンダーがチューニング済みのライブラリを使うのが最も確実

計算結果は変わらないのに、速度は倍やひどい時には10倍になってしまうということが出てきます。
メモリアクセスの連続性の担保やマスクの利用、演算強度の検討など、是非試してみてください。

![1784604466783](image/vector_firendly/1784604466783.png)

