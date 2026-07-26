import torch
import time

# GPUが使えるか確認
print("CUDA available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


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


"""

---

## 3. 期待される結果と学べること

### 結果のイメージ（Colab T4など）

- **No branch（マスク演算）**：すべてのスレッドが同じ命令を実行し続けるため、SIMTの効率が良い。
- **Branch（torch.where）**：条件がランダムな場合、ワープ内で分岐が発生し、両パスを順次実行するためオーバーヘッドが増える。

実際の環境では、**1.2〜3倍程度**の速度差が出ることが多いです（条件の偏りやGPUに依存）。

例（イメージ）：
```
[No branch] Masked: 0.0015 s
[Branch]   torch.where: 0.0030 s
Speedup: 2.00x
```

### 学べるポイント
1. **GPUでは「分岐発散」が性能を低下させる**ことがある。
2. **条件付き処理をマスク演算に置き換える**と、すべてのスレッドが同じ命令を実行し続け、SIMTの効率が上がる。
3. ただし、**条件がほぼ一定（例：ほぼ全要素が真）の場合**は、分岐のコストが小さくなり、速度差も縮まる。

---

## 4. 発展課題（余力があれば）

### (1) 条件の偏りを変えてみる

```python
# ほぼすべて x > 0 になるようにする
x_mostly_positive = torch.randn(N, device=device) + 10.0
y_branch2 = torch.where(x_mostly_positive > 0, a, b)
```
→ 条件がほぼ一定だと、分岐発散が減り、`torch.where` の速度がマスク演算に近づくことがあります。

### (2) より複雑な条件付き演算

```python
# 例：x>0 なら a*b、そうでなければ a-b
y_no_branch2 = mask * (a * b) + (1 - mask) * (a - b)
y_branch2     = torch.where(x > 0, a * b, a - b)
```
→ 演算が重くなると、分岐のコストが相対的に小さくなることもあります。

---

## まとめ

この例題では、**「同じ条件付き処理でも、分岐の有無でGPUの速度が変わる」**ことを実測で確認できます。  
特に条件がランダムな場合、マスク演算（分岐なし）の方がGPUフレンドリーであることが体感できるはずです。

実際にColabで実行していただくと、分岐発散の影響がより実感しやすくなります。
"""