# `unfold_kv()` 関数

✅ **結論：`unfold_kv()` は Local Attention のために「各トークンの周辺ウィンドウの K/V をまとめて取り出す関数」**

### ✔ 各トークン *t* について

「左右の window サイズぶんの K/V をまとめて取り出す」ための関数です。

最終的に返すテンソルは：

```
(B, H, T, window_len, D)
```

となり、

**各時刻 t に対して、window_len 個の K/V が並んだテンソル** を返します。

つまり Local Attention の *「近くだけ見る（sparse）」* を実現するための前処理です。

---

# 🔍 **戻り値の形をもう一度確認**

```
(B, H, T, window_len, D)
```

意味：

* **B** … バッチ
* **H** … ヘッド数
* **T** … シーケンス長（query の位置）
* **window_len = 2*window + 1**
* **D** … head_dim（各 head の次元）

例：window=2 の場合

```
t=0 → [0,1,2]           （padding入る）
t=1 → [0,1,2,3]
t=2 → [0,1,2,3,4]
t=3 → [1,2,3,4,5]
...
```

---

# 🧠 **なぜ PyTorch の `unfold` を使うのか？**

普通に for 文で window の切り出しを行うと遅い。

PyTorch の `F.unfold()` は convolution と同じ高速 GPU カーネルを使って、

* 連続する位置の切り出し
* padding
* スライディング処理

を一気に行えるので高速。

---

# 🧩 **コードをステップごとに解説**

---

## **step 1: （B, H, T, D）→（B*H, D, 1, T）に reshaping**

```python
x_img = x.permute(0, 1, 3, 2).reshape(B * H, D, 1, T)
```

Unfold が画像テンソルに対して動作するため

```
(B, H, T, D)
↓ チャンネル＝D、高さ＝1、幅＝T の画像
(B*H, D, 1, T)
```

に変換します。

---

## **step 2: Convolution 的な「スライド＋切り出し」処理をする**

```python
x_unf = F.unfold(
    x_img, 
    kernel_size=(1, kernel_size),
    padding=(0, padding), 
    stride=(1, 1)
)
```

`kernel_size=(1, window_len)` の1D畳み込み窓で T 方向にスライドしつつ切り出す。

---

## **step 3: unfold が返す形を整える**

unfold の出力：

```
(B*H, D * window_len, T)
```

整形：

```python
x_unf = x_unf.view(B * H, D, kernel_size, T)
```

---

## **step 4: 元の (B, H) に戻す**

```python
x_unf = x_unf.permute(0, 3, 2, 1).reshape(B, H, T, kernel_size, D)
```

これで完成。

---

# ✨ **最終的なイメージ**

### 入力：

```
(B, H, T, D)
```

### 出力（各位置 t で window_len 個の K/V を取り出す）：

```
(B, H, T, window_len, D)
```

### これを使うと：

```
Attention(Q[t], K[t-window:t+window])
```

が一気にできるようになる。

---

# 🎉 まとめ

| 機能                     | 説明                               |
| ------------------------ | ---------------------------------- |
| スライド窓処理           | 各トークンの周辺 window を抽出     |
| GPU最適化                | unfold を使うため高速              |
| Local Attention の下準備 | (B,H,T,window_len,D) の K/V を作る |
