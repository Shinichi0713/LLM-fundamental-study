PyTorchでの開発中、特にGoogle Colabなどのリソース制限がある環境や、データの次元が複雑なモデルを扱う際によく遭遇するエラーを分類してリストアップします。

---

### 1. 形状・次元に関するエラー (Shape Errors)

最も頻繁に発生するエラーです。層と層の間でデータの形が合わない場合に起こります。

* **`RuntimeError: size mismatch, m1: [a x b], m2: [c x d]`**
* **原因**: 全結合層（`nn.Linear`）の入力サイズが、前の層の出力サイズと一致していない。
* **対策**: `m1`の列数`b`と`m2`の行数`c`を一致させる。直前に`Flatten`を入れているか確認。


* **`RuntimeError: Expected 4-dimensional input for 4-dimensional weight [64, 3, 3, 3]...`**
* **原因**: 畳み込み層（`nn.Conv2d`）に3次元（C, H, W）の画像1枚をそのまま渡している。
* **対策**: バッチ次元を追加して4次元（B, C, H, W）にする必要がある。`image.unsqueeze(0)` を使用。



---

### 2. デバイスに関するエラー (Device Errors)

CPUとGPU（CUDA）が混在したときに発生します。

* **`RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!`**
* **原因**: モデルはGPUにあるのに、入力データがCPUにある（またはその逆）。
* **対策**: `data = data.to(device)` や `model = model.to(device)` を徹底する。


* **`RuntimeError: CUDA out of memory (OOM)`**
* **原因**: GPUのメモリ（VRAM）が不足。大きな画像サイズやバッチサイズが原因。
* **対策**: バッチサイズを下げる。`with torch.no_grad():` を使って不要な勾配計算を消す。Colabの「ランタイムを解除」してメモリを解放する。



---

### 3. 勾配・計算グラフに関するエラー (Gradient Errors)

* **`RuntimeError: trying to backward a second time...`**
* **原因**: 1つの `forward` に対して `backward()` を2回実行しようとした。
* **対策**: 通常はループ内で `optimizer.zero_grad()` を忘れていないか確認。


* **`RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`**
* **原因**: 勾配が必要な計算（学習）なのに、計算グラフが途切れている。
* **対策**: `with torch.no_grad():` の中で `loss.backward()` を呼んでいないか、テンソルを途中で `.numpy()` に変換していないか確認。



---

### 4. 損失関数に関するエラー (Loss Function Errors)

* **`RuntimeError: 1D target tensor expected, multi-target not supported`**
* **原因**: `nn.CrossEntropyLoss` のターゲット（正解）に、One-hotベクトルを渡している。
* **対策**: PyTorchのCrossEntropyは、クラス番号のラベル（0, 1, 2...）を期待する。`[batch_size]` の形状に修正。


* **`ValueError: Target size (torch.Size([64])) must be the same as input size (torch.Size([64, 1]))`**
* **原因**: `nn.MSELoss` や `nn.BCELoss` で、予測と正解の次元が微妙に違う（次元が1つ多い/少ない）。
* **対策**: `outputs.squeeze()` や `labels.view_as(outputs)` で形状を揃える。



---

### 5. データロードに関するエラー (DataLoader Errors)

* **`BrokenPipeError` / `RuntimeError: DataLoader worker (pid XXX) is killed by signal: Killed**`
* **原因**: `num_workers` が大きすぎて、メモリ不足で子プロセスが死んだ。
* **対策**: `num_workers=0` に設定してみる（特にWindowsやColabのメモリ制限時）。



---

### デバッグのコツ：`print` よりも `shape`

エラーが出たら、エラー行の直前でテンソルの形状を表示するのが解決への近道です。

```python
print(f"Input shape: {x.shape}")
print(f"Target shape: {y.shape}")

```

