昨日のMambaの精度が今一つだった点を受けて精度向上に取り組みます。

コードを見る限り、精度不足の主な要因は「**モデル構造がMambaの本質的なSSM処理をほとんど再現できていない**」ことと、「**データ・学習設定上の制約**」の両方が重なっていると考えられます。具体的には以下のような点が挙げられます。

## 問題点
今回精度が今一つだった点について確認してみます。

### 1. SimpleSSM の表現力が限定的

`SimpleSSM` の構造は以下の通りです。

```python
class SimpleSSM(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.linear2 = nn.Linear(d_model, d_model)

    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.linear2(x)
        return x + residual
```

このブロックの問題点：

- **局所的な畳み込みしか行わない**  
  - `kernel_size=3, padding=1` なので、各位置は「自分＋前後1トークン」の情報しか見ていません。
  - 長い文脈（例：段落全体の流れ）を捉えるには不十分です。
- **非線形変換が弱い**  
  - `Conv1d` の後はそのまま `linear2` に入り、活性化関数（SiLU, GELU など）がありません。
  - 表現力が線形寄りになり、複雑な言語パターンを学習しづらいです。
- **入力依存のパラメータがない**  
  - Mambaの特徴である「入力に応じて状態遷移を変える（selectivity）」がなく、**静的な畳み込みフィルタ**でしかありません。

→ 結果として、**「近い位置の単語を少し混ぜるだけの浅いモデル」**になっており、Transformer や本物のMambaのような深い文脈理解は期待しづらい構造です。

### 2. モデル容量と深さの不足

- `d_model=512, n_layers=6` という設定は、現代のLLM（数百～数千次元、数十～百層以上）と比べると**かなり小規模**です。
- 特に `wikitext-2` のような多様な語彙・文脈を持つコーパスに対しては、**表現力が足りず、学習が「頻出パターンの表面的な模倣」にとどまっている**可能性が高いです。
- 前回記事の生成結果でも、「Sun」が過剰に出てくるなど、**コーパスの偏りをそのまま反映しただけ**の出力になっていました。

### 3. 学習設定・データ側の要因

__(1) 学習エポック数が少ない__
- 記事では `num_epochs=3` とされています。
- 事前学習レベルの言語モデルでは、3エポックでは**損失が十分に下がりきっていない**ことが多く、特に小規模モデルでは「まだランダムに近い状態」から抜けきれていない可能性があります。

__(2) データ品質と前処理__
- `wikitext-2` はWikipedia風テキストですが、**見出し・箇条書き・表・短い文**などが混在しており、そのまま `max_length=128` で切り捨てると文脈が断片化しやすいです。
- また、`padding="max_length"` で一律128トークンに揃えているため、**短い文が多く、パディングトークンが大量に入る**可能性があります。
  - `CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)` は設定されていますが、それでも学習効率は下がります。

__(3) 生成時のサンプリング設定__
- `temperature=0.7` はやや高めで、ランダム性が強く出やすい設定です。
- モデルがまだ不安定な段階では、`temperature` を下げる（例：0.3～0.5）か、**top-k / top-p サンプリング**を導入した方が「それっぽい文」になりやすいです。

### 4. 改善の方向性（まとめ）

精度不足の主な原因は、

1. **SimpleSSM が「浅い局所畳み込み＋線形変換」に留まり、表現力が不足**  
   → 活性化関数の追加、カーネルサイズやチャネル数の拡大、あるいはSelf-Attentionや本格的なSSM層の導入。
2. **モデル容量・深さが小さく、データに対して表現力が足りない**  
   → `d_model` や `n_layers` の増加、もしくはより小さなデータセット（例：単一文書の繰り返し）で「まずは過学習できるか」を確認。
3. **学習エポック数が少なく、データ前処理・生成設定にも改善の余地がある**  
   → エポック数の増加、データのクリーニング・長文対応、生成時のサンプリング戦略の調整。

といった点に集約されます。  
特に「Mamba風」を目指すのであれば、**SSMとしての時間方向の状態更新（recurrent scan）と入力依存パラメータ（selectivity）を実装する**ことが、精度向上の鍵になると思われます。



```
Epoch 1, Step 0, Loss: 10.9802, Avg Loss: 10.9802
Epoch 1, Step 100, Loss: 6.1956, Avg Loss: 7.3794
Epoch 1, Step 200, Loss: 4.8420, Avg Loss: 6.4717
Epoch 1, Step 300, Loss: 4.5911, Avg Loss: 5.9488
Epoch 1, Step 400, Loss: 4.4673, Avg Loss: 5.5942
Epoch 1, Step 500, Loss: 4.9129, Avg Loss: 5.3118
Epoch 1, Step 600, Loss: 3.9961, Avg Loss: 5.0936
Epoch 1, Step 700, Loss: 3.8989, Avg Loss: 4.9057
Epoch 1, Step 800, Loss: 3.8737, Avg Loss: 4.7450
Epoch 1, Step 900, Loss: 3.2990, Avg Loss: 4.6016
Epoch 1, Step 1000, Loss: 3.1160, Avg Loss: 4.4724
Epoch 1, Step 1100, Loss: 2.8058, Avg Loss: 4.3510
Epoch 1, Step 1200, Loss: 2.7871, Avg Loss: 4.2405
Epoch 1, Step 1300, Loss: 3.0669, Avg Loss: 4.1339
Epoch 1, Step 1400, Loss: 2.2474, Avg Loss: 4.0327
Epoch 1, Step 1500, Loss: 2.7693, Avg Loss: 3.9388
Epoch 1, Step 1600, Loss: 3.0819, Avg Loss: 3.8528
Epoch 1, Step 1700, Loss: 2.1806, Avg Loss: 3.7720
Epoch 1, Step 1800, Loss: 1.4184, Avg Loss: 3.6913
Epoch 1, Step 1900, Loss: 1.5375, Avg Loss: 3.6141
Epoch 1, Step 2000, Loss: 2.0396, Avg Loss: 3.5412
Epoch 1, Step 2100, Loss: 2.2663, Avg Loss: 3.4714
Epoch 1, Step 2200, Loss: 2.0485, Avg Loss: 3.4049
Epoch 1, Step 2300, Loss: 1.9440, Avg Loss: 3.3399
Epoch 1, Step 2400, Loss: 1.3409, Avg Loss: 3.2792
Epoch 1, Step 2500, Loss: 1.8998, Avg Loss: 3.2199
Epoch 1, Step 2600, Loss: 1.7993, Avg Loss: 3.1650
```
