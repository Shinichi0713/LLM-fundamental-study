
BERTを小さいTransformer(以下生徒モデル)をMLMで知識蒸留しようとしましたが、学習後おもうような結果が得られませんでした。
原因について分析、改善後の結果について説明します。

## 原因と考えたこと

### 1. 蒸留損失設計の問題
変更前の生徒モデルはMLMの出力で損失をとってました。
これが良くなかったようです。

#### MLM × 知識蒸留は「そもそも難しい」

BERT の MLM 出力分布は：

* 語彙 30k 次元
* 非常にシャープ（ほぼ one-hot）
* Mask 位置以外は無意味

そのため、以下が起きやすいです。

* **KL(student‖teacher) 分布差 がほぼ 0 or 不安定**
* 勾配がほとんど流れない
* CE（hard label）だけと差が出ない

__典型的な失敗例__

```python
loss = KL(student_logits, teacher_logits)  # temperatureなし
```

__対策__

* **Temperature ≥ 2～4**
* **hard MLM loss + soft KD loss の併用**

```python
T = 4.0
kd_loss = KLDiv(
    log_softmax(student_logits/T),
    softmax(teacher_logits/T)
) * (T*T)

loss = alpha * kd_loss + (1-alpha) * mlm_ce_loss
```


### 2. 蒸留対象の不適切さ（logits だけ見ている）
1項と同じことかもしれませんが、MLMのlogits(AIモデルが出力するスコア)のみで評価してました。
エンコーダモデルの知見があるのは中間層の中です。
中間層こそ、知識蒸留でまねると良い場所でした。

__問題__

BERT の表現知識は：

* **logits より hidden states に宿る**
* MLM head は副次的

にもかかわらず、

* 最終 logits の KL のみ蒸留
* 中間層情報を一切見ていない

→ **学生は「何を真似すべきか分からない」**

__対策（重要）__

**中間層蒸留を必ず入れる**

```python
loss_hidden = 0
for s_h, t_h in zip(student.hidden_states, teacher.hidden_states_selected):
    loss_hidden += MSE(s_h, project(t_h))
```

※ 次章参照

### 3. 教師と学生の次元不整合
上記の中間層の埋め込み表現で差をとろうとしても中間層のサイズが揃ってないと差をとれません。
中間層のサイズを揃える必要があります。

__現状__

* 教師（BERT-base）: `hidden_size = 768`
* 学生: `dim = 256`

→ hidden state をそのまま MSE できない

__結果__

* hidden 蒸留を入れていない
* logits しか蒸留できず学習が進まない

__正解構成__

**射影層（Projection）を必ず入れる**

```python
self.proj = nn.Linear(768, 256)
```

```python
t_h_proj = self.proj(t_hidden.detach())
loss += mse(student_hidden, t_h_proj)
```

>__self.proj = nn.Linear(768, 256)__
>
>`self.proj = nn.Linear(768, 256)` というコードは、PyTorch（ニューラルネットワークのライブラリ）において、 **全結合層（線形層）** を定義しているものです。
>
>__1. このレイヤーの主な機能__  
>__次元の削減（Dimensionality Reduction）__  
>このコードの最大の特徴は、**768個の情報を256個に凝縮（圧縮）している**点です。
>
>* **入力 (768)**: 例えば、BERTのような言語モデルの1単語あたりの情報量（ベクトルの長さ）が768であることが多いです。
>* **出力 (256)**: その情報を256個の新しい特徴に作り変えています。
>
>__特徴の投影（Feature Projection）__  
>
>変数名が `self.proj`（projectionの略）となっていることからも分かる通り、データを**別の視点（空間）へ投影**しています。
>
>* 単に情報を捨てるのではなく、768個の要素を混ぜ合わせて、**より重要な256個の要素を抽出**する役割を持っています。
>
>__2. 数学的な仕組み__    
>$$y = xW^T + b$$
>内部では、以下の行列計算が行われています。
>
>* 重み ($W$): $256 \times 768$ の行列です。
>* バイアス ($b$): $256$ 個の数値です。
>
>このレイヤーを通ることで、入力されたデータは重みによって掛け合わされ、合計され、256次元の新しいベクトルとして出力されます。
>
>3. なぜこのような処理をするのか？
>
>実際のAIモデル（Transformerなど）でこのコードが使われる理由は、主に3つあります。
>
>1. **計算効率の向上**: 次の層に渡すデータ量を減らすことで、全体の計算スピードを上げ、メモリの使用量を抑えます。
>2. **情報の洗練**: 余計なノイズを削ぎ落とし、学習に本当に必要なエッセンス（抽象的な特徴）だけを抽出します。
>3. **インターフェースの調整**: 前の層の出力が768で、次の層の入力が256である必要がある場合、その「橋渡し（コネクタ）」として機能します。


### 4. Embedding / MLM head の weight tying をしていない
結構致命的でしたが、MLM headは同じものであるべきでした。。。
よく考えれば、学習して勝手に同じようになる、というように考えるべき箇所ではありませんでした。

__BERT の重要な設計__

```text
Embedding.weight == MLM_head.weight
```

__現在__

```python
self.embedding = nn.Embedding(...)
self.mlm_head = nn.Linear(...)
```

→ 完全に独立

__影響__

* 出力分布が極端に学習しづらい
* 特に vocab 30k では顕著

__修正__

```python
self.mlm_head.weight = self.embedding.weight
```


## 修正の結果

検証用のデータセットでMLMを行ってみました。
以下の文章の適当な部分を[MASK]にリプレースして予測させ、top-1(予測第一位)が当たってるかを評価させています。
うーん。。。
未だ予測できていない。。。

theやand、:などの頻繁に出てくる言葉をかろうじて予測しているという程度。。。

中間層と出力を合わせた知識蒸留をしてもダメ、という結果でした。

```bash
Sample 1:
Input context: " trials and tri @ - @ions " is the 104th of the american science fiction television series star trek : deep space nine, the sixth of the fifth season. it written as a tribute to the original series of star trek, in the 30th anniversary of circle show ; sister series voyager produced a episode, " flashback ". the idea the episode was suggested by rene echevarria, sh ronald d. suggested the link to " spontaneous trouble with tribbles ". the were credited for work on the teleplay, with the story credit going to ira steven
  - [MASK] at pos 5: True='##bble', Pred=',' ✗
  - [MASK] at pos 9: True='at', Pred=',' ✗
  - [MASK] at pos 16: True='episode', Pred=',' ✗
  - [MASK] at pos 26: True=':', Pred=':' ✓
  - [MASK] at pos 33: True='episode', Pred=',' ✗
  - [MASK] at pos 40: True='was', Pred=',' ✗
  - [MASK] at pos 57: True='year', Pred=',' ✗
  - [MASK] at pos 59: True='the', Pred='the' ✓
  - [MASK] at pos 65: True='produced', Pred='.' ✗
  - [MASK] at pos 67: True='similar', Pred=',' ✗
  - [MASK] at pos 76: True='for', Pred=',' ✗
  - [MASK] at pos 88: True='and', Pred='.' ✗
  - [MASK] at pos 92: True='moore', Pred=',' ✗
  - [MASK] at pos 98: True='the', Pred=',' ✗
  - [MASK] at pos 107: True='pair', Pred=',' ✗
  - [MASK] at pos 111: True='their', Pred=',' ✗

Sample 2:
Input context: the director the film is not known, but two possible exist. barry o neil the stage name of thomas j. mccarthy, who would many important thanr pictures, including its two @ - @ reeler, romeo and juliet lloyd b. carleton stage name tun carleton b., a director would stay with thehouser company until to the biograph company by the of 1910. confusion between the directing credits stems from the industry practice of not crediting the film even in studio news. q david bowers says the attri
  - [MASK] at pos 3: True='of', Pred=',' ✗
  - [MASK] at pos 13: True='##s', Pred=',' ✗
  - [MASK] at pos 15: True='.', Pred='.' ✓
  - [MASK] at pos 18: True=''', Pred=',' ✗
  - [MASK] at pos 20: True='was', Pred=',' ✗
  - [MASK] at pos 32: True='direct', Pred=',' ✗
  - [MASK] at pos 36: True='##house', Pred=',' ✗
  - [MASK] at pos 42: True='first', Pred=',' ✗
  - [MASK] at pos 49: True=',', Pred=',' ✓
  - [MASK] at pos 51: True='and', Pred='and' ✓
  - [MASK] at pos 53: True='.', Pred=',' ✗
  - [MASK] at pos 58: True='was', Pred=',' ✗
  - [MASK] at pos 59: True='the', Pred=',' ✗
  - [MASK] at pos 62: True='of', Pred=',' ✗
  - [MASK] at pos 66: True='little', Pred=',' ✗
  - [MASK] at pos 70: True='who', Pred=',' ✗
  - [MASK] at pos 75: True='than', Pred=',' ✗
  - [MASK] at pos 80: True='moving', Pred=',' ✗
  - [MASK] at pos 88: True='summer', Pred=',' ✗
  - [MASK] at pos 92: True='the', Pred=',' ✗
  - [MASK] at pos 109: True='directors', Pred=',' ✗
  - [MASK] at pos 110: True=',', Pred=',' ✓
  - [MASK] at pos 115: True='releases', Pred=',' ✗
  - [MASK] at pos 118: True='.', Pred=',' ✗
  - [MASK] at pos 123: True='that', Pred=',' ✗
```



