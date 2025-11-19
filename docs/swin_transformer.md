# SWIN TRANSFORMER
SoTAパフォーマンスをビジョンタスクで達成した・・・？

## 概要
Visionをパッチングにより分割して埋め込み表現化。→ポジショナルエンコーダを埋め込みされる。
その後は通常のTransformerに入力。

![alt text](image.png)

Vision Transformerは、自然言語処理で成功したTransformerを画像認識に適用した手法

オリジナルのVision Transformerは、画像サイズが大きくなるほど計算複雑度が2乗に増大するという問題がありました。

Swin Transformerは、この問題を解決するために、特徴マップを重なりのないウィンドウに分割し、各ウィンドウに対してself-attentionを計算するWindow-based Self-Attentionを採用しました。

結果：計算複雑度を線形に低減。処理を高速化

ウィンドウをずらしてアテンションを行うShifted Window-based Self-Attentionを組み込みむことで、隣接するウィンドウとの関連性を考慮した処理が可能になりました。

また、Swin Transformerは、画像の細部から全体までの特徴を取得するために、パッチを結合して階層的に処理する構造を採用しています。

## アーキテクチャ


![alt text](image-1.png)

1. Patch Partitionで、入力RGB画像を重ならないパッチに分割→パッチサイズは4×4なので、各パッチのチャンネルの次元は4×4×3=48です。
2. Linear Embeddingを適用して、チャンネルの次元数を、任意の次元数 $C$ に変換
3. Swin Transformer Blockでは、特徴マップをウィンドウに分割し、各ウィンドウで、Multi-head Self Attention(MSA)を適用

MSA(Multi-head Self Attention)に適用するウィンドウの配置により2種類存在する。
- Window-based MSA(W-MSA)
- Shifted W-MSA(SW-MSA)

Patch Mergingは、複数のパッチを結合して、パッチ数を減らし、階層的な構造を作成します。結果として、解像度を半分、チャンネルを2倍にします。

## W-MSAとSW-MSA

![alt text](image-2.png)


図の左側は、ウィンドウサイズ( $M$ )ごとに等間隔で分割する標準的な方法です。

右側はウィンドウサイズの半分 ( $M/2$ )
だけシフトした位置から、ウィンドウサイズごとに分割するシフトした方法です。

レイヤーごとに分割方式を交互に変えることでウィンドウ間で情報のやり取りが可能となる。

__出来るようになったこと__
ウィンドウを常に同じ方法で分割していると、そのウィンドウ内の情報しか得ることができません。

ウィンドウをシフトすると、シフトしたウィンドウは、1つ前のレイヤーで分割された隣接のウィンドウと相互作用することができます。

__ウィンドウサイズの不均一__

ここで、hとwは特徴マップの縦と横のサイズを表します。また、一部のウィンドウはM×Mよりも小さくなります。

![alt text](image-3.png)

解決方法:
小さくなったウィンドウに対してパディングを行う。
また、self-attentionを計算するときにはパディングされた部分にマスクをかけて計算しないようにする。

```python
class SwinTransformerBlock(nn.Module):
    ...
    def forward(self, x):
        ...
        # この時点で、xの形状は(B,H*W,C)=(バッチ数、特徴マップの高さ*横幅、チャンネル数)です
        # 残差結合用にxを保存します
        shortcut = x
        # nn.LayerNormを適用します
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # W-MSAまたはSW-MSAを実行します
        # C++/CUDAで書かれたウィンドウ分割/結合を行うメソッド呼び出しのコードは、省略します

        # SW-MSAの場合(self.shift_size>0)、循環シフトを適用します
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        # HxWの特徴マップを、window_sizeの大きさのウィンドウに分割します
        # ウィンドウの数はnWで表され、バッチサイズと同じ軸に配置されます
        # 分割後のx_windowsの形状は、(nW*B, window_size, window_size, C)となります
        x_windows = window_partition(shifted_x, self.window_size)
        # ウィンドウベースのマルチヘッドアテンションを実行します
        # self.attnはWindowAttentionのforwardメソッドを呼び出します
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        # nW個のウィンドウを結合して、HxWの特徴マップを作成します
        # 結合後のshifted_xの形状は、(B, H, W, C)となります
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        # SW-MSAの場合、逆循環シフトを適用します
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        # (S)W-MSAの後は、timm.models.layers.DropPathを使用して、ドロップアウトを適用します
        # その後、残差結合を行います
        x = shortcut + self.drop_path(x)
        # nn.LayerNormで正規化した後、多層パーセプトロンを適用します
        # その後、ドロップアウトを適用し、最後に残差結合します
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

# ウィンドウ分割
def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows  # num_windows*B, window_size, window_size, C

# ウィンドウ結合
def window_reverse(windows, window_size, H, W):
    # windows: num_windows*B, window_size, window_size, C
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x  # B, H, W, C

# ウィンドウベースのマルチヘッドアテンション
# Attention(Q, K, V) = softmax(Q * K^T / scale + bias) * V
# MultiHeadAttention(Q, K, V) = Concat(head_1, ..., head_n) * W_o
# ここで head_i = Attention(Q * W_i^Q, K * W_i^K, V * W_i^V) となります
class WindowAttention(nn.Module):
    ...
    def forward(self, x, mask=None):
        B_, N, C = x.shape
        # self.qkvは、入力次元Cから出力次元3Cに変換する全結合層です
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # 3, B_, nH, N, C'
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # スケール変換を行い、アテンションスコアを獲得します
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B_, nH, N, N

        # 相対位置バイアスを追加します (後述する)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        # SW-MSAでは、softmaxを適用する前に、マスクを追加します
        # 一方、W-MSAでは、マスクを利用せず、そのままsoftmaxを適用します
        if mask is not None:
            nW = mask.shape[0]  # nWはウィンドウの数を表す
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        # ドロップアウト層を適用します
        attn = self.attn_drop(attn)  # B_, nH, N, N

        # アテンションを計算します
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)  # B_, nH, N, C' -> B_, nH, C', N -> B_, N, C
        # 入力次元Cのベクトルを、同じく入力次元Cのベクトルに変換する全結合層を適用します
        x = self.proj(x)
        # ドロップアウト層を適用します
        x = self.proj_drop(x)
        return x
```

マスク
```python
# 循環シフトを適用してウィンドウ分割した領域は、ウィンドウ領域とサブウィンドウ領域に分けられます。
# 以下のimg_maskには、各領域を識別するための領域番号が割り当てられています
# 例えば、H, W, window_size, shift_size = 4, 4, 2, 1の場合、img_mask[0, :, :, 0]は次のようになります
#     tensor([[0., 0., 1., 2.],
#             [0., 0., 1., 2.],
#             [3., 3., 4., 5.],
#             [6., 6., 7., 8.]])
# アテンションはウィンドウ単位で行われるため、例えば、右上のウィンドウ(1と2)では、
# サブウィンドウ領域1は、サブウィンドウ領域2の情報を使用せずにアテンションを計算する必要があります
# 以降では、この右上のウィンドウ領域に注目して解説します
H, W = self.input_resolution
img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
h_slices = (slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None))
w_slices = (slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None))
cnt = 0
for h in h_slices:
    for w in w_slices:
        img_mask[:, h, w, :] = cnt
        cnt += 1

# img_maskをウィンドウごとに分割し、ウィンドウ数、縦のウィンドウサイズ、
# 横のウィンドウサイズ、チャンネル数(=1)の順に並んだ4次元配列mask_windowsを作成します
# 先程の例の場合、mask_windows[1]は右上のウィンドウを表します
# 具体的には、次のように領域番号が格納されます
#     mask_windows[1, :, :, 0] = tensor([[1., 2.],
#                                        [1., 2.]])
mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
# 縦横のウィンドウの次元を1つに統合します
# チャンネル数は1なので、その次元は削除できます
# 結果として、mask_windows[1] = tensor([1., 2., 1., 2.])となります
mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
# 上記の分割処理(window_partition)と次元変換(view)は、(S)W-MSAのコードでも同様の処理が行われています
# 具体的には、以下の部分です
#     x_windows = window_partition(shifted_x, self.window_size)
#     x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
# このため、x_windowsに対応する領域番号は、mask_windowsが格納しています
# この後、アテンションが実行されます
# まず、全結合層と次元変換を行います
#     qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#     q, k, v = qkv[0], qkv[1], qkv[2]
# この処理はチャンネル方向に適用されるため、空間方向に関係するmask_windowsとは無関係です
# したがって、mask_windowsの変換は必要ありません
# 次は、qとkの転置による内積計算です
#     attn = (q @ k.transpose(-2, -1))
# qとkの形状は(B_, nH, N, C')=(バッチ数*ウィンドウ数, ヘッド数, ウィンドウサイズ*ウィンドウサイズ, チャンネル数)です
# N方向が空間位置を表すので、内積計算によってmask_windowsも変換する必要があります
# 先程の例で考えると、つまり、バッチ数=1、N=4、mask_windows[1]=tensor([1., 2., 1., 2.])の場合、
# 1番目のウィンドウのq[1,0]=[q1,q2,q3,q4]とk[1,0]=[k1,k2,k3,k4]の内積は次のようになります
#         k1       k2       k3       k4
#        [<q1,k1>  <q1,k2>  <q1,k3>  <q1,k4>]  q1
# attn = [<q2,k1>  <q2,k2>  <q2,k3>  <q2,k4>]  q2
#        [<q3,k1>  <q3,k2>  <q3,k3>  <q3,k4>]  q3
#        [<q4,k1>  <q4,k2>  <q4,k3>  <q4,k4>]  q4
# これと同様な操作を、mask_windowsにも適用します
# つまり、mask_windows[1]の1,2,1,2を縦と横に並べて、クロスさせたときに、各要素がどの領域に当たるかを考えます
# 例えば、左上の要素(attn[0,0])は、領域1と領域1の内積計算になるため、領域1の情報を表します(下図左側参照)
# しかし、その右隣の要素(attn[0,1])は、領域1の情報と領域2の情報を用いて内積計算をしています
# アテンションの計算では、異なる領域の情報を使用してはいけないため、attn[0,1]は計算対象から除外する必要があります
# このように、attnにはマスクする位置とマスクしない位置があります
# マスクしない位置を「o」で、マスクする位置を「x」で表すと、下図右側のようになります
#     1  2  1  2             1 2 1 2
#     |     |  |           1 o x o x
# 1 --1     |  |    --->   2 x o x o
# 2 --------x  |           1 o x o x
# 1            |           2 x o x o
# 2 -----------2
# このマスク判別をコードで表すと、以下のようになります
# attn_maskの要素が0ならマスクしないことを、それ以外の値ならマスクすることを表します
# (mask_windowsは領域番号を表していましたが、attn_maskはマスクの有無を表します)
attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
# マスクは、softmax(attn + attn_mask)の計算で使用されます
# この値をnew_attnとします
# マスクされた位置のnew_attnの値は、理想的には0にすることです
# これは、new_attnとvの内積を計算する際に、マスクに対応するvの要素を内積計算から除外できるからです
# マスクされた位置のnew_attnの値を0にするには、マスクされた位置のattn_maskを大きな負の値にすることです
# これは、softmax関数において、大きな負の値をexp関数に入力すると、その出力がほぼ0になるからです
# 以下のコードでは、マスクされた位置のattn_maskの値を-100、マスクしない位置のattn_maskを0に設定しています
attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
```

## Relative Position Bias

Swin TransformerのRelative Position Bias（相対位置バイアス）は、Transformerアーキテクチャにおいて「位置情報」をより効果的に扱うための工夫です。以下、その考え方をわかりやすく説明します。

---

### 背景

Transformerは本来、入力系列の順序や空間的な位置関係を直接考慮しません。そのため、従来は**絶対位置エンコーディング**（絶対的な位置情報を埋め込む）が使われてきました。しかし、画像や自然言語のようなデータでは「ある要素が他の要素からどれだけ離れているか」（**相対的な位置関係**）が重要になることが多いです。

---

### Relative Position Biasの考え方

Swin Transformerでは、**Self-Attention**計算時に「クエリ」と「キー」のペアごとに、その2つの位置の**相対的な距離**に応じてバイアス（補正値）を加えます。

#### 具体的には

- 入力画像をウィンドウ（小領域）に分割し、各ウィンドウ内でSelf-Attentionを行う。
- その際、各ペア（i, j）の相対位置（例えば、横に2ピクセル、縦に-1ピクセル離れている、など）ごとに、専用のバイアス値を持つテーブル（行列）を用意する。
- Attentionスコアにこのバイアス値を加えることで、「近い位置同士はより強く、遠い位置は弱く」など、空間的な関係性を学習できる。

#### なぜ良いのか？

- **局所性の強調**：画像では近くのピクセル同士が強く関係しやすい。相対位置バイアスでこれを表現できる。
- **パラメータ効率**：絶対位置エンコーディングよりもパラメータ数が抑えられ、ウィンドウサイズに依存したシンプルなテーブルで済む。
- **転移性**：画像サイズやウィンドウサイズが変わっても、相対的な距離に基づくため、柔軟に対応できる。

---

### 数式イメージ

Attentionのスコア計算時に、  
$$
\text{Attention}(Q, K) = QK^T / \sqrt{d} + \text{RelativePositionBias}
$$  
のように、**RelativePositionBias**を加えるイメージです。

---

### まとめ

- Swin TransformerのRelative Position Biasは、「要素間の相対距離」を考慮することで、画像などの空間的な構造をより自然に扱うための仕組みです。
- これにより、Self-Attentionが局所的な関係性を捉えやすくなります。

## メモリ増加を抑制
Self-Attentionの計算範囲をウィンドウ（小領域）ごとに限定していることでメモリを節約してる。
Swin Transformerがメモリの増加を抑制できる理由は、**Self-Attentionの計算範囲をウィンドウ（小領域）ごとに限定している**からです。


### 一般的なTransformerのSelf-Attention

従来のVision Transformer（ViT）などでは、Self-Attentionは**画像全体のすべてのパッチ同士**で行われます。  
- 画像サイズが大きくなると、パッチ数（N）が増えます。
- Self-Attentionの計算量・メモリ消費量は \(O(N^2)\) で増加します。



### Swin Transformerの工夫

Swin Transformerでは、**画像を固定サイズのウィンドウ（例：7×7ピクセル）に分割**し、  
- **各ウィンドウ内だけでSelf-Attention**を計算します（ウィンドウ内のパッチ数をMとすると、計算量・メモリ消費量は \(O(M^2)\)）。
- 画像サイズが大きくなっても、1つのウィンドウ内の計算量は変わらず、**全体の計算量・メモリ消費量は \(O(N \cdot M)\) となり、線形に抑えられます**。

#### さらに
- ウィンドウを少しずつずらして（Shifted Window）、ウィンドウ間の情報も伝播できるようにしていますが、メモリ消費の急増にはつながりません。


### まとめ

- 従来: 画像全体でSelf-Attention → メモリ消費 \(O(N^2)\)（N=パッチ数）
- Swin: ウィンドウごとにSelf-Attention → メモリ消費 \(O(N \cdot M)\)（M=ウィンドウ内パッチ数、固定）

この**ウィンドウ単位のSelf-Attention**によって、Swin Transformerは大規模画像でも効率的にメモリ消費を抑制できるのです。





