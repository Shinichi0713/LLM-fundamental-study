## MLP Adapter

VLMのおそらく一番メジャーなMLP AdapterでVLMを作ってみます。

## Trial Contents

新規モデル組んでみて、画像の説明タスクが出来るようになるかを検証する。

VLM（Vision-Language Model）が解決できるタスクは、画像とテキストの「相互理解」の深さによっていくつかのカテゴリーに分類されます。

現在のトレンドであるLlama-3-VisionやLLaVAのようなモデルが、具体的にどのような実務をこなせるのか、主要なタスクを整理しました。

## 解かせるタスク

### 1. 視覚的理解・記述（Vision-to-Language）
画像の内容を言語化する、最も基本的なタスク群です。

- **Image Captioning（画像キャプション生成）:** 画像の内容を1文〜数文で説明します。
- **Detailed Description（詳細記述）:** キャプションよりも詳細に、背景、物体の色、位置関係、雰囲気などを深く描写します。
- **Visual Question Answering (VQA)（視覚的応答）:** 「テーブルの上にリンゴは何個ありますか？」といった、画像に基づいた具体的な質問に回答します。



### 2. 推論・知識活用（Reasoning）
単に見えるものを答えるだけでなく、画像から「意味」を読み取る高度なタスクです。

- **Visual Reasoning（視覚的推論）:** 「この状況で次に何が起きそうですか？」など、物理法則や常識に基づいた推論を行います。
- **OCR & Document Understanding（文書理解）:** 画像内のテキストを読み取る（OCR）だけでなく、レシートの合計金額や請求書の期限などを構造的に理解します。
- **Mathematical / Scientific Reasoning（学術的推論）:** グラフや図解を読み取り、数学の問題を解いたり、実験結果を分析したりします。



### 3. 空間的認識（Grounding & Localization）
画像内の「どこ」に「何」があるかを座標（バウンディングボックス）で特定するタスクです。

- **Visual Grounding / Referring Expression:** 「左奥にある赤い車」という言葉に対し、その場所の座標を特定します。
- **Object Detection (via Language):** 従来の物体検出モデルのように、特定のクラス（犬、猫など）の場所を言語モデルのトークンとして出力します。

### 4. 特殊・応用タスク
実務やクリエイティブの現場で活用される応用範囲です。

- **Image-to-Code（コード生成）:** Webサイトのデザイン案（画像）を見て、それを再現するHTML/CSSやReactのコードを生成します。
- **Action Recognition（動作認識）:** 動画や連続した画像から「今、何をしているか（料理中、歩行中など）」を判断します。
- **Anomaly Detection（異常検知）:** 工場のライン写真などを見て、製品の傷や不具合（アノマリー）を指摘します。

### VLMタスクの比較まとめ

| カテゴリー | 難易度 | 特徴 |
| :--- | :--- | :--- |
| **記述 (Captioning)** | 低 | 見えるものをそのまま言葉にする |
| **応答 (VQA)** | 中 | 画像と質問の整合性を取る必要がある |
| **推論 (Reasoning)** | 高 | 背景知識や論理的な思考が求められる |
| **コード生成** | 最高 | 視覚構造をプログラミング言語へ変換する |

今回は使うモデルもそこまで高度なものは使わずMLP Adapterの性能が発覚するところまでを見届ける、ということにあるため、難易度が一番容易な **Captioning** を選定する。

## アーキテクチャ検討
今回はいつものGoogle Colabでの学習ができることとする。

結論から言うと、**Colab制約下でのVLM + MLP AdapterによるCaptioning検証** は、以下の設計にすると最も現実的かつ再現性が高いです：

### 全体アーキテクチャ

```
[Image]
   ↓
Vision Encoder（凍結）
   ↓  (B, N, Dv)
MLP Adapter（学習対象）
   ↓  (B, N, Dl)
Projection / Tokenization
   ↓
LLM（凍結 or LoRA）
   ↓
Text (Caption)
```

### 1. 各コンポーネントの設計指針

__① Vision Encoder（固定）__

* 推奨：

  * CLIP ViT-B/16 または ViT-L/14
* 理由：

  * 事前学習済みで特徴抽出が安定
  * Colabでもメモリ的に扱える

**出力**

```
(B, N, Dv)
例: (batch, 197 tokens, 768 dim)
```

__② MLP Adapter（今回の主役）__

ここが検証対象。

__基本設計__

```
MLP Adapter:
  Linear(Dv → D_hidden)
  GELU
  Linear(D_hidden → Dl)
```

__推奨パラメータ__

* `Dv = 768`（ViT-B）
* `Dl = 4096`（LLaMA系なら）
* `D_hidden = 1024〜2048`

__重要ポイント__

* **全トークンに対して適用（token-wise MLP）**
* poolingしない（情報損失防止）
* LayerNormを前後に入れると安定

__③ Projection / Token Mapping__

2パターンあります：

__パターンA（シンプル）__

```
Adapter出力 → そのままLLMのembeddingとして使う
```

__パターンB（推奨）__

```
Adapter出力 → Linear → LLM埋め込み次元に完全一致
```

理由：

* LLMの埋め込み空間との整合性が改善

__④ LLM（言語モデル）__

__選択肢__

* 軽量：

  * LLaMA-7B（4bit）
  * Mistral-7B（4bit）
* Colabなら：

  * **4bit量子化 + LoRA or Freeze**

__推奨設定__

* 最初は **完全凍結**
* うまくいかなければ：

  * LoRAを追加（q,k,vのみ）

### 2. 入力フォーマット（重要）

Captioningなので、以下の形式にします：

```
<image_tokens> + "Describe this image:"
```

例：

```
[VISUAL TOKENS] + "A photo of"
```

または instruction形式：

```
"### Instruction: Describe the image\n### Response:"
```

### 3. 学習対象（最小構成）

Colab前提なら：

| モジュール          | 学習         |
| -------------- | ---------- |
| Vision Encoder | ❌          |
| MLP Adapter    | ✅          |
| LLM            | ❌（or LoRA） |

👉 **まずはAdapterのみ学習がベスト**

### 4. Loss設計

標準的な：

```
Cross Entropy Loss（次トークン予測）
```

注意点：

* visual token部分はloss計算から除外
* text部分のみmask

### 5. Colab最適化戦略

__メモリ対策__

* 4bit量子化（必須）
* gradient checkpointing
* batch size = 1〜4
* sequence length制限（<= 512）

__学習安定化__

* learning rate：

  * Adapter: 1e-4 〜 5e-4
* warmupあり

### 6. 最小構成（実験用）

```
Vision Encoder: CLIP ViT-B/16（freeze）
Adapter: 2-layer MLP
LLM: LLaMA-7B（4bit freeze）
```

👉 この構成で

* 「Adapterだけでcaptioning成立するか」
  を純粋に検証可能

### 7. 期待される結果と限界

__うまくいくケース__

* 簡単なcaption

  * "a dog"
  * "a man riding a bike"

__難しいケース__

* 詳細説明
* 抽象的理解
* 長文生成

理由：

* cross-modal alignmentが弱い（Q-Former不在）

### まとめ

Colab前提なら最適解は：

* Vision：凍結
* LLM：凍結（or LoRA）
* **MLP Adapterのみ学習**
* token-level接続（poolしない）

これで「MLP AdapterだけでVLMが成立するか」という研究的にクリーンな検証ができます。

## 実装

コードは以下に保存しています。

https://github.com/Shinichi0713/LLM-fundamental-study/tree/main/vlm/src/mlp_adapter/src

## 実験

モデルはこのように実装します。

LLMはhugging faceより任意のモデルをロード。
VisionEncoderはClip。
MLPAdapterは

1. nn.LayerNorm(dv)：入力の正規化（ガードレール）

Vision Encoder（ViTなど）から出てくる特徴量は、画像によって数値のレンジがバラつくことがあります。

2. nn.Linear(dv, hidden) + nn.GELU()：非線形な「解釈」

単層の行列変換ではなく、間に hidden（1024次元）を挟んだ2層構造にしています。

意図: 単純な線形変換（情報のスケーリング）だけでは不十分な、複雑な特徴の組み換えを可能にします。GELU を挟むことで、「特定の視覚パターンがあるときにだけ反応する」といった非線形な表現力をアダプターに持たせています。

3. nn.Linear(hidden, dl)：言語空間への「写像」

ここで最終的に、LLMが理解できる次元（dl=2560）に引き上げます。

意図: 視覚的な「概念」を、LLMの埋め込み空間（Embedding Space）のどこに配置すべきかを決定します。ここが実質的な「視覚と言語の翻訳機」の出口となります。

4. nn.LayerNorm(dl)：出力のアライメント（仕上げ）

最後に再び LayerNorm を入れています。これが非常に重要です。

意図: LLM（今回の例では dl=2560 なので、Phi-2やGemmaなどの小型モデルクラスを想定）の入力埋め込みは、通常非常にきれいに正規化された分布を持っています。アダプターの出力を最後に整えることで、LLMが「これは自分が知っている言葉のベクトルだ」と認識しやすくするための仕上げです。

```python

# MLP Adapter
class MLPAdapter(nn.Module):
    def __init__(self, dv=768, dl=2560, hidden=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dv),
            nn.Linear(dv, hidden),
            nn.GELU(),
            nn.Linear(hidden, dl),
            nn.LayerNorm(dl)
        )

    def forward(self, x):
        return self.net(x)

class SimpleVLM(nn.Module):
    def __init__(self, vision_model, llm, adapter):
        super().__init__()
        self.vision_model = vision_model
        self.llm = llm
        self.adapter = adapter

        # 凍結
        for p in self.vision_model.parameters():
            p.requires_grad = False
        for p in self.llm.parameters():
            p.requires_grad = False

    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        # Vision
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        vision_embeds = vision_outputs.last_hidden_state  # (B, N, Dv)

        # Adapter
        visual_tokens = self.adapter(vision_embeds)  # (B, N, Dl)

        # Text embedding
        text_embeds = self.llm.get_input_embeddings()(input_ids)

        # concat
        inputs_embeds = torch.cat([visual_tokens, text_embeds], dim=1)

        # mask
        visual_mask = torch.ones(
            visual_tokens.size()[:2],
            dtype=attention_mask.dtype
        ).to(device)

        attention_mask = torch.cat([visual_mask, attention_mask], dim=1)

        # labels
        if labels is not None:
            ignore = torch.full(
                visual_tokens.size()[:2],
                -100
            ).to(device)
            labels = torch.cat([ignore, labels], dim=1)

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )

        return outputs
```

### 実験内容

![1774083511417](image/README/1774083511417.png)

この画像を見て、単純に"A dog looking the master"と出力すればOK。

学習コードは先ほどのレポジトリに保存しています。

__結果__

```python
model.eval()


outputs = model(
        pixel_values=image_inputs["pixel_values"],
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
# logitsから最も確率の高いトークンIDを取り出す
predicted_token_ids = torch.argmax(outputs.logits, dim=-1) # [1, seq_len]

# Tokenizerでデコードして文章にする
predicted_text = tokenizer.decode(predicted_token_ids[0])
print(f"Predicted Text: {predicted_text}")
```

出力

```
 with A dog looking the master
```

期待通り。
但し、汎用性があるかは別問題です。
ですが、画像を認識はするようになりました。


## 一般論

Q-FormerとMLP Adapterではどちらが性能が高いと言われているでしょうか。

結論から言うと、現在のトレンドや多くのベンチマーク結果では、**「MLP Adapter」の方が精度（特に推論能力や汎用性）が高い** とされる場面が増えています。

かつては「Q-Former」が洗練された手法として注目されましたが、なぜ現在はシンプルな「MLP Adapter」が精度面でも優位に立っているのか、その理由を比較・解説します。


### 1. 精度と性能の比較まとめ

| 評価項目 | Q-Former (例: BLIP-2) | MLP Adapter (例: LLaVA-1.5/Next) |
| :--- | :--- | :--- |
| **情報の保持力** | クエリによる「要約」のため、細部が落ちやすい | 全パッチをそのまま渡すため、情報密度が高い |
| **推論能力** | 記述や検索には強いが、複雑な推論はLLMに依存 | LLMの事前学習知識を直接活用しやすく、推論に強い |
| **学習の収束** | 構造が複雑で、アライメントに時間がかかる | シンプルなため、短期間で高い精度に到達する |
| **トークン効率** | **◎ 非常に高い**（数十トークンに圧縮） | **△ 低い**（数百〜数千トークン消費） |



### 2. なぜ「MLP Adapter」の方が精度が出やすいのか？

近年の研究（LLaVA-1.5以降の論文など）で明らかになってきたのは、**「アダプターは余計なことをせず、ただの翻訳役に徹したほうがLLMの知能を引き出せる」** という事実です。

* **「生」の情報を届ける:**
    Q-Formerは「学習可能なクエリ」を使って画像を要約しますが、これは一種の「情報のボトルネック」になります。一方、MLPは画像エンコーダが捉えた情報をほぼそのままLLMの次元に変換して流し込むため、LLMが画像の細部（小さな文字や物体の位置関係）をより正確に把握できます。
* **LLMの「言語空間」への適合:**
    最近の非常に強力なLLM（Llama 3やQwenなど）は、入力が多少冗長でもそれを処理する高い能力を持っています。そのため、無理に圧縮（Q-Former）するよりも、多少トークン数が多くなっても正確な情報を渡す（MLP）方が、最終的な回答精度が高くなる傾向にあります。


### 3. どちらを選ぶべきか？

精度だけで選ぶなら「MLP Adapter」が優勢ですが、**利用環境**によって正解は変わります。

#### **MLP Adapter を選ぶべきケース（精度・推論重視）**
* Google Colab Proや高性能GPUサーバーを使える。
* 画像内の細かい文字を読ませたい（OCR）、複雑な理由を説明させたい。
* LLaVAなどの最新のオープンソース資産を活用したい。

#### **Q-Former を選ぶべきケース（効率・コスト重視）**
* 推論時のメモリ消費（VRAM）を極限まで抑えたい。
* 動画のように、大量のフレームを入力する必要があり、トークン圧縮が必須。
* 画像検索やマッチングなど、情報の「要約ベクトル」が欲しい。


## 総括

本日はVLMの視覚情報をLLMに伝えるBridgeをMLP Adapterで実装、学習してみました。
視覚認識することは確認出来ました。

後は、大きい画像に対応する技術が確認取れれば、VLMのstudyは一段落としようと思います。
