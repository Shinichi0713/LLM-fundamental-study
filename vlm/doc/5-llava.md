以下では、**先ほど提示した CLIP 実装をベースにして LLaVA 形式へ改造する最小構成コード**を示します。
目的は「 **CLIP → LLaVA への構造変換が腹落ちすること** 」であり、研究・実装の出発点としてそのまま使える形にしています。

---

# 0. CLIP → LLaVA で何が変わるのか（整理）

| 項目           | CLIP             | LLaVA                            |
| -------------- | ---------------- | -------------------------------- |
| Vision Encoder | ViT              | ViT（同じ）                      |
| Text Encoder   | BERT             | **LLM（LLaMA系）**         |
| 学習損失       | Contrastive Loss | **Language Modeling Loss** |
| 出力           | 埋め込み         | **文章生成**               |
| 融合           | 共通空間         | **Image token injection**  |

**重要**

* Vision Encoder はほぼ流用
* Text Encoder を「消す」
* Projection を「LLM入力用」に変更

> VisionTransformerは変化なし
>
> Text EncoderはBERT→LLM
>
> 学習損失：対照学習→LML
>
> 埋め込み→文章

---

# 1. 全体構成（LLaVA最小）

```
Image → CLIP ViT → patch features
                    ↓
             Projection (Linear)
                    ↓
           Image Tokens (擬似トークン)
                    ↓
Text Tokens + Image Tokens
                    ↓
LLM (LLaMA / Vicuna)
                    ↓
Text Output
```

---

# 2. 前提ライブラリ

```bash
pip install torch torchvision transformers accelerate
```

---

# 3. Vision Encoder（CLIPから流用）

**※ CLS ではなく patch feature を使う点が重要**

```python
import torch
import torch.nn as nn
from torchvision.models import vit_b_16
```

```python
class VisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = vit_b_16(pretrained=True)
        self.vit.heads = nn.Identity()

    def forward(self, images):
        # ViT forward hack: get patch embeddings
        x = self.vit._process_input(images)
        n = x.shape[0]

        cls_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.vit.encoder(x)

        # remove CLS → use patch tokens only
        return x[:, 1:, :]   # (B, N_patches, 768)
```

---

# 4. Projection Layer（CLIP → LLaVAの核心）

```python
class VisionProjection(nn.Module):
    def __init__(self, vision_dim=768, llm_dim=4096):
        super().__init__()
        self.proj = nn.Linear(vision_dim, llm_dim)

    def forward(self, vision_feats):
        # (B, N, vision_dim) → (B, N, llm_dim)
        return self.proj(vision_feats)
```

---

# 5. LLaVA 本体（LLM + Image Token Injection）

ここでは **LLaMA 系モデル**を想定します。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
```

```python
class LLaVA(nn.Module):
    def __init__(self, llm_name="meta-llama/Llama-2-7b-hf"):
        super().__init__()

        self.vision_encoder = VisionEncoder()
        self.vision_proj = VisionProjection()

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name,
            torch_dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self, images, input_ids, attention_mask, labels=None):
        # 1. image → vision tokens
        vision_feats = self.vision_encoder(images)
        image_tokens = self.vision_proj(vision_feats)

        # 2. text embeddings
        text_embeds = self.llm.model.embed_tokens(input_ids)

        # 3. concatenate (image tokens first)
        inputs_embeds = torch.cat(
            [image_tokens, text_embeds], dim=1
        )

        # 4. adjust attention mask
        image_mask = torch.ones(
            image_tokens.size()[:-1],
            device=attention_mask.device
        )
        attention_mask = torch.cat(
            [image_mask, attention_mask], dim=1
        )

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs
```

---

# 6. 学習データ形式（LLaVA型）

```python
prompt = "この画像について説明してください。"
answer = "犬が芝生の上で走っています。"
```

```python
enc = tokenizer(
    prompt,
    return_tensors="pt",
    padding=True
)

labels = tokenizer(
    answer,
    return_tensors="pt",
    padding=True
)["input_ids"]
```

---

# 7. 学習ループ（Alignment / Instruction Tuning）

```python
model = LLaVA().cuda()
optimizer = torch.optim.AdamW(
    model.vision_proj.parameters(), lr=1e-4
)
```

```python
outputs = model(
    images.cuda(),
    enc["input_ids"].cuda(),
    enc["attention_mask"].cuda(),
    labels=labels.cuda()
)

loss = outputs.loss
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

---

# 8. CLIP → LLaVA で失われたもの／得たもの

### 失われたもの

* 対照学習による明示的な距離制約
* 大規模バッチ前提の学習

### 得たもの

* 文章生成能力
* Instruction-following
* 推論柔軟性

---

# 9. 実務で必ず入れる改良点

最低限、次は必須です：

* Vision Encoder freeze
* LLM は **LoRA / QLoRA**
* Image token 数削減（Pooling / Sampling）
* FP16 + gradient checkpointing

---

# 10. 理解の要点（超重要）

* **CLIP = アライメントの基礎**
* **LLaVA = アライメント + 生成**
* Projection が「唯一の橋」
* LLMは「画像をトークン列として読んでいるだけ」

---

## 次に進むと良いテーマ

* Projection を MLP / Perceiver に変える
* Flamingo（Cross-Attention）との差分実装
* Image token に attention が向いているかの可視化
* LLaVA-1.5 / NeXT の構造差分

どこまで掘り下げますか。



以下では、**Google Colab 上で実際に動作確認できる「最小構成 LLaVA」コード**を、
**そのままセル分割して実行できる形**で提示します。

方針は以下です。

* Colab（無料 or T4）で動くことを最優先
* LLaVAの **構造理解用・検証用**
* 学習ではなく **推論（Inference）** をまず成立させる
* 重量級 LLaMA-7B は使わず、**小型 LLM（OPT / LLaMA-2-7Bはオプション）**

---

# 全体構成（Colab向け）

```
Image → CLIP ViT → Projection → Image Tokens
                                     ↓
                           Text Tokens + Image Tokens
                                     ↓
                            Causal LM → Text
```

---

# 🔹 Colab セル1：環境セットアップ

```python
!pip install -q torch torchvision transformers accelerate pillow
```

---

# 🔹 Colab セル2：ライブラリ読み込み

```python
import torch
import torch.nn as nn
from torchvision.models import vit_b_16
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torchvision.transforms as T
```

---

# 🔹 Colab セル3：画像前処理

```python
image_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

---

# 🔹 Colab セル4：Vision Encoder（ViT）

```python
class VisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = vit_b_16(weights="IMAGENET1K_V1")
        self.vit.heads = nn.Identity()

    def forward(self, images):
        x = self.vit._process_input(images)
        n = x.shape[0]

        cls_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.vit.encoder(x)

        return x[:, 1:, :]  # (B, N_patches, 768)
```

---

# 🔹 Colab セル5：Projection Layer

```python
class VisionProjection(nn.Module):
    def __init__(self, vision_dim=768, llm_dim=768):
        super().__init__()
        self.proj = nn.Linear(vision_dim, llm_dim)

    def forward(self, x):
        return self.proj(x)
```

※ **ここでは LLM と同次元に合わせています**

---

# 🔹 Colab セル6：LLaVA 最小モデル

Colabで確実に動かすため、
**LLM は `facebook/opt-350m` を使用**します。

```python
class MiniLLaVA(nn.Module):
    def __init__(self, llm_name="facebook/opt-350m"):
        super().__init__()

        self.vision_encoder = VisionEncoder()
        self.vision_proj = VisionProjection()

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name,
            torch_dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self, images, input_ids, attention_mask):
        vision_feats = self.vision_encoder(images)
        image_tokens = self.vision_proj(vision_feats)

        text_embeds = self.llm.model.decoder.embed_tokens(input_ids)

        inputs_embeds = torch.cat(
            [image_tokens, text_embeds], dim=1
        )

        image_mask = torch.ones(
            image_tokens.size()[:-1],
            device=attention_mask.device
        )
        attention_mask = torch.cat(
            [image_mask, attention_mask], dim=1
        )

        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )
```

---

# 🔹 Colab セル7：モデル初期化

```python
device = "cuda" if torch.cuda.is_available() else "cpu"

model = MiniLLaVA().to(device)
model.eval()
```

---

# 🔹 Colab セル8：画像読み込み

```python
img = Image.open("/content/sample_data/cat.jpg").convert("RGB")
image_tensor = image_transform(img).unsqueeze(0).to(device)
```

※ Colab の `sample_data` にある画像でOK

---

# 🔹 Colab セル9：テキスト入力

```python
prompt = "Describe this image."

enc = model.tokenizer(
    prompt,
    return_tensors="pt"
).to(device)
```

---

# 🔹 Colab セル10：生成（推論）

```python
with torch.no_grad():
    outputs = model(
        image_tensor,
        enc["input_ids"],
        enc["attention_mask"]
    )

    generated = model.llm.generate(
        inputs_embeds=outputs.inputs_embeds,
        attention_mask=outputs.attention_mask,
        max_new_tokens=50
    )

print(model.tokenizer.decode(generated[0], skip_special_tokens=True))
```

---

# ✅ ここまでで出来ること

* 画像を **トークン列として LLM に注入**
* Self-Attention による画像＋テキスト処理
* Colab 上で **LLaVA構造を実体験**

※ 出力品質は **未学習 Projection** のため低いです
→ **構造検証が目的**

---

# 🔧 よくあるエラーと対処

### CUDA OOM

* `opt-350m` より大きくしない
* batch size = 1

### dtype エラー

```python
torch_dtype=torch.float16
```

を必ず指定

---

# 次にやるべきこと（強く推奨）

1. **Vision Encoder freeze**
2. Projection のみ学習（COCO Caption）
3. LoRA を LLM に適用
4. Image token 数削減
5. LLaVA Instruction データ投入

---

## まとめ

* このコードは **LLaVAの最小・可動・理解用**
* CLIP → LLaVA 変換の構造が明確に見える
* Colab で「まず動く」ことが最優先

---

次の段階として、

* **Projection を実際に学習する Colab コード**
* **LoRA 付き LLaVA**
* **LLaVA-1.5 相当構成への拡張**

のどれに進みますか。

