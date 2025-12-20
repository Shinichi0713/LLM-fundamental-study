以下では **CLIP（Contrastive Language–Image Pretraining）** を
**「学習が実際に回る最小構成」** かつ **VLM（LLaVA等）への接続を意識した実装** として示します。

目的は

* CLIPの **本質（対照学習）をコードで理解する**
* 後続の **VLM学習にそのまま流用できる設計**
  です。

---

# 1. 実装全体像

### 構成

* Vision Encoder：ViT（torchvision）
* Text Encoder：Transformer Encoder
* 共通埋め込み空間への Projection
* InfoNCE（CLIP Loss）

```
Image ── ViT ── proj_v ┐
                       ├─ cosine similarity → contrastive loss
Text  ── Transformer ─ proj_t ┘
```

---

# 2. 前提

```bash
pip install torch torchvision transformers
```

---

# 3. CLIPモデル本体（最小構成）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16
from transformers import BertTokenizer, BertModel
```

---

## 3.1 Vision Encoder

```python
class VisionEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.vit = vit_b_16(pretrained=True)
        self.vit.heads = nn.Identity()  # CLS embedding
        self.proj = nn.Linear(768, embed_dim)

    def forward(self, images):
        feats = self.vit(images)        # (B, 768)
        feats = self.proj(feats)         # (B, embed_dim)
        return F.normalize(feats, dim=-1)
```

---

## 3.2 Text Encoder

```python
class TextEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.proj = nn.Linear(768, embed_dim)

    def forward(self, input_ids, attention_mask):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls = out.last_hidden_state[:, 0]  # CLS
        feats = self.proj(cls)
        return F.normalize(feats, dim=-1)
```

---

## 3.3 CLIPモデル統合

```python
class CLIP(nn.Module):
    def __init__(self, embed_dim=512, temperature=0.07):
        super().__init__()
        self.image_encoder = VisionEncoder(embed_dim)
        self.text_encoder = TextEncoder(embed_dim)
        self.logit_scale = nn.Parameter(torch.tensor(1 / temperature))

    def forward(self, images, input_ids, attention_mask):
        img_emb = self.image_encoder(images)
        txt_emb = self.text_encoder(input_ids, attention_mask)

        logits = self.logit_scale.exp() * img_emb @ txt_emb.t()
        return logits
```

---

# 4. CLIP Loss（最重要）

CLIPは **双方向分類問題** として学習します。

```python
def clip_loss(logits):
    batch_size = logits.size(0)
    labels = torch.arange(batch_size, device=logits.device)

    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)

    return (loss_i2t + loss_t2i) / 2
```

---

# 5. 学習ループ（簡易）

```python
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = CLIP().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for images, texts in dataloader:
    images = images.cuda()

    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to("cuda")

    logits = model(images, enc["input_ids"], enc["attention_mask"])
    loss = clip_loss(logits)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"loss: {loss.item():.4f}")
```

---

# 6. 実装上の重要ポイント（必ず理解）

## 6.1 正例・負例の作られ方

* 同一バッチ内で
  * 正例：対応する image–text
  * 負例：それ以外すべて

→ **バッチサイズが性能に直結**

---

## 6.2 温度パラメータ

```python
self.logit_scale = nn.Parameter(torch.tensor(1 / temperature))
```

* 学習可能
* 類似度分布の鋭さを制御
* 学習が不安定ならここを疑う

---

## 6.3 正規化は必須

```python
F.normalize(feats, dim=-1)
```

これがないと CLIP は成立しません。

---

# 7. 推論（Zero-shot検索）

```python
with torch.no_grad():
    img_emb = model.image_encoder(images)
    txt_emb = model.text_encoder(input_ids, attention_mask)

    similarity = img_emb @ txt_emb.t()
```

---

# 8. VLM（LLaVA）への接続ポイント

CLIPで学んだ以下が  **そのままLLaVAに繋がります** ：

| CLIP            | LLaVA          |
| --------------- | -------------- |
| Vision Encoder  | Vision Encoder |
| Image embedding | Image tokens   |
| Projection      | Projection     |
| Alignment       | Alignment      |

LLaVAでは：

* **Text EncoderをLLMに置き換える**
* Contrastive Lossを **LM Loss** に変える

だけです。

---

# 9. よくある失敗

* バッチサイズが小さすぎる
* 正規化忘れ
* temperature固定
* CLSだけでなくパッチを使いたいのに設計ミス

---

# 10. 次にやるべき発展

次のステップとして自然なのは：

1. **ViTのパッチ特徴を使うCLIP**
2. **LAION風大規模データ設計**
3. **CLIP → LLaVA最小構成接続**
4. **Image tokenとしてLLMに注入**

---

必要であれば次は
**「このCLIPをLLaVAに改造するコード」**
または
**「CLIP学習が失敗する典型例のデバッグ」**
を詳しく解説できます。

どこまで進めますか。
