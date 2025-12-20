import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16
from transformers import BertTokenizer, BertModel


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

def clip_loss(logits):
    batch_size = logits.size(0)
    labels = torch.arange(batch_size, device=logits.device)

    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)

    return (loss_i2t + loss_t2i) / 2

def train(dataloader):
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


