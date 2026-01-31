# Q-Former minimal training on Google Colab
# ------------------------------------------------------------
# This notebook is a *minimal but correct* implementation to
# verify that Q-Former can bridge vision features to an LLM.
# Designed to run on a single Colab GPU (T4 / A100).

# ============================================================
# 0. Install dependencies (run once)
# ============================================================
# !pip install -q torch torchvision transformers timm einops pillow

# ============================================================
# 1. Imports and global settings
# ============================================================
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from transformers import (
    CLIPVisionModel,
    BertConfig,
    BertModel,
    AutoTokenizer,
    AutoModelForCausalLM,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16

# ============================================================
# 2. Dataset (COCO-style minimal caption dataset)
# ============================================================
# Expected format: list of (image_path, caption)

class CaptionDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, caption = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, caption

# ============================================================
# 3. Frozen Vision Encoder (CLIP ViT-B/16)
# ============================================================
class FrozenCLIPVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-base-patch16"
        )
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, images):
        out = self.vision(pixel_values=images)
        return out.last_hidden_state  # [B, N, 768]

# ============================================================
# 4. Q-Former (minimal BLIP-2 style)
# ============================================================
class QFormer(nn.Module):
    def __init__(self, num_queries=16, vision_dim=768, llm_dim=768):
        super().__init__()
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_queries, llm_dim)
        )

        config = BertConfig(
            hidden_size=llm_dim,
            num_hidden_layers=6,
            num_attention_heads=12,
            encoder_width=vision_dim,
            add_cross_attention=True,
            is_decoder=True,
        )
        self.qformer = BertModel(config)

    def forward(self, vision_feats):
        B = vision_feats.size(0)
        queries = self.query_tokens.expand(B, -1, -1)

        out = self.qformer(
            inputs_embeds=queries,
            encoder_hidden_states=vision_feats,
            encoder_attention_mask=torch.ones(
                vision_feats.shape[:-1],
                device=vision_feats.device
            )
        )
        return out.last_hidden_state  # [B, Q, D]

# ============================================================
# 5. Frozen LLM (GPT-2 for Colab safety)
# ============================================================
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

llm = AutoModelForCausalLM.from_pretrained("gpt2")
llm.to(DEVICE)
llm.eval()
for p in llm.parameters():
    p.requires_grad = False

# ============================================================
# 6. Vision → Q-Former → LLM forward
# ============================================================

def forward_llm(q_tokens, captions):
    tokens = tokenizer(
        captions,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    input_ids = tokens.input_ids.to(DEVICE)

    text_embeds = llm.transformer.wte(input_ids)
    inputs_embeds = torch.cat([q_tokens, text_embeds], dim=1)

    outputs = llm(
        inputs_embeds=inputs_embeds,
        labels=input_ids,
    )
    return outputs.loss

# ============================================================
# 7. Initialize models
# ============================================================
vision_encoder = FrozenCLIPVision().to(DEVICE)
qformer = QFormer(num_queries=16).to(DEVICE)

optimizer = torch.optim.AdamW(qformer.parameters(), lr=1e-4)

# ============================================================
# 8. Dummy data example (replace with COCO subset)
# ============================================================
# Replace this with real image paths on Colab
samples = [
    ("/content/sample_data/your_image.jpg", "A dog running on the grass."),
]

dataset = CaptionDataset(samples)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# ============================================================
# 9. Training loop (minimal)
# ============================================================
for epoch in range(3):
    for images, captions in dataloader:
        images = images.to(DEVICE, dtype=DTYPE)

        with torch.no_grad():
            vision_feats = vision_encoder(images)

        q_tokens = qformer(vision_feats)
        loss = forward_llm(q_tokens, captions)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

# ============================================================
# 10. What to expect
# ============================================================
# Before training:
#   - Captions ignore image content
# After training:
#   - Generated text becomes image-dependent
# ============================================================
