import torch
import torch.nn as nn
from transformers import (
    CLIPVisionModel, CLIPImageProcessor,
    AutoTokenizer, AutoModelForCausalLM, AutoConfig
)
from peft import LoraConfig, get_peft_model

device = "cuda" if torch.cuda.is_available() else "cpu"

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
    

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "microsoft/phi-2"

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# config修正（重要）
config = AutoConfig.from_pretrained(model_name)
config.pad_token_id = tokenizer.eos_token_id

# model
llm = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=config,
    torch_dtype=torch.float16,
    device_map="auto"
)

# tokenizer側も統一
tokenizer.pad_token = tokenizer.eos_token

# 安定化
llm.config.pad_token_id = tokenizer.eos_token_id
llm.config.use_cache = False

# ---- Vision ----
vision_model = CLIPVisionModel.from_pretrained(
    "openai/clip-vit-base-patch16"
).to(device)

processor = CLIPImageProcessor.from_pretrained(
    "openai/clip-vit-base-patch16"
)

# ---- Adapter ----
adapter = MLPAdapter(
    dv=vision_model.config.hidden_size,
    dl=llm.config.hidden_size,  # 2560
    hidden=1024
).to(device)

# ---- VLM ----
model = SimpleVLM(vision_model, llm, adapter).to(device)