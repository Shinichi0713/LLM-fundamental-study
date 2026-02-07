import torch
import torch.nn as nn
from transformers import (
    CLIPVisionModel,
    CLIPProcessor,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Blip2QFormerModel,
    Blip2QFormerConfig,
)
from peft import LoraConfig, get_peft_model

device = "cuda" if torch.cuda.is_available() else "cpu"

vision_encoder = CLIPVisionModel.from_pretrained(
    "openai/clip-vit-base-patch32"
).to(device)

for p in vision_encoder.parameters():
    p.requires_grad = False

processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)
NUM_QUERIES = 16

qformer_config = Blip2QFormerConfig(
    hidden_size=768,
    num_hidden_layers=6,
    num_attention_heads=12,
    intermediate_size=3072,
    encoder_hidden_size=vision_encoder.config.hidden_size,
    num_query_tokens=NUM_QUERIES,
)

qformer = Blip2QFormerModel(qformer_config).to(device)

# ★重要：query tokens は外部で定義
query_tokens = nn.Parameter(
    torch.randn(1, NUM_QUERIES, qformer_config.hidden_size)
).to(device)


from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import LoraConfig, get_peft_model

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

llm = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn", "c_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

llm = get_peft_model(llm, lora_config)

# LLM 本体 freeze（LoRA以外）
for name, p in llm.named_parameters():
    if "lora_" not in name:
        p.requires_grad = False

llm.print_trainable_parameters()

proj = nn.Linear(
    qformer_config.hidden_size,
    llm.config.n_embd
).to(device)

class FlickrVLM(Dataset):
    def __init__(self, image_dir, captions, image_list, processor, tokenizer):
        self.image_dir = image_dir
        self.captions = captions
        self.image_list = image_list
        self.processor = processor
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img = self.image_list[idx]
        image = Image.open(
            os.path.join(self.image_dir, img)
        ).convert("RGB")

        caption = random.choice(self.captions[img])

        pixel_values = self.processor(
            images=image,
            return_tensors="pt"
        )["pixel_values"].squeeze(0)

        tokens = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=32,
            return_tensors="pt"
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": tokens.input_ids.squeeze(0),
        }


import tqdm

optimizer = torch.optim.AdamW(
    list(qformer.parameters()) +
    [query_tokens] +
    list(proj.parameters()) +
    list(llm.parameters()),
    lr=1e-4
)


qformer.train()
llm.train()

for epoch in range(3):
    total_loss = 0.0

    for batch in tqdm(dataloader):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)

        with torch.no_grad():
            vision_feats = vision_encoder(
                pixel_values
            ).last_hidden_state

        B = pixel_values.size(0)

        # ★修正点：外部 query_tokens を使う
        query_embeds = query_tokens.expand(B, -1, -1)

        q_out = qformer(
            query_embeds=query_embeds,
            encoder_hidden_states=vision_feats,
        ).last_hidden_state

        img_tokens = proj(q_out)

        text_embeds = llm.transformer.wte(input_ids)

        inputs_embeds = torch.cat(
            [img_tokens, text_embeds],
            dim=1
        )

        labels = torch.cat([
            torch.full(
                (B, img_tokens.size(1)),
                -100,
                device=device
            ),
            input_ids
        ], dim=1)

        outputs = llm(
            inputs_embeds=inputs_embeds,
            labels=labels
        )

        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch+1} | "
        f"Loss: {total_loss/len(dataloader):.4f}"
    )

