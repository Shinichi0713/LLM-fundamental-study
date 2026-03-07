import torch
import torch.nn as nn
from transformers import (
    CLIPVisionModel,
    CLIPImageProcessor,
    Blip2QFormerModel,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)
from peft import LoraConfig, get_peft_model
from transformers import Blip2ForConditionalGeneration, AutoProcessor, AutoTokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)

class QFormerVLM(nn.Module):
    def __init__(self, vision_model, qformer, query_tokens, proj, llm):
        super().__init__()
        self.vision_model = vision_model
        self.qformer = qformer
        self.query_tokens = query_tokens  # nn.Parameter
        self.proj = proj                  # Linear(768 → 2048)
        self.llm = llm                    # Flan-T5

    def forward(self, pixel_values, input_ids, attention_mask, labels):
        B = pixel_values.size(0)

        # ======================
        # Vision
        # ======================
        with torch.no_grad():
            vision_outputs = self.vision_model(pixel_values=pixel_values)
            image_embeds = vision_outputs.last_hidden_state  # (B, N, 768)

        # ======================
        # Query tokens
        # ======================
        queries = self.query_tokens.expand(B, -1, -1).contiguous()

        # ======================
        # Q-Former
        # ======================
        q_outputs = self.qformer(
            query_embeds=queries,
            encoder_hidden_states=image_embeds,
            return_dict=True
        )

        q_hidden = q_outputs.last_hidden_state  # (B, num_query, 768)

        # ======================
        # Projection → T5 dim
        # ======================
        q_hidden = self.proj(q_hidden)  # (B, num_query, 2048)

        # ---- T5 scale adjustment (重要) ----
        q_hidden = q_hidden * (self.llm.config.d_model ** -0.5)

        # ======================
        # Encoder attention mask
        # ======================
        encoder_attention_mask = torch.ones(
            B, q_hidden.size(1), device=q_hidden.device
        )

        # ======================
        # T5 Encoder
        # ======================
        encoder_outputs = self.llm.encoder(
            inputs_embeds=q_hidden,
            attention_mask=encoder_attention_mask,
            return_dict=True
        )

        # ======================
        # Decoder (loss)
        # ======================
        outputs = self.llm(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
            labels=labels,
            decoder_attention_mask=attention_mask,
            return_dict=True
        )

        return outputs
device = "cuda" if torch.cuda.is_available() else "cpu"

# Vision Encoder
vision_model = CLIPVisionModel.from_pretrained(
    "openai/clip-vit-base-patch32"
).to(device)

vision_model.eval()
for p in vision_model.parameters():
    p.requires_grad = False

qformer_config = Blip2QFormerConfig(
    hidden_size=768,
    num_hidden_layers=6,
    num_attention_heads=12,
    intermediate_size=3072,
    encoder_hidden_size=vision_model.config.hidden_size
)

qformer = Blip2QFormerModel(qformer_config).to(device)

# Query tokens（重要）
num_query_tokens = 32
query_tokens = torch.nn.Parameter(
    torch.randn(1, num_query_tokens, qformer_config.hidden_size)
).to(device)
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

llm = T5ForConditionalGeneration.from_pretrained(
    "google/flan-t5-base"
).to(device)

# LLMは凍結（BLIP-2戦略）
for p in llm.parameters():
    p.requires_grad = False

proj = torch.nn.Linear(
    qformer_config.hidden_size,
    llm.config.d_model
).to(device)

vision_processor = CLIPImageProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

query_tokens = nn.Parameter(
    torch.randn(num_query_tokens, llm.config.d_model)
)

model = QFormerVLM(
    vision_model,
    qformer,
    query_tokens,
    proj,
    llm
).to(device)
