import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    CLIPVisionModel, AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from PIL import Image


class ImprovedLLaVAProjector(nn.Module):
    """
    LLaVA-1.5で採用されたMLPプロジェクタ（2層 + LayerNorm）
    """
    def __init__(self, vision_hidden_size, text_hidden_size, intermediate_size=4096):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(vision_hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, text_hidden_size),
        )
        self.norm = nn.LayerNorm(text_hidden_size)

    def forward(self, x):
        # x: (batch, num_patches, vision_hidden_size)
        x = self.mlp(x)
        x = self.norm(x)
        return x


class ImprovedLLaVA(nn.Module):
    """
    性能向上を意識したLLaVA実装
    - MLPプロジェクタ（LLaVA-1.5風）
    - マルチターン対話対応（<image>トークン）
    - LoRA対応（言語モデル側のみ）
    """
    def __init__(self,
                 vision_model_name="openai/clip-vit-large-patch14",
                 language_model_name="meta-llama/Llama-2-7b-chat-hf",
                 use_lora=True,
                 lora_r=16,
                 lora_alpha=32,
                 lora_dropout=0.1):
        super().__init__()

        # ビジョンエンコーダ（CLIP ViT-L/14）
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_model_name)
        self.vision_hidden_size = self.vision_encoder.config.hidden_size  # 1024

        # 言語モデル（LLaMA-2 / Vicuna）
        # 4bit量子化＋LoRAでメモリ節約
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.language_model = AutoModelForCausalLM.from_pretrained(
            language_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        self.text_hidden_size = self.language_model.config.hidden_size  # 4096

        # LoRA設定（言語モデル側のみ）
        if use_lora:
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
            )
            self.language_model = get_peft_model(self.language_model, lora_config)

        # 改良版プロジェクタ（LLaVA-1.5風）
        self.image_proj = ImprovedLLaVAProjector(
            vision_hidden_size=self.vision_hidden_size,
            text_hidden_size=self.text_hidden_size,
            intermediate_size=4096
        )

        # トークナイザ
        self.tokenizer = AutoTokenizer.from_pretrained(language_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # <image>トークンを追加（マルチターン対話用）
        self.image_token = "<image>"
        self.tokenizer.add_tokens([self.image_token], special_tokens=True)
        self.language_model.resize_token_embeddings(len(self.tokenizer))

    def encode_images(self, images):
        """
        画像をCLIPでエンコードし、プロジェクタでLLM空間に射影
        """
        if isinstance(images, list):
            from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
            transform = Compose([
                Resize((224, 224)),
                CenterCrop((224, 224)),
                ToTensor(),
                Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711])
            ])
            images = torch.stack([transform(img) for img in images]).to(self.vision_encoder.device)

        vision_outputs = self.vision_encoder(pixel_values=images)
        image_features = vision_outputs.last_hidden_state  # (batch, num_patches, 1024)
        image_embeds = self.image_proj(image_features)      # (batch, num_patches, 4096)
        return image_embeds

    def build_multimodal_inputs(self, images, texts):
        """
        画像とテキストから、マルチモーダル入力（埋め込み＋マスク）を構築
        - texts: ["<image> Describe this image.", ...] のような形式
        """
        batch_size = len(texts)

        # 画像エンコード
        image_embeds = self.encode_images(images)  # (batch, num_patches, hidden)
        num_image_tokens = image_embeds.size(1)

        # テキストトークン化（<image>を特別扱い）
        text_inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        input_ids = text_inputs["input_ids"].to(self.language_model.device)        # (batch, text_len)
        attention_mask = text_inputs["attention_mask"].to(self.language_model.device)

        # テキスト埋め込み
        text_embeds = self.language_model.get_input_embeddings()(input_ids)  # (batch, text_len, hidden)

        # <image>トークンの位置を特定し、画像埋め込みで置換
        image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        for i in range(batch_size):
            # 各サンプルで<image>トークンの位置を探す
            image_positions = (input_ids[i] == image_token_id).nonzero(as_tuple=True)[0]
            if len(image_positions) == 0:
                continue
            # 最初の<image>を画像トークンに置換（LLaVA-1.5風）
            pos = image_positions[0]
            text_embeds[i, pos:pos+num_image_tokens] = image_embeds[i]

        # アテンションマスクを画像トークン分拡張
        image_mask = torch.ones(batch_size, num_image_tokens, device=attention_mask.device)
        attention_mask = torch.cat([image_mask, attention_mask], dim=1)

        return text_embeds, attention_mask

    def forward(self, images, texts, labels=None):
        """
        学習用フォワード（CrossEntropyLossを計算）
        - images: PIL画像のリスト
        - texts: 入力テキスト（<image>を含む）
        - labels: ターゲットラベル（テキスト部分のみ）
        """
        inputs_embeds, attention_mask = self.build_multimodal_inputs(images, texts)

        # ラベルがなければテキストをそのままターゲットとする（簡易）
        if labels is None:
            # 実際には指示チューニング用のラベル生成が必要
            labels = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True
            )["input_ids"].to(self.language_model.device)

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        return outputs

    def generate(self, images, prompt, max_new_tokens=100, temperature=0.7, top_p=0.9):
        """
        推論用生成（サンプリング付き）
        """
        # プロンプトに<image>を挿入
        full_prompt = f"{self.image_token} {prompt}"
        inputs_embeds, attention_mask = self.build_multimodal_inputs(images, [full_prompt])

        generated_ids = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        # 生成部分のみデコード
        input_len = inputs_embeds.size(1)
        generated_text = self.tokenizer.decode(generated_ids[0][input_len:], skip_special_tokens=True)
        return generated_text
    


# モデル初期化（LoRA＋4bit量子化）
model = ImprovedLLaVA(
    vision_model_name="openai/clip-vit-large-patch14",
    language_model_name="meta-llama/Llama-2-7b-chat-hf",
    use_lora=True
)
model.eval()

# 画像読み込み
image_path = "example.jpg"
image = Image.open(image_path).convert("RGB")

# プロンプト（<image>トークンを含む）
prompt = "Describe this image in detail."

# 生成（サンプリング付き）
with torch.no_grad():
    response = model.generate(images=[image], prompt=prompt, max_new_tokens=150)

print("Prompt:", prompt)
print("Response:", response)