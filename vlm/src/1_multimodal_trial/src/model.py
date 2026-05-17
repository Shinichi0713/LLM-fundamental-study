import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor, AutoModelForCausalLM, AutoTokenizer
from PIL import Image

class SimpleLLaVA(nn.Module):
    """
    LLaVAの基本構造を再現した簡易モデル
    - CLIPで画像特徴を抽出
    - 線形層でLLMの埋め込み空間に射影
    - 画像トークン + テキストトークンをLLMに入力
    """
    def __init__(self, vision_model_name="openai/clip-vit-large-patch14",
                 language_model_name="meta-llama/Llama-2-7b-chat-hf"):
        super().__init__()

        # ビジョンエンコーダ（CLIP）
        self.vision_encoder = CLIPModel.from_pretrained(vision_model_name).vision_model
        # 画像特徴の次元（例：1024）
        self.vision_hidden_size = self.vision_encoder.config.hidden_size

        # 言語モデル（LLaMA / Vicuna 風）
        self.language_model = AutoModelForCausalLM.from_pretrained(language_model_name)
        # LLMの埋め込み次元
        self.text_hidden_size = self.language_model.config.hidden_size

        # 画像特徴をLLMの埋め込み空間に射影するプロジェクション層
        self.image_proj = nn.Linear(self.vision_hidden_size, self.text_hidden_size)

        # トークナイザ
        self.tokenizer = AutoTokenizer.from_pretrained(language_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self, images, input_text):
        """
        images: PIL画像のリスト or 前処理済みテンソル
        input_text: テキストのリスト
        """
        # 1. 画像特徴の抽出（CLIP）
        # 実際にはCLIPProcessorで前処理が必要ですが、ここでは簡略化
        if isinstance(images, list):
            # PIL画像をテンソルに変換（簡易）
            # 実際にはCLIPProcessorを使うべき
            from torchvision.transforms import ToTensor, Resize, CenterCrop, Normalize
            transform = nn.Sequential(
                Resize((224, 224)),
                CenterCrop((224, 224)),
                ToTensor(),
                Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711])
            )
            images = torch.stack([transform(img) for img in images]).to(self.language_model.device)

        # CLIPで画像特徴を取得（最後の隠れ状態）
        vision_outputs = self.vision_encoder(pixel_values=images)
        image_features = vision_outputs.last_hidden_state  # (batch, num_patches, hidden_size)

        # 2. 画像特徴をLLMの埋め込み空間に射影
        image_embeds = self.image_proj(image_features)  # (batch, num_patches, text_hidden_size)

        # 3. テキストトークン化
        text_inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.language_model.device)
        input_ids = text_inputs["input_ids"]  # (batch, text_len)
        attention_mask = text_inputs["attention_mask"]

        # 4. 画像トークンとテキストトークンを結合
        batch_size = image_embeds.size(0)
        num_image_tokens = image_embeds.size(1)

        # テキスト埋め込みを取得
        text_embeds = self.language_model.get_input_embeddings()(input_ids)  # (batch, text_len, hidden_size)

        # 画像トークン + テキストトークン
        inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)  # (batch, num_image_tokens+text_len, hidden_size)

        # アテンションマスクも拡張（画像トークン部分は1）
        image_mask = torch.ones(batch_size, num_image_tokens, device=attention_mask.device)
        attention_mask = torch.cat([image_mask, attention_mask], dim=1)  # (batch, num_image_tokens+text_len)

        # 5. LLMで生成
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True
        )

        return outputs

    def generate(self, images, prompt, max_new_tokens=50):
        """
        画像とプロンプトからテキストを生成（簡易版）
        """
        # 1. 画像特徴 + プロンプト埋め込みを構築（上記forwardと同様）
        if isinstance(images, list):
            from torchvision.transforms import ToTensor, Resize, CenterCrop, Normalize
            transform = nn.Sequential(
                Resize((224, 224)),
                CenterCrop((224, 224)),
                ToTensor(),
                Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711])
            )
            images = torch.stack([transform(img) for img in images]).to(self.language_model.device)

        vision_outputs = self.vision_encoder(pixel_values=images)
        image_features = vision_outputs.last_hidden_state
        image_embeds = self.image_proj(image_features)

        # プロンプトのトークン化
        text_inputs = self.tokenizer(
            [prompt],
            return_tensors="pt"
        ).to(self.language_model.device)
        input_ids = text_inputs["input_ids"]
        attention_mask = text_inputs["attention_mask"]

        # 埋め込み
        text_embeds = self.language_model.get_input_embeddings()(input_ids)

        # 結合
        batch_size = image_embeds.size(0)
        num_image_tokens = image_embeds.size(1)
        inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)
        image_mask = torch.ones(batch_size, num_image_tokens, device=attention_mask.device)
        attention_mask = torch.cat([image_mask, attention_mask], dim=1)

        # 2. 生成（簡易greedy decoding）
        generated_ids = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # 生成された部分だけをデコード
        # 入力トークン数（画像+プロンプト）を除く
        input_len = inputs_embeds.size(1)
        generated_text = self.tokenizer.decode(generated_ids[0][input_len:], skip_special_tokens=True)

        return generated_text



# モデルの初期化（実際には事前学習済みチェックポイントをロード）
model = SimpleLLaVA(
    vision_model_name="openai/clip-vit-large-patch14",
    language_model_name="meta-llama/Llama-2-7b-chat-hf"  # 実際にはLLaVA用のLLMを使う
)
model.eval()
model.to("cuda" if torch.cuda.is_available() else "cpu")

# 画像読み込み
image_path = "example.jpg"
image = Image.open(image_path).convert("RGB")

# プロンプト
prompt = "Describe this image in detail."

# 生成
with torch.no_grad():
    response = model.generate(images=[image], prompt=prompt, max_new_tokens=100)

print("Prompt:", prompt)
print("Response:", response)