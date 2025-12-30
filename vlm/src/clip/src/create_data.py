from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn

# STAIR CaptionsはMS COCOをベースにしているので、
# Hugging Face上の日本語キャプション付きデータセットを利用するのが近道です
# 例: yagays/stair-captions (公開されている場合) 
dataset = load_dataset("stair_captions", "default")

# データの構造確認
# examples['image'] に画像、examples['caption'] に日本語テキストがある状態にします
print(dataset['train'][0])



class JapaneseCLIP(nn.Module):
    def __init__(self, image_encoder, text_model_name="cl-tohoku/bert-base-japanese-v3", embed_dim=512):
        super().__init__()
        self.image_encoder = image_encoder # 既存のCLIP画像エンコーダーを流用
        
        # 日本語BERTをテキストエンコーダーとして採用
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        
        # BERTの768次元をCLIPの共通空間（例: 512次元）に変換
        self.text_projection = nn.Linear(768, embed_dim)
        
    def get_text_features(self, text):
        inputs = self.text_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = self.text_encoder(**inputs)
        # [CLS]トークンの特徴量を使用
        return self.text_projection(outputs.last_hidden_state[:, 0, :])


