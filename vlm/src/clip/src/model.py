import torch
import torch.nn as nn
from transformers import CLIPVisionModel, AutoModelForCausalLM, AutoTokenizer

class SimpleVLM(nn.Module):
    def __init__(self, vision_model_name, llm_model_name):
        super().__init__()
        
        # 1. Vision Encoder: 画像をベクトル化する「目」
        # CLIPなどの学習済みモデルを使用。最終層のパッチ特徴量を取り出す。
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_model_name)
        vision_hidden_size = self.vision_encoder.config.hidden_size
        
        # 2. LLM: 思考と生成を行う「脳」
        # 今回はデコーダーのみの言語モデル（例: Llama, TinyLlama）を想定
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_name)
        llm_hidden_size = self.llm.config.hidden_size
        
        # 3. Connector (Projector): 視覚と脳を繋ぐ「神経」
        # Visionの次元をLLMの次元に変換する2層のMLP。これが画像と言語を「整列」させる。
        self.connector = nn.Sequential(
            nn.Linear(vision_hidden_size, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )
        
        # Vision Encoderの重みは基本固定（Frozen）にすることが多い
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

    def forward(self, image_pixels, input_ids):
        """
        image_pixels: (B, C, H, W) - 前処理済みの画像
        input_ids: (B, T) - トークナイズされた質問テキスト
        """
        # --- 視覚トークンの生成 ---
        # Vision Encoderから特徴抽出 (B, Patch_Num, Vision_Dim)
        vision_outputs = self.vision_encoder(image_pixels)
        last_hidden_state = vision_outputs.last_hidden_state # 画像パッチの特徴量
        
        # ConnectorでLLMの次元へ変換 (B, Patch_Num, LLM_Dim)
        visual_tokens = self.connector(last_hidden_state)
        
        # --- テキストトークンの埋め込み ---
        # LLMのEmbedding層を使いテキストをベクトル化 (B, T, LLM_Dim)
        text_embeddings = self.llm.get_input_embeddings()(input_ids)
        
        # --- 統合 ---
        # 視覚トークンとテキストトークンを連結 (B, Patch_Num + T, LLM_Dim)
        # LLMから見れば、画像は「非常に情報量の多い単語」の並びに見える
        combined_embeds = torch.cat([visual_tokens, text_embeddings], dim=1)
        
        # LLMに流し込む
        outputs = self.llm(inputs_embeds=combined_embeds)
        return outputs.logits

# --- インスタンス化の例 ---
# model = SimpleVLM("openai/clip-vit-base-patch32", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")