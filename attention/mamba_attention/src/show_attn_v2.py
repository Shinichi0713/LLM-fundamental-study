import torch
from transformers import AutoTokenizer, MambaForCausalLM
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def analyze_mamba_hf_sensitivity(model_name, text):
    # 1. モデルとトークナイザーのロード
    # device_map="auto" でGPUへ。CUDAが使えない場合は "cpu" になります。
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = MambaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).cuda()
    model.eval()

    # 2. 入力の準備
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    input_ids = inputs["input_ids"]
    token_labels = [tokenizer.decode([t]) for t in input_ids[0]]

    # 3. Embedding層の出力を取得し、勾配を有効化
    # Hugging Face版MambaのEmbedding層は model.backbone.embeddings
    embeddings = model.backbone.embeddings(input_ids)
    embeddings.retain_grad()
    embeddings.requires_grad_(True)

    # 4. 順伝播 (Forward Pass)
    # 直接 model(inputs) とすると Embedding 計算が重複するため、
    # inputs_embeds 引数を使用して計算を開始します
    outputs = model(inputs_embeds=embeddings, output_hidden_states=True)
    last_hidden_state = outputs.hidden_states[-1] # 最終レイヤーの出力
    
    # 5. ターゲットの指定 (最後のトークンのベクトルを対象にする)
    target_token_idx = -1 
    # 最後のトークンの出力ベクトルのノルムをスカラーとして取得
    target_value = last_hidden_state[0, target_token_idx].norm()

    # 6. 逆伝播 (Backward Pass)
    model.zero_grad()
    target_value.backward()

    # 7. 勾配の抽出と加工
    if embeddings.grad is not None:
        # (batch, seq_len, d_model) -> 各トークンの寄与度 (seq_len,)
        grads = embeddings.grad.abs().sum(dim=-1).squeeze(0)
        
        # 可視化のために 0.0 - 1.0 にスケーリング
        grads_norm = (grads - grads.min()) / (grads.max() - grads.min() + 1e-8)
        scores = grads_norm.cpu().detach().numpy()
    else:
        print("Error: Gradient not captured.")
        return None, None

    return token_labels, scores

def plot_sensitivity(tokens, scores):
    plt.figure(figsize=(14, 3))
    sns.heatmap([scores], annot=[tokens], fmt="", cmap="YlGnBu", cbar=True)
    plt.title("Effective Receptive Field: Token Sensitivity for the Last Output")
    plt.xlabel("Input Sequence")
    plt.tight_layout()
    plt.show()

# --- 実行 ---
model_id = "state-spaces/mamba-130m-hf"  # 軽量な130Mモデル
sample_text = "The quick brown fox jumps over the lazy dog."

tokens, scores = analyze_mamba_hf_sensitivity(model_id, sample_text)

if tokens and scores is not None:
    plot_sensitivity(tokens, scores)