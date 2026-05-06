import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_mamba_sensitivity(model, tokenizer, text):
    # 1. 入力の準備
    tokens = tokenizer.encode(text)
    input_ids = torch.tensor([tokens]).cuda()
    token_labels = [tokenizer.decode([t]) for t in tokens]

    # 2. Embedding層の出力を取得し、勾配追跡を有効にする
    # model.embedding はモデルの構造に合わせて適宜変更してください
    embeddings = model.get_submodule("backbone.embedding")(input_ids)
    embeddings.retain_grad()
    embeddings.requires_grad_(True)

    # 3. 順伝播（Forward Pass）
    # Embedding後のテンソルから計算を開始させる
    hidden_states = embeddings
    for block in model.backbone.layers:
        hidden_states = block(hidden_states)
    output = model.backbone.norm_f(hidden_states)
    
    # 4. ターゲットの指定（例：シーケンスの最後のトークンの出力を対象にする）
    # 出力全体のL2ノルム、あるいは特定の次元をターゲットにします
    target_token_idx = -1 
    target_value = output[0, target_token_idx].norm()

    # 5. 逆伝播（Backward Pass）
    model.zero_grad()
    target_value.backward()

    # 6. 勾配の抽出と加工
    # shape: (1, seq_len, d_model) -> (seq_len,)
    gradients = embeddings.grad.abs().sum(dim=-1).squeeze(0)
    
    # 0-1に正規化（可視化用）
    gradients = (gradients - gradients.min()) / (gradients.max() - gradients.min())
    
    return token_labels, gradients.cpu().detach().numpy()

# --- 可視化コード ---
def plot_sensitivity(tokens, scores):
    plt.figure(figsize=(12, 2))
    sns.heatmap([scores], annot=[tokens], fmt="", cmap="YlGnBu", cbar=False)
    plt.title("Mamba Token Sensitivity (Gradient-based ERF)")
    plt.xlabel("Input Tokens")
    plt.show()

# 使用イメージ
# text = "Mamba is a selective state space model that scales linearly."
# tokens, scores = analyze_mamba_sensitivity(mamba_model, mamba_tokenizer, text)
# plot_sensitivity(tokens, scores)