import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAutoencoder(nn.Module):
    def __init__(self, d_model, n_features, sparsity_lambda=1e-3):
        super().__init__()
        self.W_enc = nn.Linear(d_model, n_features, bias=False)
        self.W_dec = nn.Linear(n_features, d_model, bias=False)
        self.sparsity_lambda = sparsity_lambda

    def forward(self, x):
        # x: (B, seq_len, d_model)
        a = F.relu(self.W_enc(x))  # (B, seq_len, n_features)
        x_recon = self.W_dec(a)   # (B, seq_len, d_model)
        # L1スパース正則化
        loss_recon = F.mse_loss(x_recon, x)
        loss_sparse = self.sparsity_lambda * a.abs().mean()
        loss = loss_recon + loss_sparse
        return x_recon, a, loss
    

class CrossLayerTranscoder(nn.Module):
    def __init__(self, n_layers, d_model, n_features_per_layer):
        super().__init__()
        # W_dec^{l'→l}: 層l'の特徴 → 層lの再構成
        self.W_dec_list = nn.ModuleList([
            nn.Linear(n_features_per_layer[l_prime], d_model, bias=False)
            for l_prime in range(n_layers)
        ])

    def forward(self, features_list):
        # features_list[l_prime]: (B, seq_len, n_features[l_prime])
        y_hat = 0.0
        for l_prime, a_lprime in enumerate(features_list):
            y_hat += self.W_dec_list[l_prime](a_lprime)
        return y_hat
    
def forward_with_frozen_attention(model, input_ids):
    # 通常のforward
    outputs = model(input_ids, output_hidden_states=True)
    # Attentionパターン・LayerNorm分母を固定（ここでは概念例）
    # 実際には、各層のAttention/LayerNormの出力をdetach()する
    # 例：layer.attention.self.value = layer.attention.self.value.detach()
    return outputs


def compute_attribution_graph(model, sae_list, input_ids, target_logit_idx):
    # 1. 特徴量a_iを取得（各層のSAE活性）
    features_list = []
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        for l, sae in enumerate(sae_list):
            _, a_l, _ = sae(outputs.hidden_states[l])
            features_list.append(a_l)

    # 2. ターゲットlogitに対するヤコビアンを計算
    #    （PyTorchのautograd.gradで、target_logitに対する各特徴の勾配を取得）
    # 3. Attribution: A_{s→t} = a_s * (∂logit_t / ∂a_s)
    #    ここでは概念例として、勾配を重みとして扱う
    attribution_weights = []
    for a_l in features_list:
        a_l.requires_grad_(True)
        # model forward with CLT/local replacement
        # logit_t = model(...)[target_logit_idx]
        # grad = torch.autograd.grad(logit_t, a_l, retain_graph=True)[0]
        # attribution = a_l * grad
        # attribution_weights.append(attribution)
        pass  # 実装はモデル構造に依存

    return attribution_weights

def prune_attribution_graph(attribution_weights, top_k_ratio=0.1):
    # attribution_weights: 各層の(B, seq_len, n_features)テンソル
    all_weights = torch.cat([w.flatten() for w in attribution_weights])
    threshold = torch.quantile(all_weights.abs(), 1.0 - top_k_ratio)
    pruned = [w * (w.abs() >= threshold).float() for w in attribution_weights]
    return pruned


def inhibit_feature(model, sae_list, input_ids, layer_idx, feature_idx, multiplier=-1.0):
    # 1. 通常のforwardで特徴量を取得
    outputs = model(input_ids, output_hidden_states=True)
    x_l = outputs.hidden_states[layer_idx]
    _, a_l, _ = sae_list[layer_idx](x_l)

    # 2. 特定特徴を抑制
    a_l_inhibited = a_l.clone()
    a_l_inhibited[:, :, feature_idx] *= multiplier

    # 3. CLT/local replacementで再構成し、モデルに流す
    # （ここでは概念例）
    # y_hat_l = CLT(a_l_inhibited)
    # outputs_inhibited = model.forward_with_replaced_mlp(y_hat_l)
    # return outputs_inhibited

    