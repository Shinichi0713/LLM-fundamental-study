import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import r2_score

def mse(x, y):
    return F.mse_loss(x, y).item()

def correlation(x, y):
    # x, y: (B, ...) → フラット化して相関係数
    x_flat = x.detach().flatten().cpu().numpy()
    y_flat = y.detach().flatten().cpu().numpy()
    return np.corrcoef(x_flat, y_flat)[0, 1]

def sparsity_metrics(a):
    # a: (B, seq_len, n_features)
    a_flat = a.detach().flatten().cpu().numpy()
    l1_norm = np.abs(a_flat).mean()
    non_zero_ratio = (a_flat != 0).mean()
    return l1_norm, non_zero_ratio

def evaluate_sae(sae_list, model, dataloader, device, n_layers=12):
    model.eval()
    mse_list = []
    corr_list = []
    l1_list = []
    nz_list = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            mlp_outputs = get_mlp_outputs(input_ids)  # (n_layers, B, seq_len, d_model)

            for l in range(n_layers):
                x_l = mlp_outputs[l]
                x_recon, a, _ = sae_list[l](x_l)

                # 再構成精度
                mse_val = mse(x_recon, x_l)
                corr_val = correlation(x_recon, x_l)
                mse_list.append(mse_val)
                corr_list.append(corr_val)

                # スパース性
                l1, nz = sparsity_metrics(a)
                l1_list.append(l1)
                nz_list.append(nz)

    # 層ごとの平均を出力
    mse_arr = np.array(mse_list).reshape(-1, n_layers).mean(axis=0)
    corr_arr = np.array(corr_list).reshape(-1, n_layers).mean(axis=0)
    l1_arr = np.array(l1_list).reshape(-1, n_layers).mean(axis=0)
    nz_arr = np.array(nz_list).reshape(-1, n_layers).mean(axis=0)

    print("=== SAE Evaluation ===")
    for l in range(n_layers):
        print(f"Layer {l}: MSE={mse_arr[l]:.6f}, Corr={corr_arr[l]:.4f}, L1={l1_arr[l]:.4f}, NonZero={nz_arr[l]:.4f}")
    

if name == "__main__":
    # SAEとモデルのロード（例）
    sae_list = [SparseAutoencoder(d_model=768, n_features=4096).to(device) for _ in range(12)]
    model = load_bert_model().to(device)
    dataloader = get_dataloader()

    evaluate_sae(sae_list, model, dataloader, device)
