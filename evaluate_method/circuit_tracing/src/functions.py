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

def evaluate_clt(clt_list, sae_list, model, dataloader, device, n_layers=12):
    model.eval()
    mse_list = []
    corr_list = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            mlp_outputs = get_mlp_outputs(input_ids)

            # SAEで特徴量取得
            features_list = []
            for l in range(n_layers):
                _, a_l, _ = sae_list[l](mlp_outputs[l])
                features_list.append(a_l)

            # CLT再構成
            for l in range(n_layers):
                y_l = mlp_outputs[l]
                y_hat_l = clt_list[l](features_list[:l+1])

                mse_val = mse(y_hat_l, y_l)
                corr_val = correlation(y_hat_l, y_l)
                mse_list.append(mse_val)
                corr_list.append(corr_val)

    mse_arr = np.array(mse_list).reshape(-1, n_layers).mean(axis=0)
    corr_arr = np.array(corr_list).reshape(-1, n_layers).mean(axis=0)

    print("=== CLT Evaluation ===")
    for l in range(n_layers):
        print(f"Layer {l}: MSE={mse_arr[l]:.6f}, Corr={corr_arr[l]:.4f}")

def evaluate_clt(clt_list, sae_list, model, dataloader, device, n_layers=12):
    model.eval()
    mse_list = []
    corr_list = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            mlp_outputs = get_mlp_outputs(input_ids)

            # SAEで特徴量取得
            features_list = []
            for l in range(n_layers):
                _, a_l, _ = sae_list[l](mlp_outputs[l])
                features_list.append(a_l)

            # CLT再構成
            for l in range(n_layers):
                y_l = mlp_outputs[l]
                y_hat_l = clt_list[l](features_list[:l+1])

                mse_val = mse(y_hat_l, y_l)
                corr_val = correlation(y_hat_l, y_l)
                mse_list.append(mse_val)
                corr_list.append(corr_val)

    mse_arr = np.array(mse_list).reshape(-1, n_layers).mean(axis=0)
    corr_arr = np.array(corr_list).reshape(-1, n_layers).mean(axis=0)

    print("=== CLT Evaluation ===")
    for l in range(n_layers):
        print(f"Layer {l}: MSE={mse_arr[l]:.6f}, Corr={corr_arr[l]:.4f}")

def extract_important_features(sae_list, model, input_ids, target_logit_idx, top_k=10):
    """
    簡易版：ターゲットlogitに対する勾配が大きい特徴を抽出
    """
    model.eval()
    input_ids = input_ids.to(device)
    input_ids.requires_grad_(True)

    # BERTの出力（分類ヘッドがある場合を想定）
    # 例：modelは分類タスク用に微調整済みと仮定
    outputs = model(input_ids)
    logits = outputs.logits  # (B, num_classes)
    target_logit = logits[0, target_logit_idx]

    # 特徴量と勾配を取得
    features_grads = []
    for l, sae in enumerate(sae_list):
        x_l = model.get_hidden_state(l, input_ids)  # 実装に応じて調整
        x_recon, a, _ = sae(x_l)
        # 勾配計算
        grad = torch.autograd.grad(target_logit, a, retain_graph=True)[0]
        # 特徴ごとの平均勾配（絶対値）
        importance = (a * grad).abs().mean(dim=(0, 1))  # (n_features,)
        features_grads.append((l, importance))

    # 上位k特徴を抽出
    all_importances = []
    for l, imp in features_grads:
        for f_idx in range(imp.shape[0]):
            all_importances.append((l, f_idx, imp[f_idx].item()))
    all_importances.sort(key=lambda x: x[2], reverse=True)
    top_features = all_importances[:top_k]

    print("=== Top Important Features ===")
    for l, f_idx, score in top_features:
        print(f"Layer {l}, Feature {f_idx}: Importance={score:.6f}")

    return top_features


def feature_inhibition_test(sae_list, clt_list, model, input_ids, target_logit_idx, layer_idx, feature_idx, multiplier=-1.0):
    """
    特定特徴を抑制（multiplier倍）し、ターゲットlogitの変化を観察
    """
    model.eval()
    input_ids = input_ids.to(device)

    # 元のlogit
    with torch.no_grad():
        outputs_orig = model(input_ids)
        logit_orig = outputs_orig.logits[0, target_logit_idx].item()

    # 特徴抑制後のlogit（Local Replacement Model経由）
    # ここでは簡易版として、SAE/CLTで再構成した特徴を流す
    with torch.no_grad():
        mlp_outputs = get_mlp_outputs(input_ids)
        features_list = []
        for l in range(len(sae_list)):
            _, a_l, _ = sae_list[l](mlp_outputs[l])
            features_list.append(a_l)

        # 特定特徴を抑制
        a_inhibited = features_list[layer_idx].clone()
        a_inhibited[:, :, feature_idx] *= multiplier
        features_list[layer_idx] = a_inhibited

        # CLTで再構成し、モデルに流す（概念例）
        # 実際には、Local Replacement Modelのforwardを実装する必要あり
        # ここでは簡易に、最後の層の再構成出力を分類ヘッドに流す
        y_hat_last = clt_list[-1](features_list)
        # model.classifier(y_hat_last) などでlogitを計算

    # 簡易版：抑制後のlogitを計算（実装に応じて調整）
    logit_inhibited = ...  # 実装依存

    change = logit_inhibited - logit_orig
    print(f"Layer {layer_idx}, Feature {feature_idx}: Logit change = {change:.4f}")

    return change

if name == "__main__":
    # SAEとモデルのロード（例）
    sae_list = [SparseAutoencoder(d_model=768, n_features=4096).to(device) for _ in range(12)]
    model = load_bert_model().to(device)
    dataloader = get_dataloader()

    evaluate_sae(sae_list, model, dataloader, device)
    evaluate_clt(clt_list, sae_list, model, dataloader, device)
    evaluate_clt(clt_list, sae_list, model, train_dataloader, device)

    # 例：ParisのLOCラベルlogitをターゲット
    target_logit_idx = label2id["LOC"]  # 事前に定義
    top_features = extract_important_features(sae_list, model, input_ids, target_logit_idx, top_k=10)
    # 重要特徴のうち1つを選んで抑制
    l, f_idx, _ = top_features[0]
    change = feature_inhibition_test(sae_list, clt_list, model, input_ids, target_logit_idx, l, f_idx, multiplier=-1.0)
