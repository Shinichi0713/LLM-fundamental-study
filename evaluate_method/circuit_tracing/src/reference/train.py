from transformers import BertModel, BertTokenizer

model = BertModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def get_mlp_outputs(input_ids):
    # BERTの全層の隠れ状態を取得
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # (n_layers+1, B, seq_len, d_model)
    # MLP出力は、各層の隠れ状態の「残差加算前」に相当するが、
    # 実装上は、各層の出力（LayerNorm前）をMLP出力の近似として使うことが多い
    mlp_outputs = []
    for l in range(len(hidden_states) - 1):
        # 層lの出力（LayerNorm前）をMLP出力の近似として使用
        # 実際には、BERTの実装に合わせて調整が必要
        mlp_out = hidden_states[l]  # 近似
        mlp_outputs.append(mlp_out)
    return mlp_outputs

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
        loss_recon = F.mse_loss(x_recon, x)
        loss_sparse = self.sparsity_lambda * a.abs().mean()
        loss = loss_recon + loss_sparse
        return x_recon, a, loss


d_model = 768
n_features_per_layer = 4096  # 例
n_layers = 12
sae_list = [SparseAutoencoder(d_model, n_features_per_layer) for _ in range(n_layers)]
optimizers = [torch.optim.Adam(sae.parameters(), lr=1e-3) for sae in sae_list]

# BERTは勾配不要（重み固定）
model.requires_grad_(False)


class CrossLayerTranscoder(nn.Module):
    def __init__(self, n_layers, d_model, n_features_per_layer):
        super().__init__()
        self.W_dec_list = nn.ModuleList([
            nn.Linear(n_features_per_layer, d_model, bias=False)
            for _ in range(n_layers)
        ])

    def forward(self, features_list):
        # features_list[l_prime]: (B, seq_len, n_features)
        y_hat = 0.0
        for l_prime, a_lprime in enumerate(features_list):
            y_hat += self.W_dec_list[l_prime](a_lprime)
        return y_hat

clt_list = [CrossLayerTranscoder(n_layers, d_model, n_features_per_layer) for _ in range(n_layers)]
clt_optimizers = [torch.optim.Adam(clt.parameters(), lr=1e-3) for clt in clt_list]

num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for epoch in range(num_epochs):
    # エポックごとのロスを記録
    epoch_losses = []

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        mlp_outputs = get_mlp_outputs(input_ids)  # (n_layers, B, seq_len, d_model)

        # SAEで特徴量を取得（推論モード）
        features_list = []
        with torch.no_grad():
            for l in range(n_layers):
                _, a_l, _ = sae_list[l](mlp_outputs[l])
                features_list.append(a_l)

        # 各ターゲット層lに対してCLTを学習
        batch_loss = 0.0
        for l in range(n_layers):
            clt = clt_list[l]
            optimizer = clt_optimizers[l]
            optimizer.zero_grad()

            y_l = mlp_outputs[l]  # 層lのMLP出力（教師信号）
            y_hat_l = clt(features_list[:l+1])  # 下位層のみ使用（l' <= l）

            loss = F.mse_loss(y_hat_l, y_l)
            loss.backward()
            optimizer.step()

            batch_loss += loss.item()

        # バッチごとの平均ロスを記録
        avg_batch_loss = batch_loss / n_layers
        epoch_losses.append(avg_batch_loss)

        # 進捗表示（例：100バッチごと）
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Avg Loss: {avg_batch_loss:.6f}")

    # エポック終了時の平均ロス
    epoch_avg_loss = sum(epoch_losses) / len(epoch_losses)
    print(f"Epoch {epoch+1}/{num_epochs} finished. Average CLT Loss: {epoch_avg_loss:.6f}")

def local_replacement_forward(model, sae_list, clt_list, input_ids):
    # Attention/LayerNormを固定したforward（概念例）
    # 実際には、BERTの各層をラップして実装
    outputs = model(input_ids, output_hidden_states=True)
    hidden_states = outputs.hidden_states

    # 各層のMLP出力をCLTで置き換え
    replaced_hidden_states = []
    features_list = []
    for l in range(len(hidden_states) - 1):
        # SAEで特徴量取得
        _, a_l, _ = sae_list[l](hidden_states[l])
        features_list.append(a_l)
        # CLTで再構成
        y_hat_l = clt_list[l](features_list[:l+1])
        replaced_hidden_states.append(y_hat_l)

    return replaced_hidden_states

