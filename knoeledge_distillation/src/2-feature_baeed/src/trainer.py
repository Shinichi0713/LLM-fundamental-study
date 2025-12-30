import torch
from tqdm import tqdm

# デバイスの設定（GPUが使えればGPU、なければCPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデルをデバイスに移動
teacher.to(device)
trainer.to(device) # FeatureDistillationTrainerの中にstudentとregressorが含まれています

# 教師モデルは常にevalモード（重み固定）
teacher.eval()

# 学習用ループの例
def train_one_epoch(trainer, train_loader, optimizer, device):
    trainer.train() # 生徒モデルとregressorを訓練モードに
    total_loss = 0

    # tqdmで進捗を表示
    pbar = tqdm(train_loader, desc="Training")
    
    for batch in pbar:
        # 1. データの準備（DataLoaderから取得）
        # batchが辞書形式（{'input_ids': ..., 'attention_mask': ...}）を想定
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device) if 'attention_mask' in batch else None

        # 2. 勾配の初期化
        optimizer.zero_grad()

        # 3. 順伝播（Forward）
        # FeatureDistillationTrainerのforwardを呼び出し、蒸留損失（MSE）を計算
        loss = trainer(input_ids, attention_mask=attention_mask)

        # 4. 逆伝播（Backward）
        loss.backward()

        # 5. パラメータ更新
        optimizer.step()

        # 統計情報の更新
        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / len(train_loader)
    return avg_loss

# --- 実行例 ---
# 注意: train_loader は事前に作成されている必要があります（Dataset/DataLoader）
# avg_loss = train_one_epoch(trainer, train_loader, optimizer, device)
# print(f"Average Distillation Loss: {avg_loss:.4f}")