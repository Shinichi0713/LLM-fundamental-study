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

import os

# 1. 保存用ディレクトリの作成
save_dir = "saved_models/distilled_student"
os.makedirs(save_dir, exist_ok=True)

# 2. 保存用パスの設定
student_path = os.path.join(save_dir, "student_model.pth")
regressor_path = os.path.join(save_dir, "regressor.pth")
full_trainer_path = os.path.join(save_dir, "full_trainer_checkpoint.pth")

# --- パターンA: 個別に保存（おすすめ） ---
# 生徒モデルのみを別のタスク（分類や検索）に転用しやすくなります
torch.save(trainer.student.state_dict(), student_path)
torch.save(trainer.regressor.state_dict(), regressor_path)

# --- パターンB: 学習の「中断・再開」用に保存 ---
# optimizerやepoch数も含めて保存することで、後から学習を再開できます
# checkpoint = {
#     'epoch': num_epochs,
#     'model_state_dict': trainer.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
#     'loss': avg_loss,
# }
# torch.save(checkpoint, full_trainer_path)

print(f"モデルの保存が完了しました：\n- {student_path}\n- {regressor_path}")