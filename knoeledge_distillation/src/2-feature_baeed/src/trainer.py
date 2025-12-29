import torch
from torch.optim import AdamW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- モデルの準備 (前述のクラスを利用) ---
# teacher, student は定義済みと仮定
trainer = FeatureDistillationTrainer(
    teacher_model=teacher, 
    student_model=student, 
    teacher_dim=768, 
    student_dim=256
).to(device)

# 生徒モデルの全パラメータと、次元変換用のregressorを最適化対象にする
optimizer = AdamW(
    list(trainer.student.parameters()) + list(trainer.regressor.parameters()), 
    lr=5e-5
)

# --- 学習ループ ---
trainer.student.train()
num_epochs = 3

for epoch in range(num_epochs):
    total_loss = 0
    for step, batch in enumerate(train_dataloader):
        # バッチをGPU/CPUへ転送
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        # 1. 順伝播と蒸留損失の計算
        loss = trainer(input_ids, attention_mask=attention_mask)
        
        # 2. 逆伝播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if step % 50 == 0:
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_dataloader)
    print(f"Average Loss for Epoch {epoch}: {avg_loss:.4f}")