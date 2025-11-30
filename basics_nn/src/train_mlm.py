import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.cuda.amp import GradScaler, autocast # 混合精度学習用

# --- モデルとオプティマイザの準備 ---
model.to(device)

# PyTorch 2.0以上の場合はコンパイル機能を使う（非常に高速化します）
try:
    model = torch.compile(model)
    print("Model compiled with torch.compile!")
except Exception as e:
    print(f"Skipping torch.compile: {e}")

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# 混合精度学習のためのスケーラー
scaler = GradScaler()

# --- データローダーの最適化 ---
# pin_memory=True: CPUメモリからGPUメモリへの転送を高速化
# num_workers=2: データの読み込みを並列化してGPUの待ち時間を減らす
train_loader = DataLoader(
    dataset, # ここにはご自身のdataset変数を指定してください
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=True
)

# --- 学習ループ ---
model.train()
global_step = 0

for epoch in range(EPOCHS):
    total_loss = 0
    optimizer.zero_grad() # ループの最初に勾配をリセット
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for step, batch in enumerate(progress_bar):
        # 1. バッチをGPUへ転送 (non_blocking=Trueで非同期転送)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        global_mask = None

        # 2. 混合精度学習 (Autocast)
        # 前方計算を float16 (または bfloat16) で行い、メモリ削減と高速化
        with autocast(enabled=True): # device_type='cuda' は自動判別されることが多いですが明示してもOK
            logits, loss = model(input_ids, global_mask=global_mask, labels=labels)
            
            # 3. 勾配アキュムレーション
            # 損失をステップ数で割る（合計したときに元のバッチサイズ相当の大きさにするため）
            loss = loss / ACCUMULATION_STEPS

        # 4. スケーリング付き逆伝播
        # float16での勾配消失を防ぐためにスケーラーを使用
        scaler.scale(loss).backward()

        # 5. パラメータ更新 (指定ステップごとに実行)
        if (step + 1) % ACCUMULATION_STEPS == 0:
            # 勾配のクリッピング (爆発防止) - オプションだが推奨
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # オプティマイザのステップ実行
            scaler.step(optimizer)
            scaler.update()
            
            # 勾配初期化 (set_to_none=True はゼロ埋めより高速)
            optimizer.zero_grad(set_to_none=True)
            
            global_step += 1

        # ロスの記録（表示用に戻すために掛け算する）
        current_loss = loss.item() * ACCUMULATION_STEPS
        total_loss += current_loss
        progress_bar.set_postfix({"loss": f"{current_loss:.4f}"})

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")

    # 定期保存 (例: 1エポックごと)
    torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")

print("Training finished!")