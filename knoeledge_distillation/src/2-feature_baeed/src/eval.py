import matplotlib.pyplot as plt
import torch

def demo_student_output(trainer, dataloader, device, num_samples=1):
    # モデルを評価モードに
    trainer.eval()
    teacher.eval()
    
    # データローダーから1バッチだけ取得
    batch = next(iter(dataloader))
    input_ids = batch['input_ids'][:num_samples].to(device)
    attention_mask = batch['attention_mask'][:num_samples].to(device)

    with torch.no_grad():
        # 1. 教師の出力を取得 (Target)
        teacher_hidden = teacher(input_ids, attention_mask=attention_mask).hidden_states[-1]
        
        # 2. 生徒の出力を取得し、次元変換 (Prediction)
        student_outputs = trainer.student(input_ids, attention_mask=attention_mask)
        student_hidden = student_outputs.hidden_states[-1]
        projected_student_hidden = trainer.regressor(student_hidden)

    # データの形状確認 [Batch, SeqLen, Dim] -> 今回は最初の1トークン目のベクトルを比較
    # ベクトルの最初の50要素を抽出して比較
    t_vector = teacher_hidden[0, 0, :50].cpu().numpy()
    s_vector = projected_student_hidden[0, 0, :50].cpu().numpy()

    # 3. 視覚化
    plt.figure(figsize=(12, 5))
    plt.plot(t_vector, label='Teacher (Original)', marker='o', linestyle='--')
    plt.plot(s_vector, label='Student (Projected)', marker='x', linestyle='-')
    plt.title("Feature Matching: Teacher vs Student (First 50 dimensions)")
    plt.xlabel("Dimension Index")
    plt.ylabel("Activation Value")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 4. 数値で誤差を表示
    mse = torch.nn.functional.mse_loss(projected_student_hidden, teacher_hidden)
    print(f"Cosine Similarity: {torch.nn.functional.cosine_similarity(projected_student_hidden, teacher_hidden).mean():.4f}")
    print(f"Mean Squared Error: {mse.item():.6f}")

# --- 実行 ---
# val_dataloader（学習に使っていないデータ）を使用して確認
demo_student_output(trainer, val_dataloader, device)