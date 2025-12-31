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



def evaluate_mlm(model, tokenizer, dataloader, device, num_samples=5):
    model.eval()  # 推論モード
    samples_count = 0
    
    print(f"--- MLM 推論結果 (Top-1 予測) ---")
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device) # MLMの正解ラベル (-100は無視対象)

            # 1. 生徒モデルで予測
            # Note: FeatureDistillationTrainerの中のstudentを直接呼び出します
            outputs = model(input_ids, attention_mask=attention_mask)
            
            # 通常、AutoModelForMaskedLMなどは logits を返します
            # trainer.student が MLM用の出力層を持っている前提です
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

            # 2. [MASK] トークンの場所を特定 (labels が -100 以外の場所)
            mask_indices = (labels != -100).nonzero(as_tuple=True)

            if mask_indices[0].numel() == 0:
                continue

            # 3. 予測結果のデコード
            # 各マスク位置で最も確率の高いIDを取得
            predictions = torch.argmax(logits, dim=-1)

            # サンプルを表示
            for i in range(len(input_ids)):
                if samples_count >= num_samples:
                    return

                # この行にマスクがあるか確認
                row_mask_idx = (mask_indices[0] == i).nonzero(as_tuple=True)[0]
                if row_mask_idx.numel() == 0:
                    continue

                # オリジナルの文章（マスク済み）と予測をデコード
                # labelsを使って元の単語を復元
                original_input = input_ids[i].clone()
                true_labels = labels[i]
                
                # 表示用に[MASK]部分を特定
                masked_positions = (true_labels != -100).nonzero(as_tuple=True)[0]
                
                print(f"\nSample {samples_count + 1}:")
                # マスクされた状態のテキスト
                print(f"Input context: {tokenizer.decode(input_ids[i], skip_special_tokens=True)}")
                
                for pos in masked_positions:
                    true_token = tokenizer.decode(true_labels[pos].item())
                    pred_token = tokenizer.decode(predictions[i, pos].item())
                    print(f"  - [MASK] at pos {pos.item()}: True='{true_token}', Pred='{pred_token}'")
                
                samples_count += 1

# --- 実行 ---
# trainer.student が MLMヘッドを持っている(BertForMaskedLMなど)場合
evaluate_mlm(trainer.student, tokenizer, val_dataloader, device)

# --- 実行 ---
# val_dataloader（学習に使っていないデータ）を使用して確認
demo_student_output(trainer, val_dataloader, device)