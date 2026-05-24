import pandas as pd
from huggingface_hub import list_repo_files, hf_hub_download
from tqdm import tqdm

repo_id = "sabdalja/HalluVerse-M3"
print(f"🎯 Excelファイルを検知しました。ダウンロードと統合を開始します...")

try:
    # 全ファイル一覧（提示していただいたリスト）
    all_files = list_repo_files(repo_id, repo_type="dataset")
    
    # .xlsx ファイルだけを抽出
    excel_files = [f for f in all_files if f.endswith('.xlsx')]
    print(f"📦 見つかったデータファイル（計{len(excel_files)}件）: {excel_files}")
    
    combined_dfs = []
    
    # 各ExcelファイルをダウンロードしてPandasに読み込む
    for filename in tqdm(excel_files, desc="Excelファイルの処理中"):
        # ファイルをローカルにキャッシュ
        local_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
        
        # Excelファイルの読み込み
        df_temp = pd.read_excel(local_path, engine='openpyxl')
        
        # どのファイルから来たデータか分かるように、パスからタスクと言語を自動付与
        # 例: FINAL_DATASET/QA/QA_en.xlsx -> タスク: QA, 言語: en
        path_parts = filename.split('/')
        task = path_parts[1]  # QA または summarization
        lang = path_parts[2].split('_')[1].replace('.xlsx', '')  # ar, en, hi, tr
        
        df_temp['task_category'] = task
        df_temp['language_code'] = lang
        
        combined_dfs.append(df_temp)
        
    # 2. すべてのExcelデータを1つのデータフレームに結合
    if combined_dfs:
        df = pd.concat(combined_dfs, ignore_index=True)
        
        print("\n✨ データセットの復元に100%成功しました！！！")
        print(f"📊 総データ件数 (全言語・全タスク合計): {len(df)} 件")
        print("\n--- 利用可能なカラム一覧 ---")
        print(df.columns.tolist())
        
        # データのサンプルを表示
        print("\n--- データの最初の1件 ---")
        print(df.head(1).to_dict(orient="records"))
    else:
        print("⚠️ 読み込めるExcelファイルがありませんでした。")

except Exception as e:
    print(f"❌ エラーが発生しました: {e}")

# 英語サンプルだけを抽出
df_en = df[df["language_code"] == "tr"].copy()

print(f"英語サンプルの件数: {len(df_en)}")
print(df_en.head())

def predict_hallucination(question, answer, max_new_tokens=10):
    prompt = build_prompt(question, answer)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # プロンプト部分を除去
    response = response[len(prompt):].strip()
    return response


import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# 評価用サンプル数を制限（例：100件）
eval_samples = eval_data.select(range(min(100, len(eval_data))))

pred_labels = []
true_labels = []

for example in eval_samples:
    question = example["question"]
    answer = example["answer"]
    true_label = example["label"]  # 0: no hallucination, 1: hallucination

    pred_text = predict_hallucination(question, answer)
    pred_label = 1 if "yes" in pred_text.lower() else 0

    pred_labels.append(pred_label)
    true_labels.append(true_label)

pred_labels = np.array(pred_labels)
true_labels = np.array(true_labels)

acc = accuracy_score(true_labels, pred_labels)
f1 = f1_score(true_labels, pred_labels)

print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")