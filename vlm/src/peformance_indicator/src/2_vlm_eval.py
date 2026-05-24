import json
import os

# AMBER の data ディレクトリを指定
amber_data_dir = "AMBER/data"

# 識別タスク用の JSON を読み込み（ファイル名は README を参照）
with open(os.path.join(amber_data_dir, "query_discriminative.json"), "r") as f:
    queries = json.load(f)

print(f"識別タスクの質問数: {len(queries)}")
print("1件目の例:")
print(json.dumps(queries[0], indent=2, ensure_ascii=False))



import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm

def download_image(url):
    """URLから画像をダウンロードしてPIL Imageに変換"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        return img
    except Exception as e:
        print(f"画像ダウンロード失敗: {url}, エラー: {e}")
        return None

def vlm_predict(image, question, max_new_tokens=10):
    """VLMに画像と質問を入力し、回答を生成"""
    if image is None:
        return ""

    # プロンプトを組み立て（Yes/No 質問なら "Answer with yes or no." を追加）
    prompt = f"<|im_start|>user\n<|image_1|>\n{question} Answer with yes or no.<|im_end|>\n<|im_start|>assistant\n"

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id
        )

    response = processor.decode(outputs[0], skip_special_tokens=True)
    # プロンプト部分を除去
    response = response[len(prompt):].strip()
    return response

# 評価用にサンプル数を制限（Colab の時間制限を考慮）
eval_queries = queries[:20]  # 20件だけ評価

pred_responses = []

for q in tqdm(eval_queries):
    img_url = q.get("image_url")
    question = q["question"]

    img = download_image(img_url) if img_url else None
    pred = vlm_predict(img, question)
    pred_responses.append(pred)

# 結果を確認
for i, (q, pred) in enumerate(zip(eval_queries, pred_responses)):
    print(f"ID: {q['id']}")
    print(f"Question: {q['question']}")
    print(f"Ground Truth: {q['answer']}")
    print(f"Model Response: {pred}")
    print("-" * 40)


import numpy as np
from sklearn.metrics import accuracy_score

def normalize_answer(ans):
    """回答を小文字に変換し、yes/no に正規化"""
    ans = ans.lower().strip()
    if "yes" in ans:
        return "yes"
    elif "no" in ans:
        return "no"
    else:
        return ans  # そのまま（評価対象外として扱うことも可能）

true_labels = [q["answer"] for q in eval_queries]
pred_labels = [normalize_answer(pred) for pred in pred_responses]

# yes/no に正規化できたサンプルだけを評価
eval_mask = [p in ["yes", "no"] for p in pred_labels]
filtered_true = [t for t, m in zip(true_labels, eval_mask) if m]
filtered_pred = [p for p, m in zip(pred_labels, eval_mask) if m]

if len(filtered_true) > 0:
    acc = accuracy_score(filtered_true, filtered_pred)
    print(f"評価サンプル数: {len(filtered_true)}")
    print(f"Accuracy: {acc:.4f}")
else:
    print("yes/no 形式の回答が得られなかったため、Accuracy を計算できません。")

