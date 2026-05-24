import json
import os

# パス設定
amber_dir = "/content/AMBER"
inference_data_path = "/content/inference_data.json"
annotation_path = os.path.join(amber_dir, "data", "annotations.json")

# 推論結果とアノテーションを読み込み
with open(inference_data_path, "r", encoding="utf-8") as f:
    inference_data = json.load(f)
with open(annotation_path, "r", encoding="utf-8") as f:
    annotations = json.load(f)

# アノテーションを id -> エントリの辞書に変換
annotation_dict = {item["id"]: item for item in annotations}

# 推論結果を id -> エントリの辞書に変換
inference_dict = {item["id"]: item for item in inference_data}

# 指標の初期化
metrics = {
    "qa_correct_num": 0,
    "qa_correct_score": 0,
    "qa_no_num": 0,
    "qa_no_score": 0,
    "qa_ans_no_num": 0,
    "qa_ans_no_score": 0,
    "as_qa_correct_num": 0,
    "as_qa_correct_score": 0,
    "as_qa_no_num": 0,
    "as_qa_no_score": 0,
    "as_qa_ans_no_num": 0,
    "as_qa_ans_no_score": 0,
    "an_qa_correct_num": 0,
    "an_qa_correct_score": 0,
    "an_qa_no_num": 0,
    "an_qa_no_score": 0,
    "an_qa_ans_no_num": 0,
    "an_qa_ans_no_score": 0,
    "aa_qa_correct_num": 0,
    "aa_qa_correct_score": 0,
    "aa_qa_no_num": 0,
    "aa_qa_no_score": 0,
    "aa_qa_ans_no_num": 0,
    "aa_qa_ans_no_score": 0,
    "ha_qa_correct_num": 0,
    "ha_qa_correct_score": 0,
    "ha_qa_no_num": 0,
    "ha_qa_no_score": 0,
    "ha_qa_ans_no_num": 0,
    "ha_qa_ans_no_score": 0,
    "asso_qa_correct_num": 0,
    "asso_qa_correct_score": 0,
    "asso_qa_no_num": 0,
    "asso_qa_no_score": 0,
    "asso_qa_ans_no_num": 0,
    "asso_qa_ans_no_score": 0,
}

# 各サンプルを評価
for inf in inference_data:
    qid = inf["id"]
    ann = annotation_dict.get(qid)

    if ann is None:
        continue

    truth = ann["truth"]
    response = inf["response"]

    # 判別タスクのみ対象
    if not ann["type"].startswith("discriminative"):
        continue

    # 全体
    metrics["qa_correct_num"] += 1
    if truth == "yes":
        if response == "Yes":
            metrics["qa_correct_score"] += 1
    else:
        metrics["qa_no_num"] += 1
        if response == "No":
            metrics["qa_correct_score"] += 1
            metrics["qa_no_score"] += 1

    if response == "No":
        metrics["qa_ans_no_num"] += 1
        if truth == "no":
            metrics["qa_ans_no_score"] += 1

    # タスク種別ごと
    task_type = ann["type"]
    if task_type == "discriminative-attribute-state":
        metrics["as_qa_correct_num"] += 1
        if truth == "yes":
            if response == "Yes":
                metrics["as_qa_correct_score"] += 1
        else:
            metrics["as_qa_no_num"] += 1
            if response == "No":
                metrics["as_qa_correct_score"] += 1
                metrics["as_qa_no_score"] += 1
        if response == "No":
            metrics["as_qa_ans_no_num"] += 1
            if truth == "no":
                metrics["as_qa_ans_no_score"] += 1

    elif task_type == "discriminative-attribute-number":
        metrics["an_qa_correct_num"] += 1
        if truth == "yes":
            if response == "Yes":
                metrics["an_qa_correct_score"] += 1
        else:
            metrics["an_qa_no_num"] += 1
            if response == "No":
                metrics["an_qa_correct_score"] += 1
                metrics["an_qa_no_score"] += 1
        if response == "No":
            metrics["an_qa_ans_no_num"] += 1
            if truth == "no":
                metrics["an_qa_ans_no_score"] += 1

    elif task_type == "discriminative-attribute-action":
        metrics["aa_qa_correct_num"] += 1
        if truth == "yes":
            if response == "Yes":
                metrics["aa_qa_correct_score"] += 1
        else:
            metrics["aa_qa_no_num"] += 1
            if response == "No":
                metrics["aa_qa_correct_score"] += 1
                metrics["aa_qa_no_score"] += 1
        if response == "No":
            metrics["aa_qa_ans_no_num"] += 1
            if truth == "no":
                metrics["aa_qa_ans_no_score"] += 1

    elif task_type == "discriminative-hallucination":
        metrics["ha_qa_correct_num"] += 1
        if truth == "yes":
            if response == "Yes":
                metrics["ha_qa_correct_score"] += 1
        else:
            metrics["ha_qa_no_num"] += 1
            if response == "No":
                metrics["ha_qa_correct_score"] += 1
                metrics["ha_qa_no_score"] += 1
        if response == "No":
            metrics["ha_qa_ans_no_num"] += 1
            if truth == "no":
                metrics["ha_qa_ans_no_score"] += 1

    else:  # relation など
        metrics["asso_qa_correct_num"] += 1
        if truth == "yes":
            if response == "Yes":
                metrics["asso_qa_correct_score"] += 1
        else:
            metrics["asso_qa_no_num"] += 1
            if response == "No":
                metrics["asso_qa_correct_score"] += 1
                metrics["asso_qa_no_score"] += 1
        if response == "No":
            metrics["asso_qa_ans_no_num"] += 1
            if truth == "no":
                metrics["asso_qa_ans_no_score"] += 1

# 指標計算
def safe_divide(a, b):
    return a / b if b > 0 else 0.0

# 全体
Accuracy = round(safe_divide(metrics["qa_correct_score"], metrics["qa_correct_num"]) * 100, 1)
Precision = round(safe_divide(metrics["qa_ans_no_score"], metrics["qa_ans_no_num"]) * 100, 1)
Recall = round(safe_divide(metrics["qa_no_score"], metrics["qa_no_num"]) * 100, 1)
F1 = round(2 * (Precision/100) * (Recall/100) / ((Precision/100) + (Recall/100) + 0.0001) * 100, 1)

print("=== Discriminative Task Summary ===")
print(f"Accuracy:\t{Accuracy}%")
print(f"Precision:\t{Precision}%")
print(f"Recall:\t\t{Recall}%")
print(f"F1:\t\t{F1}%")
print()

# Existence (hallucination)
if metrics["ha_qa_correct_num"] > 0:
    hallucination_Accuracy = round(safe_divide(metrics["ha_qa_correct_score"], metrics["ha_qa_correct_num"]) * 100, 1)
    hallucination_Precision = round(safe_divide(metrics["ha_qa_ans_no_score"], metrics["ha_qa_ans_no_num"]) * 100, 1)
    hallucination_Recall = round(safe_divide(metrics["ha_qa_no_score"], metrics["ha_qa_no_num"]) * 100, 1)
    hallucination_F1 = round(2 * (hallucination_Precision/100) * (hallucination_Recall/100) / ((hallucination_Precision/100) + (hallucination_Recall/100) + 0.001) * 100, 1)

    print("=== Existence (Hallucination) ===")
    print(f"Accuracy:\t{hallucination_Accuracy}%")
    print(f"Precision:\t{hallucination_Precision}%")
    print(f"Recall:\t\t{hallucination_Recall}%")
    print(f"F1:\t\t{hallucination_F1}%")
    print()

# Attribute (state + number + action)
if metrics["as_qa_correct_num"] + metrics["an_qa_correct_num"] + metrics["aa_qa_correct_num"] > 0:
    attr_Accuracy = round(safe_divide(
        metrics["as_qa_correct_score"] + metrics["an_qa_correct_score"] + metrics["aa_qa_correct_score"],
        metrics["as_qa_correct_num"] + metrics["an_qa_correct_num"] + metrics["aa_qa_correct_num"]
    ) * 100, 1)
    attr_Precision = round(safe_divide(
        metrics["as_qa_ans_no_score"] + metrics["an_qa_ans_no_score"] + metrics["aa_qa_ans_no_score"],
        metrics["as_qa_ans_no_num"] + metrics["an_qa_ans_no_num"] + metrics["aa_qa_ans_no_num"]
    ) * 100, 1)
    attr_Recall = round(safe_divide(
        metrics["as_qa_no_score"] + metrics["an_qa_no_score"] + metrics["aa_qa_no_score"],
        metrics["as_qa_no_num"] + metrics["an_qa_no_num"] + metrics["aa_qa_no_num"]
    ) * 100, 1)
    attr_F1 = round(2 * (attr_Precision/100) * (attr_Recall/100) / ((attr_Precision/100) + (attr_Recall/100) + 0.0001) * 100, 1)

    print("=== Attribute (State + Number + Action) ===")
    print(f"Accuracy:\t{attr_Accuracy}%")
    print(f"Precision:\t{attr_Precision}%")
    print(f"Recall:\t\t{attr_Recall}%")
    print(f"F1:\t\t{attr_F1}%")
    print()

# Relation
if metrics["asso_qa_correct_num"] > 0:
    relation_Accuracy = round(safe_divide(metrics["asso_qa_correct_score"], metrics["asso_qa_correct_num"]) * 100, 1)
    relation_Precision = round(safe_divide(metrics["asso_qa_ans_no_score"], metrics["asso_qa_ans_no_num"]) * 100, 1)
    relation_Recall = round(safe_divide(metrics["asso_qa_no_score"], metrics["asso_qa_no_num"]) * 100, 1)
    relation_F1 = round(2 * (relation_Precision/100) * (relation_Recall/100) / ((relation_Precision/100) + (relation_Recall/100) + 0.0001) * 100, 1)

    print("=== Relation ===")
    print(f"Accuracy:\t{relation_Accuracy}%")
    print(f"Precision:\t{relation_Precision}%")
    print(f"Recall:\t\t{relation_Recall}%")
    print(f"F1:\t\t{relation_F1}%")

    