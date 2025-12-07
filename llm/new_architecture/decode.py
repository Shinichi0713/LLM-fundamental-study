import torch
import random
from IPython.display import HTML, display

# -----------------------------------------------
# ★ tokenizer, model, device はすでにロード済み
# -----------------------------------------------

# ランダムに MASK を付与
def random_mask_text(text, mask_ratio=0.15):

    encoding = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoding["input_ids"][0]

    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    num_mask = max(1, int(len(tokens) * mask_ratio))

    mask_indices = random.sample(range(len(tokens)), num_mask)

    masked_tokens = tokens.copy()
    for idx in mask_indices:
        masked_tokens[idx] = tokenizer.mask_token

    return tokens, masked_tokens, mask_indices


# MASK部分を予測
def predict_masked_text(masked_tokens):
    input_ids = tokenizer.convert_tokens_to_ids(masked_tokens)
    input_tensor = torch.tensor([input_ids]).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)

    logits = outputs[0]  # tuple の 0 番目が logits

    predicted_tokens = []
    for i, token in enumerate(masked_tokens):
        if token == tokenizer.mask_token:
            pred_id = logits[0, i].argmax().item()
            predicted_tokens.append(tokenizer.convert_ids_to_tokens(pred_id))
        else:
            predicted_tokens.append(token)

    return predicted_tokens


# HTMLで可視化
def visualize(tokens, masked_tokens, predicted_tokens, mask_indices):

    html = ""

    for i, (orig, masked, pred) in enumerate(zip(tokens, masked_tokens, predicted_tokens)):

        if i in mask_indices:
            # 正解 → 緑 / 不一致 → 赤
            color = "lightgreen" if orig == pred else "salmon"
            html += f'<span style="background-color:{color}; padding:2px; margin:2px;">'
            html += f'原:{tokenizer.convert_tokens_to_string([orig])} → 予:{tokenizer.convert_tokens_to_string([pred])}'
            html += '</span> '
        else:
            html += tokenizer.convert_tokens_to_string([orig]) + " "

    display(HTML(html))


# メイン処理
def run_mask_prediction():
    text = input("文章を入力してください：\n\n")

    tokens, masked_tokens, mask_indices = random_mask_text(text)
    predicted_tokens = predict_masked_text(masked_tokens)

    print("\n【元の文章】")
    print(tokenizer.convert_tokens_to_string(tokens))

    print("\n【MASK を含む文章】")
    print(tokenizer.convert_tokens_to_string(masked_tokens))

    print("\n【予測後の文章】")
    print(tokenizer.convert_tokens_to_string(predicted_tokens))

    print("\n【色付き可視化】")
    visualize(tokens, masked_tokens, predicted_tokens, mask_indices)


run_mask_prediction()
