from PIL import Image
import requests
import torch

# ---- 画像 ----
url = "https://images.unsplash.com/photo-1518717758536-85ae29035b6d"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

image_inputs = processor(images=image, return_tensors="pt").to(device)

# ---- テキスト ----
prompt = "Describe this image:"
caption = " A dog looking the master"  # ★ 先頭にスペース重要

full_text = prompt + caption

# ---- tokenize ----
inputs = tokenizer(
    full_text,
    return_tensors="pt"
).to(device)

input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# ---- labels作成 ----
labels = input_ids.clone()

# ★ prompt部分を無視
prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
prompt_len = prompt_ids.shape[1]

labels[:, :prompt_len] = -100


optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-5)

for step in range(100):
    outputs = model(
        pixel_values=image_inputs["pixel_values"],
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )

    loss = outputs.loss

    # ★ NaN防止
    if torch.isnan(loss):
        print(f"step {step} | NaN detected, skip")
        optimizer.zero_grad()
        continue

    loss.backward()

    # ★ 勾配クリッピング（必須）
    torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)

    optimizer.step()
    optimizer.zero_grad()

    if step % 10 == 0:
        print(f"step {step} | loss {loss.item():.4f}")

model.eval()

with.torch.no_grad():
    outputs = model(
            pixel_values=image_inputs["pixel_values"],
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    # logitsから最も確率の高いトークンIDを取り出す
    predicted_token_ids = torch.argmax(outputs.logits, dim=-1) # [1, seq_len]

    # Tokenizerでデコードして文章にする
    predicted_text = tokenizer.decode(predicted_token_ids[0])
    print(f"Predicted Text: {predicted_text}")
