text = "The movie was surprisingly good and very enjoyable."
inputs = tokenizer(text, return_tensors="pt").to(device)

# 勾配
activations = {}

def get_hook(name):
    def hook(module, inp, out):
        out.retain_grad()   # ← これが必須
        activations[name] = out
    return hook

for name, module in model.named_modules():
    if "encoder.layer" in name and "output.dense" in name:
        module.register_forward_hook(get_hook(name))

# 勾配取得
# input_ids は long のまま
inputs["input_ids"] = inputs["input_ids"].long()

# 埋め込み層をバイパスして直接埋め込み重みを掛ける
embedding_weight = model.get_input_embeddings().weight  # (vocab_size, hidden_size)
inputs_embeds = embedding_weight[inputs["input_ids"]]  # long なのでそのまま indices に変換
inputs_embeds = inputs_embeds.detach()  # 計算グラフから切り離してリーフにする
inputs_embeds.requires_grad = True  # ここで requires_grad=True を設定

inputs["inputs_embeds"] = inputs_embeds
inputs.pop("input_ids", None)  # input_ids は使わないなら削除

outputs = model(**inputs)

logits = outputs.logits

target_class = logits.argmax(dim=-1)

loss = logits[0, target_class]
loss.backward()

importance = {}

for name, act in activations.items():
    if act.grad is not None:
        grad = act.grad
        score = (act * grad).abs().mean(dim=1).squeeze()
        importance[name] = score.cpu().detach().numpy()
    else:
        print(f"{name}: grad is None, skipping")

print("=== importance keys ===")
print(list(importance.keys()))

top_neurons = {}

k = 5

for name, score in importance.items():
    idx = score.argsort()[-k:][::-1]
    top_neurons[name] = idx

print("=== top_neurons ===")
print(top_neurons)