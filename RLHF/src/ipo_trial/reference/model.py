import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================
# 1. モデルとトークナイザの準備
# ============================
model_name = "mistralai/Mistral-7B-v0.1"  # 例：軽量モデルを想定
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # パディングトークンの設定

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

# ============================
# 2. ログ確率取得関数（平均を取る）
# ============================
def get_logp_mean(model, tokenizer, prompt, response):
    """
    プロンプト＋応答のテキストから、応答部分のログ確率の平均を計算する。
    """
    # プロンプト＋応答を結合
    text = prompt + response
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", None)

    # ログ確率を計算
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # (1, seq_len, vocab_size)
        log_probs = torch.log_softmax(logits, dim=-1)

    # プロンプト部分の長さを取得
    prompt_inputs = tokenizer(prompt, return_tensors="pt")
    prompt_len = len(prompt_inputs["input_ids"][0])

    # 応答部分のログ確率を抽出
    # log_probs[0, prompt_len-1:-1, :] が応答部分に対応（シフトに注意）
    response_log_probs = log_probs[0, prompt_len-1:-1, :]

    # 実際に生成されたトークンのログ確率を取得
    response_ids = input_ids[0, prompt_len:]
    selected_log_probs = response_log_probs[torch.arange(len(response_ids)), response_ids]

    # 平均を取る（Hugging Faceブログで強調されているポイント）
    return selected_log_probs.mean()

# ============================
# 3. IPO損失の計算関数
# ============================
def ipo_loss(model, tokenizer, prompts, chosens, rejecteds, beta=0.01):
    """
    IPO損失を計算する。
    - prompts: list[str]
    - chosens: list[str]
    - rejecteds: list[str]
    - beta: IPOの正則化パラメータ
    """
    c = 1.0 / (2.0 * beta)  # ターゲットマージン
    losses = []

    for prompt, chosen, rejected in zip(prompts, chosens, rejecteds):
        logp_w = get_logp_mean(model, tokenizer, prompt, chosen)
        logp_l = get_logp_mean(model, tokenizer, prompt, rejected)
        delta_r = logp_w - logp_l  # 暗黙の報酬マージン

        # IPO損失: (delta_r - c)^2
        loss_i = (delta_r - c) ** 2
        losses.append(loss_i)

    # バッチ平均
    return torch.stack(losses).mean()

# ============================
# 4. 簡易データセットの定義
# ============================
class IPODataset(Dataset):
    def __init__(self, data):
        self.data = data  # list of dicts: {"prompt": ..., "chosen": ..., "rejected": ...}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# ダミーデータ（実際にはUltraFeedback等のペアデータを使う）
dummy_data = [
    {
        "prompt": "Explain the concept of IPO.",
        "chosen": "IPO stands for Identity Preference Optimization...",
        "rejected": "IPO is a financial term meaning Initial Public Offering..."
    },
    {
        "prompt": "What is RLHF?",
        "chosen": "RLHF is Reinforcement Learning from Human Feedback...",
        "rejected": "RLHF is not important for LLM alignment."
    },
    # 追加データ...
]

dataset = IPODataset(dummy_data)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# ============================
# 5. 学習ループ
# ============================
from transformers import AdamW

optimizer = AdamW(model.parameters(), lr=5e-6)
num_epochs = 3
beta = 0.01  # IPOのbeta（小さめが推奨）

for epoch in range(num_epochs):
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        prompts = batch["prompt"]
        chosens = batch["chosen"]
        rejecteds = batch["rejected"]

        loss = ipo_loss(model, tokenizer, prompts, chosens, rejecteds, beta=beta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

