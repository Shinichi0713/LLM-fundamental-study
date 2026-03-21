import torch
from torch import nn
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments
)

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead



# モデルのロード
model_name = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)


# データのロード

data = [
{
"prompt":"Explain AI",
"response":"AI is a technology that allows machines to learn from data."
},
{
"prompt":"What is machine learning",
"response":"Machine learning is a method where computers learn patterns from data."
},
{
"prompt":"Explain neural network",
"response":"Neural networks are models inspired by the human brain."
},
]

dataset = Dataset.from_list(data)

def preprocess(example):

    text = example["prompt"] + "\n" + example["response"]

    tokens = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=64
    )

    tokens["labels"] = tokens["input_ids"].copy()

    return tokens

tokenized_dataset = dataset.map(preprocess)

# SFT training

training_args = TrainingArguments(
    output_dir="./sft",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=1
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

trainer.train()

# PPO training
reward_data = [
{
"prompt":"Explain AI",
"good":"AI is a technology that enables machines to learn.",
"bad":"AI is computer maybe thinking."
},
{
"prompt":"Explain neural network",
"good":"Neural networks are layered models inspired by biological neurons.",
"bad":"Neural network is something in computer."
}
]

reward_dataset = Dataset.from_list(reward_data)

class RewardModel(nn.Module):

    def __init__(self, base_model):

        super().__init__()

        self.model = base_model
        hidden = base_model.config.n_embd

        self.value_head = nn.Linear(hidden,1)

    def forward(self,input_ids,attention_mask):

        outputs = self.model.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        last_hidden = outputs.last_hidden_state[:,-1,:]

        reward = self.value_head(last_hidden)

        return reward

# Reward modelの定義と学習
reward_model = RewardModel(model).cuda()

optimizer = torch.optim.Adam(reward_model.parameters(),lr=1e-5)

for epoch in range(3):

    for sample in reward_data:

        prompt = sample["prompt"]

        good = prompt + sample["good"]
        bad  = prompt + sample["bad"]

        good_tok = tokenizer(good,return_tensors="pt").to("cuda")
        bad_tok  = tokenizer(bad,return_tensors="pt").to("cuda")

        r_good = reward_model(**good_tok)
        r_bad  = reward_model(**bad_tok)

        loss = -torch.log(torch.sigmoid(r_good - r_bad)).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("loss",loss.item())



