from datasets import load_dataset

# wikitext-103-raw-v1 など、事前学習用テキストデータ
dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

import torch
from torch.utils.data import Dataset

class WikiTextDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length=128):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 空行を除いた有効な文のみを抽出
        self.valid_texts = [text for text in hf_dataset["text"] if len(text.strip()) > 0]

    def __len__(self):
        return len(self.valid_texts)

    def __getitem__(self, idx):
        text = self.valid_texts[idx]
        # トークナイズ（truncation/paddingあり）
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),  # (max_length,)
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

from torch.utils.data import DataLoader
train_dataset = WikiTextDataset(dataset, tokenizer, max_length=128)
batch_size = 32
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,  # マルチプロセス読み込み（環境に応じて調整）
)

