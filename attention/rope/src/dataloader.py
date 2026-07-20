
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from datasets import load_dataset
import math

# Load the available English Wikipedia snapshot safely via the modern namespace
dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train[:10000]")

class WikipediaMLMDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=128, mlm_probability=0.15):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]["text"]
        # トークナイズ
        encodings = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encodings["input_ids"].squeeze(0)  # (max_length,)
        attention_mask = encodings["attention_mask"].squeeze(0)  # (max_length,)

        # MLM 用のラベルを生成
        labels = input_ids.clone()
        # 特殊トークン（[CLS], [SEP], [PAD]）はマスクしない
        special_tokens_mask = torch.tensor(
            [1 if token_id in self.tokenizer.all_special_ids else 0 for token_id in input_ids],
            dtype=torch.bool
        )
        # マスクする位置を決定
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # 80% は [MASK], 10% はランダムトークン, 10% はそのまま
        # まず、[MASK] に置き換える位置を決定
        mask_token_id = self.tokenizer.mask_token_id
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = mask_token_id

        # 10% はランダムトークンに置き換え
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        # 残り 10% はそのまま（labels は元のトークンIDのまま）

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

mlm_dataset = WikipediaMLMDataset(dataset, tokenizer, max_length=128, mlm_probability=0.15)
dataloader = DataLoader(mlm_dataset, batch_size=16, shuffle=True, num_workers=2)


