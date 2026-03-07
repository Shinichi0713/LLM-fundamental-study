
from torch.utils.data import Dataset
from PIL import Image
import torch
import os
import random
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader

class VLMCaptionDataset(Dataset):
    def __init__(self, image_dir, captions, image_list, processor, max_length=32):
        """
        image_dir   : 画像フォルダ
        captions    : dict {image_name: [caption1, caption2, ...]}
        image_list  : 使用画像のリスト
        processor   : Blip2Processor
        """
        self.image_dir = image_dir
        self.captions = captions
        self.image_list = image_list
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # --- image ---
        image = Image.open(img_path).convert("RGB")

        pixel_values = self.processor(
            images=image,
            return_tensors="pt"
        )["pixel_values"].squeeze(0)

        # --- caption ---
        caption = random.choice(self.captions[img_name])

        tokenized = self.tokenizer(
            caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )

        input_ids = tokenized.input_ids.squeeze(0)
        attention_mask = tokenized.attention_mask.squeeze(0)

        # --- labels ---
        # padding tokenはloss無視
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

def collate_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }



dataset = VLMCaptionDataset(
    image_dir=image_dir,
    captions=captions,
    image_list=selected_images,
    processor=type("obj", (), {
        "tokenizer": tokenizer,
        "__call__": vision_processor
    })()
    # tokenizer=tokenizer
)

dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn
)

batch = next(iter(dataloader))
print(batch["pixel_values"].shape)  # [B, 3, 224, 224]
print(batch["input_ids"].shape)     # [B, seq_len]
