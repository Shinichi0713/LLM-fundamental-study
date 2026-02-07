# 作業ディレクトリ
import os
os.makedirs("data", exist_ok=True)
os.chdir("data")

# Flickr30k annotations (CSV)
!wget https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k.token.txt
!wget https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip
!unzip Flickr8k_Dataset.zip

from collections import defaultdict

def load_captions(path):
    captions = defaultdict(list)
    with open(path, "r") as f:
        for line in f:
            img, caption = line.strip().split("\t")
            img = img.split("#")[0]
            captions[img].append(caption)
    return captions

captions = load_captions("Flickr8k.token.txt")
import random

image_dir = "Flickr8k_Dataset/Flicker8k_Dataset"
all_images = list(captions.keys())
random.shuffle(all_images)

# 小規模サンプル
NUM_SAMPLES = 300
selected_images = all_images[:NUM_SAMPLES]

from torch.utils.data import Dataset
from PIL import Image
import torch

class VLMCaptionDataset(Dataset):
    def __init__(self, image_dir, captions, image_list, processor, tokenizer):
        self.image_dir = image_dir
        self.captions = captions
        self.image_list = image_list
        self.processor = processor
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_name)

        image = Image.open(img_path).convert("RGB")

        caption = random.choice(self.captions[img_name])

        pixel_values = self.processor(
            images=image,
            return_tensors="pt"
        )["pixel_values"].squeeze(0)

        tokenized = self.tokenizer(
            caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=32
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": tokenized.input_ids.squeeze(0),
        }

from transformers import CLIPProcessor, GPT2Tokenizer

processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

from torch.utils.data import DataLoader

dataset = VLMCaptionDataset(
    image_dir=image_dir,
    captions=captions,
    image_list=selected_images,
    processor=processor,
    tokenizer=tokenizer
)

dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=2
)

batch = next(iter(dataloader))
print(batch["pixel_values"].shape)  # [B, 3, 224, 224]
print(batch["input_ids"].shape)     # [B, seq_len]



from torch.utils.data import Dataset
from PIL import Image
import torch

class VLMCaptionDataset(Dataset):
    def __init__(self, image_dir, captions, image_list, processor, tokenizer):
        self.image_dir = image_dir
        self.captions = captions
        self.image_list = image_list
        self.processor = processor
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_name)

        image = Image.open(img_path).convert("RGB")

        caption = random.choice(self.captions[img_name])

        pixel_values = self.processor(
            images=image,
            return_tensors="pt"
        )["pixel_values"].squeeze(0)

        tokenized = self.tokenizer(
            caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=32
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": tokenized.input_ids.squeeze(0),
        }

from transformers import CLIPProcessor, GPT2Tokenizer

processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

from torch.utils.data import DataLoader

dataset = VLMCaptionDataset(
    image_dir=image_dir,
    captions=captions,
    image_list=selected_images,
    processor=processor,
    tokenizer=tokenizer
)

dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=2
)

batch = next(iter(dataloader))
print(batch["pixel_values"].shape)  # [B, 3, 224, 224]
print(batch["input_ids"].shape)     # [B, seq_len]
