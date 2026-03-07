## Re-challenging My Custom VLM: Improving Image Captioning Performance

I decided to take another shot at building a custom VLM after my previous attempt failed. The task is **Image Captioning**, and the primary objective is to enable the model to understand and describe images properly.

In my previous attempt, I tried training a model with a **Q-Former** architecture. Specifically, I constructed a model based on **GPT-2**, but the result was that it did not work effectively.

[Link to blog post: [https://yoshishinnze.hatenablog.com/entry/2026/02/01/000000](https://yoshishinnze.hatenablog.com/entry/2026/02/01/000000)]

## Purpose

To clarify the goals for this session:

> **Improve VQA performance significantly compared to the previous minimal VLM (CLIP + Q-Former + GPT-2) within the constraints of running on Google Colab.**

Under these conditions, rather than simply increasing the model size, it is crucial to balance:

* **An architecture suited for Image Captioning**
* **Utilization of pre-trained multimodal representations**
* **Computational complexity manageable for training and inference on Colab**

---

## Analysis of Causes

In the previous configuration (**CLIP + Q-Former + GPT-2**), the issue where **no image caption was generated** (outputting `tensor([[50256]])` → empty output) was not just a single bug, but a typical example of several structural failure factors overlapping.

### Conclusion (The Most Fundamental Reason)

**GPT-2 has weak "conditional generation" capabilities and was likely almost entirely unable to utilize the visual information provided via the Q-Former.**

As a result:

* It ignored visual features.
* It outputted `<eos>` immediately at the start.
* This resulted in an empty caption.

`50256` is the **eos token** for GPT-2.

### Primary Failure Factors (Structural Level)

#### 1. GPT-2 is not an Encoder-Decoder

GPT-2 is **Decoder-only**, and its original purpose is next-token prediction (text continuation). In the previous setup:
`Image → CLIP → Q-Former → GPT-2`
The problem is that GPT-2 lacks a built-in mechanism to "force" the use of external features (it lacks standard Cross-Attention). Consequently, even during training, the model finds that simply outputting `<eos>` results in a smaller loss than trying to use visual features, leading to **condition ignoring.**

#### 2. Insufficient Training Data

Using Flickr8k (6,000–8,000 images) against GPT-2's 100 million parameters essentially meant trying to learn visual-conditioned language generation from scratch, leading to mode collapse.

#### 3. Insufficient Information Flow (Q-Former → GPT-2)

In a typical implementation where the Q-Former output is passed through a Linear layer to the GPT-2 embedding dimension, GPT-2's weak prefix conditioning means the influence of visual features often vanishes after just a few tokens. This is why BLIP-2 utilizes OPT or T5 instead.

#### 4. Training Stability and Settings

Issues such as full-parameter fine-tuning of GPT-2, small batch sizes, high learning rates, or the absence of LoRA often lead to the collapse of the language prior. Additionally, the "pad token is same as eos token" warning likely encouraged the model to overfit on the EOS token.

---

## Model Construction

Following the BLIP-2 philosophy, I will construct a **Q-Former type VLM using Flan-T5 as the language model**.

### Overall Architecture (Flan-T5 + Q-Former)

```
Image
  ↓
Vision Encoder (CLIP ViT)
  ↓
Q-Former (Query Transformer)
  ↓
Flan-T5 (Encoder-Decoder LLM)
  ↓
Text Generation

```

### Step 1–5: Components and Setup (Colab-oriented)

* **Vision:** `openai/clip-vit-base-patch32` (Frozen)
* **Q-Former:** `Blip2QFormerModel` with 32 learnable query tokens.
* **LLM:** `google/flan-t5-base` (Frozen). This is chosen because it fits within Colab's VRAM (approx. 8–10GB).
* **Projection:** A Linear layer to map Q-Former's hidden size to T5's embedding dimension.

### Step 6–7: Forward Processing and Trainable Parameters

The core logic involves passing the Q-Former's output as the `inputs_embeds` for the T5 Encoder. Following the BLIP-2 strategy, the **Vision Encoder and LLM are frozen**, while the **Q-Former, Query Tokens, and Projection layer are trained.**

### Step 8: Why this configuration is superior

1. **Realistic for Colab:** Flan-T5-base is approx. 1GB.
2. **Stronger than GPT-2:** Being an Encoder-Decoder with native Cross-Attention and pre-training via Instruction Tuning, it has much higher VLM suitability.

---

## Experimental Results

Training was conducted for 5 epochs. The loss transitioned as follows:

* Epoch 1: 9.9075
* ...
* Epoch 5: 9.9096

### Testing

After training, I picked a sample from the loader and generated text.

**Model Output:**

* **Generated:** `- The Associated Press - The Associated Press is a news organization that covers the political and economic news from Washington, D.`

**Observation:**
Well... it’s completely different. While the model is now capable of generating sentences, the meaning is entirely unrelated to the image.

However, the fact that it can now generate text at all is progress. The short training time necessitated by Google Colab's limits likely prevented the model from fully learning the translation between visual and textual space.

---

## Reflection

In this attempt, I revised the internal structure based on previous failures. I kept the Q-Former but replaced the backbone LLM with Flan-T5, based on the hypothesis that **Cross-Attention is essential for multimodal systems to function properly.**

Although it still fails to generate descriptions that match the images—likely due to insufficient training—the model has progressed to the point of generating actual sentences.

I feel like I am just one step away.
