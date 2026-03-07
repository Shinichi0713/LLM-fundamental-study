
## Report on VLM (Vision-Language Model) Implementation and Training

Building on my previous work, I have selected the training code, implemented the model, and conducted an actual training run for a Vision-Language Model (VLM). Below is a summary of the implementation process and the test results.

## 1. Objectives and Methodology

This experiment focuses on a minimal VLM configuration using  **Q-Former + LoRA** .

### 1.1 Intent of the Test: What am I trying to verify?

#### Background

A VLM typically consists of:

* A **Vision Encoder**
* A **Language Model (LLM)**
* An **Intermediate Mechanism** (such as a Q-Former) to bridge them.

Because implementation and training are complex, it often becomes unclear:

> "Is the visual information truly being used for language generation?"
>
> "Which module is performing which role?"

#### Goal of this Test

**This experiment aims to confirm one specific point:**

> **Does providing visual features via a Q-Former cause the LLM's output to change depending on the image?**

Specifically, I want to verify if the **division of labor** is functioning correctly:

* **Vision Encoder:** Acts as a "feature extractor."
* **Q-Former:** Acts as a "summarizer/selector" of visual information.
* **LLM (with LoRA):** Acts as an "image-conditioned language generator."

#### Why Q-Former + LoRA?

* If we train the entire LLM, the *re-learning of language abilities* and *visual conditioning* become entangled.
* If we only train the Q-Former, the LLM might not fully utilize the visual tokens.

Therefore, I adopted a  **minimal configuration where causality is easy to track** :

* **Q-Former:** Extracts and compresses visual information.
* **LoRA:** Learns only *how* the LLM should use these visual tokens.

### 1.2 What this test is NOT

This test is  **not a performance competition** . I have intentionally avoided:

* ❌ High-precision evaluation on large-scale datasets.
* ❌ Full fine-tuning of the Vision Encoder or LLM backbone.
* ❌ Complex tasks (VQA, instruction following, etc.).

The goal is strictly **structural understanding and behavioral verification.**

### 1.3 Model Architecture (Conceptual Diagram)

```
Image
 ↓
Vision Encoder (CLIP, frozen)
 ↓
Visual Feature Sequence
 ↓
Q-Former (Trainable)
 ↓
Small number of Visual Tokens
 ↓
Linear Projection (Trainable)
 ↓
LLM Input Space
 ↓
LLM + LoRA (Only LoRA is Trainable)
 ↓
Text Generation
```

### 1.4 Test Procedure

* **Step 1: Preparation with Minimal Data:** Use a small subset of COCO or Flickr30k for Image Captioning to check if the loss decreases and the training pipeline flows correctly.
* **Step 2: Verification of Freeze/Train Separation:** Ensure that only a small number of parameters (Q-Former, LoRA, Projection) are being updated.
* **Step 3: Confirmation of Learning Success:** Verify that loss decreases per epoch and that outputs change when the image is changed.
* **Step 4: Controlled Experiments:** Compare the "With Q-Former" setup against "No Q-Former (Mean Pooling)" and "No LoRA" to isolate the role of each component.

---

## 2. Experimental Target

The target task is  **Image Captioning** , a fundamental VLM task. The goal is to verify that the model can describe an image as intended. The experiment was conducted in a **Google Colab** environment.

### 2.1 Dataset: Flickr8k

I used the  **Flickr8k Dataset** , a representative dataset for learning the correspondence between images and natural language captions.

| **Item**           | **Content**                      |
| ------------------------ | -------------------------------------- |
| **Images**         | 8,000                                  |
| **Captions**       | 5 sentences per image                  |
| **Total Captions** | ~40,000 sentences                      |
| **Language**       | English                                |
| **Content**        | People, animals, everyday scenes, etc. |

---

## 3. Model Configuration Details

This is a **minimal BLIP-2 style VLM** scaled down for Google Colab.

1. **Vision Encoder (CLIP ViT):** Frozen. It extracts high-quality visual features without needing further tuning.
2. **Q-Former:** The "star" of this experiment. It uses **learnable query tokens** and **Cross-Attention** to extract essential information from the raw visual features into a fixed length (e.g., 32 tokens).
3. **Projection Layer:** A simple Linear layer that maps the Q-Former's output to the LLM's embedding space (handling the difference in dimensions and distribution).
4. **LLM (GPT-2 + LoRA):** Uses  **GPT-2 Small** . LoRA is used to allow the model to adapt to visual conditioning without forgetting its language priors or exceeding Colab's VRAM.

---

## 4. Experimental Results

### 4.1 Implementation Code

The code used for this experiment is stored in the following repository:

[https://github.com/Shinichi0713/LLM-fundamental-study/tree/main/vlm/src/simple_llava](https://github.com/Shinichi0713/LLM-fundamental-study/tree/main/vlm/src/simple_llava)

### 4.2 Training

I trained the model for  **3 epochs** . While training might still be insufficient, I proceeded to check the initial performance.

### 4.3 Evaluation

When I input an image for testing, **no text was output.** The model output an `[EOS]` token immediately, resulting in no caption generation. This result was consistent across multiple test images; the current model failed to generate captions as expected.

---

## 5. Conclusion

To ensure the model ran on Google Colab, I had to significantly reduce the model size and use **Q-LoRA** to minimize trainable parameters in the LLM.

Regarding the results: the fact that an `[EOS]` token was output suggests that the code-level training was actually functioning (otherwise, the model would likely output a string of nonsensical, random tokens).

However, the interpretation is that **the LLM does not yet understand what it is supposed to do.** The likely causes are:

1. **Insufficient training time/data.**
2. **Insufficient VLM capacity (model size).**

I will take these points as lessons learned for my next attempt.
