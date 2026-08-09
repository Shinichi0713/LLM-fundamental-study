LLMs have something called **positional encoders** embedded inside them.  
Positional encoders play a crucial role in letting the LLM know **where** in the sequence each token is located, and **what** appears at that position.

Today, I will explain in detail the main approaches to positional encoding, dividing them into three broad families: **additive**, **bias-based**, and **rotary**.

## Positional Encoders

For more details, please refer to the following article:

https://yoshishinnze.hatenablog.com/entry/2025/11/13/000000

The self-attention mechanism in Transformers essentially treats the input as a **set** of tokens.  
Without positional encoding, the model would treat “The dog bit the person” and “The person bit the dog” as **exactly the same sequence**.

Positional encoders allow the model to understand **where** tokens are in the sequence.  
We can broadly categorize them into three approaches: **additive**, **bias-based**, and **rotary**.


### 1. Additive (Absolute Position): “Stamping” a position onto the word vector

This is the simplest approach, used in early Transformers (Sinusoidal) and GPT‑2.

**Mechanism**

1. We have a **word vector** representing the meaning of the token.
2. For each position (1st, 2nd, …), we create a **position vector**—a unique numerical pattern like a “stamp.”
3. We simply add them element-wise:

$$
\text{Input vector} = \text{Word Embedding} + \text{Positional Embedding}
$$

**Why does this encode position?**

In vector space, adding a **position-specific vector** slightly shifts the original word vector.  
The feed-forward networks (FFN) and attention layers can learn to read this **shift** and infer, for example, “this ‘dog’ token is the one at the beginning of the sentence.”

> **Analogy**:  
> Imagine attaching a colored sticker to the word “apple”: a red sticker for the 1st seat, a blue sticker for the 2nd seat, and so on.



### 2. Bias-based (Relative Position): Injecting “distance” directly into attention scores

Used in ALiBi, T5, etc., this approach focuses not on absolute position but on **how far apart** tokens are.

**Mechanism**

In attention, we compute a relevance score between Query (Q) and Key (K) via their dot product.  
In bias-based methods, we **add a penalty** to this score that depends on the distance between tokens:

$$
\text{Attention Score} = Q \cdot K^T - \text{(distance-based penalty)}
$$

**Why does this encode position?**

For example, ALiBi adds a linear penalty:  
- Distance 1 → subtract 1  
- Distance 2 → subtract 2  
- etc.

This means the model doesn’t need to memorize absolute indices.  
Instead, it directly learns to **pay more attention to nearby tokens** and **less attention to far-away tokens**.



### 3. Rotary (RoPE): Rotating vectors by “angle”

RoPE (Rotary Position Embedding) is the most widely used method today (LLaMA, Qwen, Mistral, etc.).  
It uses a clever mathematical trick with rotations.

**Mechanism**

We treat pairs of vector dimensions as coordinates in the complex plane.  
Then we **rotate** the Query and Key vectors by an angle proportional to their position index $ m $:

- 1st token → rotate by $ 1 \cdot \theta $  
- 2nd token → rotate by $ 2 \cdot \theta $  
- $ m $-th token → rotate by $ m \cdot \theta $

**Why does this encode position?**

When we compute the dot product between a rotated Query (position $ m $) and a rotated Key (position $ n $), trigonometric identities ensure that the result depends **only on the relative distance** $ m - n $.

1. **Absolute position representation**: Each vector is rotated by an angle tied to its own position.
2. **Relative position extraction**: The dot product automatically yields a value that depends solely on the **difference in angles** (i.e., the distance).

This “absolute encoding that becomes relative in attention” is the key mathematical trick behind RoPE.

In all these methods, **attention reads the positional “shift” (or rotation) encoded in each token’s vector** to understand relative distances and ordering.



## Why Attention Can Read the Shift Without Confusion

You might wonder: if both **semantic information** (word meaning) and **positional information** are mixed into a single vector (e.g., 4096 numbers), how does attention tell them apart?

The answer lies in three mechanisms:  
- the geometry of **high-dimensional spaces**,  
- **orthogonality** (separate subspaces), and  
- the **linear transformations** in attention.



### 1. The “room” in high-dimensional space

LLMs work in very high-dimensional spaces (4096D, 8192D, etc.).  
In such spaces:

- There are **many nearly orthogonal directions**.
- As dimensionality increases, randomly chosen vectors are almost always nearly orthogonal (angle ≈ 90°), meaning they barely interfere.

Thus, the model can effectively allocate:

- one subspace for **semantic meaning**, and  
- another subspace for **positional information**,

with minimal interference between them.

> **Analogy**:  
> In 3D space, you could use the X–Y plane for “meaning” and the Z-axis for “position.”  
> Changing Z (height) doesn’t change the X–Y coordinates (meaning).



### 2. Attention as an “information filter”

Even when word and position vectors are added into one vector, the linear layers $ W_Q, W_K, W_V $ can learn to separate them.

During training, the weight matrices learn to:

- **$ W_Q, W_K $ (for scoring)**: respond strongly to the **positional component** (shift/angle).
- **$ W_V $ (for value)**: extract mainly the **semantic component**.

In effect, these matrices act like **optical filters**, splitting the mixed vector into separate “channels” for meaning and position.



### 3. RoPE: clean separation of magnitude and phase

In RoPE, the separation is mathematically cleaner:

- **Semantic meaning** is encoded in the **magnitude** (norm) and relative direction of the vector.
- **Position** is encoded purely in the **phase (angle)** of rotation.

Because the dot product in attention depends only on the **difference in angles**, the semantic magnitude and positional phase remain independent and do not interfere.



## Traditional Positional Encoding Methods

As research has evolved, positional encoders can be grouped into four broad categories:

1. **Absolute Positional Encoding (APE)**  
2. **Relative Positional Encoding (RPE)**  
3. **Rotary / Hybrid Approaches**  
4. **No Positional Encoding (NoPE)**



### 1. Absolute Positional Encoding (APE)

Assigns a fixed vector to each absolute position (1st, 2nd, …) and **adds** it to the token embedding.

**Sinusoidal Positional Encoding**

- **Overview**: Uses sine and cosine functions with different frequencies to deterministically compute position vectors (used in the original “Attention Is All You Need” Transformer).
- **Characteristics**: No learned parameters, intuitive, but limited extrapolation to longer contexts.

**Learned Absolute Positional Embedding**

- **Overview**: Treats position vectors as learnable parameters (used in BERT, GPT‑2, GPT‑3, etc.).
- **Characteristics**: High expressivity, but cannot handle sequences longer than the pre-trained maximum length (e.g., 2048 tokens).



### 2. Relative Positional Encoding (RPE)

Focuses on **relative distance** between tokens rather than absolute indices, injecting it into attention.

**Relative Positional Bias (T5-style)**

- **Overview**: Adds a scalar bias to the attention score $ QK^T $ based on relative distance $ i - j $.
- **Characteristics**: Often uses bucketing (e.g., log-scale bins) to handle long distances efficiently.

**ALiBi (Attention with Linear Biases)**

- **Overview**: Abolishes explicit position vectors. Instead, adds a **fixed linear penalty** to attention scores: farther tokens get larger negative biases.
- **Characteristics**: Very strong extrapolation; can handle much longer contexts at inference than seen during training (used in MPT, BLOOM, etc.).



### 3. Rotary Positional Encoding (RoPE) and Its Extensions

RoPE is currently the dominant method in major LLMs (LLaMA, Qwen, Mistral, Gemma, etc.).

**RoPE (Rotary Position Embedding)**

- **Principle**: Instead of adding vectors, it **rotates** Query and Key vectors in the complex plane by angles proportional to their positions.
- **Characteristics**: Dot products automatically yield **relative distances**, combining the benefits of absolute and relative encoding.

**RoPE Context-Extension Techniques (YaRN, NTK-aware, SuPE, etc.)**

- **Overview**: Adjust or interpolate the base frequencies of RoPE to scale context length from, say, 4k tokens at training to 128k–1M+ tokens at inference.
- **Examples**:
  - Linear Interpolation / Position Interpolation (PI): shrink angles linearly.
  - NTK-aware Scaled RoPE: preserve high frequencies (local info) while interpolating low frequencies (long-range info).
  - YaRN (Yet Another RoPE Extension): more sophisticated interpolation with temperature-like adjustments to preserve attention distributions.



### Summary of Traditional Methods

| Method                | Where applied          | Extrapolation (long context) | Example models              |
|--| --- | ---- | ---- |
| **Learned APE**       | Input embedding        | ✕ (none)                     | GPT‑2, GPT‑3, BERT         |
| **T5 Relative Bias**  | Attention score        | △ (limited)                   | T5, FLAN‑T5                 |
| **ALiBi**             | Attention score        | ◎ (very strong)               | MPT, BLOOM                  |
| **RoPE**              | Q / K vectors          | ◯ (◎ with extensions)         | LLaMA, Qwen, Mistral, Gemma |



## Successors to Traditional Positional Encoders

Research on positional encoding is very active. Beyond the mainstream methods, many specialized variants address issues like 2D/multimodal extension, computational efficiency, and strict extrapolation guarantees.

Here are some representative advanced methods by category.



### 1. Complex / Rotary Evolutions (RoPE variants and alternatives)

Extensions of RoPE’s “2D rotation” to higher dimensions or nonlinear spaces.

**xPos (Extrapolatable Position Embedding)**

- **Overview**: Combines RoPE with an exponential decay factor.
- **Characteristics**: Naturally decays attention weight with distance, leading to more stable long-context extrapolation than vanilla RoPE (proposed in Microsoft’s SunNet, etc.).

**RoPE‑3D / Multi-dimensional RoPE**

- **Overview**: Extends 1D RoPE to 2D (images) or 3D (video, point clouds, spatial coordinates).
- **Characteristics**: Splits channels into blocks and applies rotations along $ x, y, z $ axes, enabling natural encoding of spatiotemporal positions in multimodal LLMs.



### 2. Continuous / Implicit Representations

Instead of explicit formulas or fixed tables, these methods treat position as a **function** generated dynamically by neural networks or differential equations.

**KERPLE (Kernelized Relative Positional Embedding)**

- **Overview**: Uses kernel functions (e.g., Gaussian, log kernels) to learn attention biases as a smooth function of relative distance.
- **Characteristics**: Generalizes linear penalties like ALiBi and provides stronger mathematical guarantees for extrapolation.

**CoPE (Contextual Position Encoding)**

- **Overview**: Assigns position not by token count, but by **semantic or contextual boundaries** (proposed by Meta).
- **Characteristics**: For example, it can encode “the third noun before this token,” overcoming limitations of purely token-count-based position encoding.



### 3. Recurrent / State-Space-Inspired Methods

These borrow ideas from RNNs and state-space models to inject positional information indirectly.

**RPE via Recurrent State / Dynamic Bias**

- **Overview**: Inserts small RNNs or 1D convolutions into the causal attention computation, letting position be implicitly tracked in hidden states.

**Sandwich / Chunk-based Positional Encoding**

- **Overview**: Splits long sequences into chunks and encodes **local position** (within chunk) and **global position** (between chunks) separately.
- **Characteristics**: Reduces computational cost and ambiguity when processing very long texts.



### 4. 2D / Vision-Language Specialized Methods

Developed for multimodal LLMs that handle images and document layouts (PDFs, web pages, etc.).

**2D Spatial Embedding (LayoutLM-style)**

- **Overview**: Converts bounding-box coordinates $ (x_0, y_0, x_1, y_1) $ into embedding vectors and adds them to token embeddings.
- **Characteristics**: Preserves visual layout in structured documents or web pages when feeding them into LLMs.

**2D RoPE (2D Rotary Position Embedding)**

- **Overview**: Extends RoPE to 2D images. Rotates feature vectors according to **X (horizontal)** and **Y (vertical)** positions.
- **Characteristics**:
  - Naturally encodes **2D relative positions** between patches.
  - Preserves vector magnitude (semantic info) while adding position via rotation.
  - Relatively stable under resolution changes; strong length/resolution extrapolation.
  - Widely adopted in modern **Vision Transformers (ViT)** and **Vision-Language Models (VLMs)** such as InternVL, Qwen2.5‑VL, LLaVA-family models, etc.



## Example: 2D APE vs. 2D RoPE in Vision Transformers

To understand the difference between **2D Absolute Positional Encoding** and **2D RoPE**, consider a simple “target search” task on an image grid.

### Problem: Target Search on an 8×8 Grid

On an 8×8 grid of image patches, we place two objects: **🍎 (apple)** and **🔪 (knife)**.

We want the ViT’s attention to find, for a given 🍎 patch, the object that is **“one step down and to the right”** relative to it.

```
Grid layout:
    0    1    2    3    4    5    6    7 (X-axis)
0 [  ] [  ] [  ] [  ] [  ] [  ] [  ] [  ]
1 [  ] [🍎] [  ] [  ] [  ] [  ] [  ] [  ]  <-- 🍎 at (X=1, Y=1)
2 [  ] [  ] [🔪1] [  ] [  ] [  ] [  ] [  ]  <-- 🔪1 at (2,2) [1 step down-right]
3 [  ] [  ] [  ] [  ] [  ] [  ] [  ] [  ]
4 [  ] [  ] [  ] [  ] [  ] [  ] [  ] [  ]
5 [  ] [  ] [  ] [  ] [🔪2] [  ] [  ] [  ]  <-- 🔪2 at (4,5) [further down-right]
```

**Question to the model**:

> Find the object whose **relative position** to 🍎 (1,1) is exactly **“one step down and to the right.”**



### How the Two Encodings Behave Internally

#### 1. 2D Absolute Positional Encoding (APE)

APE assigns a unique absolute coordinate ID (or vector) to each grid cell and adds it to the patch feature.

- 🍎’s position vector: $ P_{(1,1)} $  
- 🔪1’s position vector: $ P_{(2,2)} $  
- 🔪2’s position vector: $ P_{(4,5)} $

**Internal attention computation**

The attention score between 🍎 and 🔪1 involves:

$$
\text{Score} = (E_{\text{🍎}} + P_{(1,1)}) \cdot (E_{\text{🔪}} + P_{(2,2)})^T
$$

Expanding, we get terms like $ P_{(1,1)} \cdot P_{(2,2)}^T $—an inner product of **absolute position vectors**.

**Limitations**

- **No direct representation of relative distance**:  
  The inner product $ P_{(1,1)} \cdot P_{(2,2)}^T $ does not inherently encode “down-right by 1.” The model must **memorize** that (1,1) and (2,2) are close.
- **Poor extrapolation to unseen coordinates / higher resolution**:  
  If we enlarge the image and 🍎 moves to (2,2) and 🔪1 to (4,4), the absolute coordinate pairs change. Even if the relative step is the same, the model may fail to recognize the relationship if it hasn’t seen those exact coordinates during training.

#### 2. 2D RoPE

2D RoPE rotates vectors by angles tied to X and Y coordinates.

- 🍎 is rotated by angles $ \theta_{x1}, \theta_{y1} $ for (X=1, Y=1).  
- 🔪1 is rotated by $ \theta_{x2}, \theta_{y2} $ for (X=2, Y=2).

**Internal attention computation**

The dot product between rotated Query (🍎) and Key (🔪1) yields, via trigonometric identities, a result that depends only on the **differences** $ \Delta X = 1, \Delta Y = 1 $:

$$
\text{Score} \propto \cos(\Delta X \cdot \theta_x) + \cos(\Delta Y \cdot \theta_y)
$$

**Advantages**

- **Exact relative position capture**:  
  The attention score directly encodes “$ \Delta X = +1, \Delta Y = +1 $” (one step down-right).  
  The model can recognize this relationship with high precision.
- **Strong resolution extrapolation**:  
  Even if the grid becomes 16×16 or 🍎 and 🔪1 move elsewhere, **as long as the relative step is the same**, the angle differences—and thus the attention scores—remain identical.



### Comparison Summary

| Aspect                          | Absolute PE (APE)                              | 2D RoPE                                         |
|----|----|----|
| **How position is encoded**    | Fixed/learned vectors per coordinate, **added** | Vectors **rotated** by X/Y angles              |
| **Relative position mechanism**| Model must learn coordinate combinations        | Dot product **automatically extracts differences** |
| **2D spatial accuracy**         | X and Y can interfere                           | **X and Y rotations are independent**, clean 2D separation |
| **Extrapolation to higher res** | Weak (breaks beyond trained resolution)         | **Strong** (angles scale naturally)            |
| **Effect on semantic info**    | Can distort vector magnitude                    | **Preserves magnitude**, leaves semantics intact |

This ability to **mathematically extract 2D relative distances and directions in one shot** is why 2D RoPE has become standard in modern VLMs (Qwen2.5‑VL, InternVL, etc.).



## Overall Summary

### 1. Main Functions of Positional Encoding

Positional encoding lets Transformers—which treat tokens as a **set**—handle:

1. **Order (sequence)**:  
   Distinguish “The dog bit the person” from “The person bit the dog.”
2. **Relative distance**:  
   Adjust attention based on how far apart tokens are.
3. **2D/3D spatial relationships (for VLMs/ViTs)**:  
   Accurately represent “above/below,” “left/right,” and diagonal directions in images.

To achieve this, we either **superimpose** position onto vectors (additive/bias methods) or **rotate** them (RoPE).



### 2. How to Choose Among Major Methods

**(1) Additive (Absolute: Learned / Sinusoidal APE)**

- **Characteristics**: Simple, just add a per-position vector.  
  Learned APE is expressive but **cannot extrapolate beyond the trained length**.
- **Best for**: Fixed-length tasks where extrapolation is not critical (BERT, GPT‑2/3), or small prototypes.

**(2) Bias-based (Relative: ALiBi, T5 Relative Bias)**

- **Characteristics**: Add distance-based penalties to attention scores.  
  ALiBi offers **very strong extrapolation** to longer contexts.
- **Best for**: Models where **long-context generation/understanding** is key (MPT, BLOOM).

**(3) Rotary (RoPE and its extensions)**

- **Characteristics**: Rotate Q/K vectors; dot products yield relative distances.  
  Preserves semantic magnitude while encoding position.  
  With YaRN/NTK-aware/SuPE, **long-context extrapolation is excellent**.
- **Best for**: **Mainstream LLMs** (LLaMA, Qwen, Mistral, Gemma) where both long-context and positional accuracy matter.

**(4) 2D RoPE / 2D Spatial Embedding (for VLMs/ViTs)**

- **Characteristics**: Apply rotations (or coordinate embeddings) along X and Y axes.  
  Accurately captures **2D relative positions**.  
  Strong extrapolation under resolution changes.
- **Best for**: **Vision-Language Models (VLMs)** and **Vision Transformers (ViTs)** that need precise spatial relationships (“above,” “to the left,” “diagonally down-right”).



### 3. Concise Takeaway

- **Core role of positional encoding**:  
  Give Transformers **order, distance, and direction** so they can distinguish sequences, handle relative distances, and reason in 2D/3D space.

- **How to choose**:
  - **Small, fixed-length, simple**: Learned / Sinusoidal APE (additive).
  - **Strong long-context extrapolation**: ALiBi (bias-based).
  - **General-purpose LLM**: RoPE (rotary).
  - **Image / 2D spatial reasoning**: 2D RoPE or 2D Spatial Embedding.

- **Common goal**:  
  All methods aim to let **semantic information** and **positional information** coexist in high-dimensional space, so that attention can read the **positional shift or rotation** encoded in each token’s vector.

This concludes the overview of positional encoders and how to choose among them based on your use case.
Thank you for reading!

