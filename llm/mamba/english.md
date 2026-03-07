Transformer—the foundation of today’s leading models such as GPT and Gemini—was introduced about 8–9 years ago.
Since then, although there have been incremental improvements, no architecture has clearly emerged to challenge Transformer’s dominance.

In this context, I would like to introduce **Mamba**, a method that has the potential to change the landscape.

---

# Mamba: A Potential Challenger to Transformer

## The Problems Mamba Set Out to Solve

The core issue Mamba addresses can be summarized as:

**“Transformer’s quadratic computational wall and the memory dilemma.”**

---

### 1. The Quadratic Scaling Problem

The core mechanism of Transformers—**attention**—computes relationships between **all pairs of tokens** in a sequence.

* **Problem:**
  If the number of tokens is $ L $, both computation and memory scale as $O(L^2)$.
* **Impact:**
  As input length increases, memory requirements explode, limiting practical context length to a few thousand to tens of thousands of tokens.

Mamba reduces this to **linear scaling** (proportional to $ L $), enabling efficient processing of extremely long sequences—even exceeding one million tokens.

---

### 2. The Inference-Time Memory Dilemma (KV Cache Problem)

During chat or text generation, Transformers store past tokens in a **KV cache**.

* **Problem:**

  * Memory usage grows with conversation length.
  * Each generated token must attend to the entire past, causing generation speed to slow over time.
* **Mamba’s Solution:**
  Like an RNN, it compresses past information into a **fixed-size state**.
  Memory usage remains constant regardless of sequence length, and generation speed stays stable.

---

### 3. Overcoming the “Forgetting Problem” of SSMs

State Space Models (SSMs) existed before Mamba and offered lower computational cost, but they lacked accuracy compared to Transformers.

* **Problem:**
  Traditional SSMs are **time-invariant**—they compress information using fixed rules regardless of input content.
  Important and unimportant tokens are treated similarly, causing important information to be forgotten.
* **Mamba’s Solution:**
  Introduces **Selective SSM**, which dynamically decides whether to retain or discard information based on the input.
  This enables Transformer-level contextual understanding with the efficiency of SSMs.

---

## Key Characteristics

Mamba, introduced in late 2023, is a new architecture based on **State Space Models (SSM)**.
It is attracting attention as a potential successor to Transformer.

---

### 1. Breaking the Computational Barrier

Transformer:

* Sequence length ×2 → Cost ×4

Mamba:

* Sequence length ×2 → Cost ×2

This makes long documents, video streams, and massive data sequences much more practical to process.

---

### 2. Combining the Best of RNNs and Transformers

Mamba achieves:

* **RNN-like efficiency**

  * Sequential processing
  * Compact state representation
* **Transformer-like intelligence**

  * Efficient large-scale training
  * High contextual understanding

The core mechanism is **Selective SSM**, which dynamically decides:

> What to keep and what to forget.

---

### 3. Significantly Faster Inference

Compared to Transformers:

* Up to **5× faster generation**
* Large context windows with minimal performance degradation

---

## Core Technology: Selective SSM

The key innovation can be described mathematically as:

> **Turning fixed parameters into input-dependent functions.**

If traditional SSM is a “lazy river,” Selective SSM is a **gate system that adjusts flow dynamically**.

---

### 1. Standard SSM Formulation

$$
h_t = \mathbf{A} h_{t-1} + \mathbf{B} x_t
$$

$$
y_t = \mathbf{C} h_t
$$

Where:

* $ x_t $: input token
* $ h_t $: hidden state
* $ \mathbf{A}, \mathbf{B}, \mathbf{C} $: fixed parameters

**Limitation:** These parameters are constant after training.

---

### 2. Input-Dependent Parameters (The “Selective” Mechanism)

Mamba makes parameters functions of the current input:

$$
\mathbf{B}_t = Linear_B(x_t)
$$

$$
\mathbf{C}*t = Linear_C(x_t)
$$

$$
\Delta_t = Softplus(Parameter + Linear*{\Delta}(x_t))
$$

Example:

* If the token is **“however”**:

  * The model increases memory update strength.
* If the token is **“the”**:

  * The gate suppresses updates and preserves past state.

Thus, the model dynamically filters information.

---

### 3. Mathematical Filtering Mechanism

The discretized transition:

$$
\bar{\mathbf{A}}_t = \exp(\Delta_t \mathbf{A})
$$

* $ \Delta_t \approx 0 $
  → $ h_t \approx h_{t-1} $
  → **Preserve past (ignore current input)**

* Large $ \Delta_t $
  → Reset past influence
  → **Overwrite with new information**

This enables mathematical control over memory retention and forgetting.

---

### 4. Efficient Computation: Parallel Scan

Input-dependent parameters break convolution-based acceleration.

Naively, computation would become sequential like an RNN.

Mamba solves this via:

* **Parallel scan algorithm**
* **Hardware-aware kernel design**

  * Efficient use of SRAM/HBM
  * Reduced memory transfers

This combines selective intelligence with high throughput.

---

### Summary

Mamba achieves Transformer-level capability because it:

> Dynamically rewrites memory rules based on the input itself.

This resolves Transformer’s computational and memory limitations while maintaining high performance.

---

## Architecture

Mamba can be described as:

> **An RNN-like structure with Transformer-level training efficiency and dynamic information selection.**

It consists of:

1. **Selective SSM**
2. **Block-level architecture**

---

### 1. Core Unit: Selective SSM

Input-dependent parameters:

* $ B $: what to write into state
* $ C $: what to read out
* $ \Delta $: retention vs. overwrite

Interpretation:

* Small $ \Delta $ → keep memory
* Large $ \Delta $ → overwrite memory

This reproduces attention-like focus without attention.

---

### 2. Mamba Block Structure

1. **Input split into two paths**

**Main path (SSM):**

* Linear projection
* 1D convolution (local context)
* Selective SSM (long-range dependencies)

**Gate path:**

* SiLU activation

**Fusion:**

* Element-wise multiplication (Hadamard product)

This amplifies or suppresses information selectively.

---

### 3. Hardware-Aware Design

* Parallel Scan for parallelization
* Kernel Fusion to minimize DRAM access
* Maximizes GPU efficiency

---

### 4. Comparison with Transformer

| Feature         | Transformer       | Mamba             |
| --------------- | ----------------- | ----------------- |
| Core mechanism  | Attention         | Selective SSM     |
| Memory          | Full KV cache     | Fixed-size state  |
| Long context    | Quadratic cost    | Linear cost       |
| Inference speed | Slows with length | Constant and fast |

---

## Experimental Results

The paper evaluates Mamba using the **Induction Head** task.

This measures the model’s ability to learn contextual patterns.

---

### Induction Task

Example:
If “Harry Potter” appeared earlier, when “Harry” appears again, predict “Potter”.

The test checks whether the model can:

* Selectively remember relevant patterns
* Ignore noise
* Work over extremely long sequences

---

### Results

Mamba:

* **Perfect accuracy**
* **Generalizes up to 1 million tokens**
* **4000× length extrapolation** (256 → 1M)

Transformer/previous SSM:

* Works only near training length
* Breaks around 2× extrapolation

---

### Why It Works

* **Selective gating ignores noise**
* **State-based memory preserves information regardless of distance**
* Transformer attention struggles when distance becomes extremely large

---

## Conclusion

The most important function of attention is **selective information processing**.

This capability enabled deep language understanding but came with enormous computational cost.

Mamba achieves:

* Transformer-level selectivity
* Linear computational scaling
* High-speed inference

Recent versions (Mamba-2, Mamba-3) further refine the mathematical foundations.

Transformer has been a powerful but resource-heavy paradigm.
If Mamba continues to advance, it may contribute to the **democratization of intelligence**.

This is the promise behind this emerging architecture.
