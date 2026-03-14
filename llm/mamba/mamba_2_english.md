
# Mamba-2 Architecture (Continuation from the Previous Mamba Article)

This article introduces **Mamba-2**, the architecture that follows the previous Mamba model.

Mamba-2 is a next-generation architecture released in **May 2024**, following the success of the original Mamba.

In short, it can be described as:

**“A model redesigned to fully utilize GPU performance by discovering the mathematical commonality between Transformers and SSMs (State Space Models).”**

---

# Overview

## 1. Problems Mamba-2 Aimed to Solve

Although the original Mamba was an excellent model, it had three major challenges regarding practical deployment and scaling.

### ① Inability to Fully Utilize GPU Matrix Computation Units

Modern GPUs (such as NVIDIA Tensor Cores) are designed to perform **large matrix multiplications (MatMul)** extremely efficiently.

**Problem:**
The core computation method of the original Mamba, called **parallel scan**, differed from MatMul. As a result, it was difficult to fully exploit the hardware's computational potential.

---

### ② Theoretical Gap with Transformers

SSMs and Transformers evolved independently and follow very different theoretical frameworks.

**Problem:**
It was unclear how successful techniques from Transformers (such as **multi-head structures**) should be applied to SSM-based architectures.

Because the two architectures were theoretically disconnected, there was insufficient theoretical grounding for designing hybrid models.

---

### ③ Efficiency in Large-Scale Training

When scaling models to **hundreds of billions of parameters (LLM scale)**, the original Mamba architecture risked bottlenecks in memory control and computational efficiency.

---

## 2. Core Idea of Mamba-2: SSD (Structured State Space Duality)

To solve these issues, Mamba-2 introduced a theory called **SSD (Structured State Space Duality)**.

* **Discovery of duality:**
  The researchers proved that when certain structural constraints are added to the SSM formulation, it becomes **mathematically equivalent to linear attention**.

* **Hybrid behavior:**
  As a result, Mamba-2 can flexibly choose between two computation styles:

  * Processing data **sequentially like an RNN**
  * Processing data **in parallel like a Transformer**

This allows the model to adopt the most efficient computation strategy depending on the situation.

---

## 3. Key Features and Improvements of Mamba-2

### ① Dramatically Faster Computation (2–8× Speedup)

Through the SSD theory, the SSM computation was successfully rewritten in the form of **matrix multiplication**, which GPUs handle extremely efficiently.

This enables **much faster training and inference** compared to the original Mamba.

---

### ② Adoption of a Multi-Head Structure

Inspired by the **multi-head attention** mechanism in Transformers, Mamba-2 introduces a **multi-head SSM** architecture.

**Effect:**

Different heads can process different types of information in parallel, such as:

* syntax
* factual knowledge
* sentiment

This greatly improves the model's representational power.

---

### ③ Integration with Transformers (e.g., Jamba)

Since Mamba-2 establishes a mathematical relationship with Transformers, it becomes easy to construct **hybrid architectures** such as:

* alternating **Transformer layers**
* and **Mamba layers**

This allows models to combine:

* the **strong reasoning ability** of Transformers
* the **speed and long-context capability** of Mamba

Many such hybrid models are now emerging.

---

# State Space Models

**SSM (State Space Model)** is a mathematical framework that has long been used in **control theory** and **statistics**.

In the context of deep learning, an SSM aims to combine the strengths of both:

* **RNNs**, which summarize past information into a compact state
* **Transformers**, which allow parallel computation

We can understand the mechanism in three steps.

---

## 1. Basic Idea: Information Summarization

The essence of SSMs is to **continuously compress incoming data into a small vector representing the current state**.

Instead of storing all past data explicitly, the model updates a **hidden state vector**.

* **Input ($x$):**
  The current word or signal.

* **State ($h$):**
  A memory that contains the entire past context.

* **Output ($y$):**
  The predicted value based on the memory and the current input.

---

## 2. Mathematical Mechanism: Two Equations

SSMs are defined using two linear differential (or difference) equations.

### State Equation

$$
h'(t) = \mathbf{A}h(t) + \mathbf{B}x(t)
$$

This equation determines how to update the memory by combining the current state with new input.

---

### Output Equation

$$
y(t) = \mathbf{C}h(t)
$$

This equation determines how to extract useful information from the accumulated memory.

The most important component is the matrix **$\mathbf{A}$**, which defines **how past information decays or persists over time**.

---

## 3. Why SSMs Are Attracting Attention

Transformers suffer from the **quadratic complexity problem**, where computation grows with the square of the sequence length.

SSMs overcome this by switching between two computational modes.

---

### ① RNN-Like Behavior During Inference (Memory Efficient)

Data is processed one element at a time while updating only the state $h$.

Therefore, **memory consumption does not grow with sequence length**, making SSMs suitable for **edge AI and on-device inference**.

---

### ② CNN-Like Behavior During Training (Parallelizable)

By applying mathematical transformations (discretization and convolution conversion), the sequential computation can be rewritten as **parallel operations across the entire sequence**.

This allows SSMs to train at speeds comparable to Transformers.

---

## 4. Evolution of SSMs: Toward Mamba

Traditional SSMs used **fixed matrices** $\mathbf{A}, \mathbf{B}, \mathbf{C}$ independent of input.

This limited their ability to capture complex context.

Mamba introduced **selectivity**, meaning:

**the matrices dynamically change depending on the input token.**

This innovation finally allowed SSMs to achieve intelligence comparable to Transformers.

---

# Mechanism of Mamba-2

The mechanism of Mamba-2 can be summarized as:

**Mathematically rewriting SSMs into matrix computations (MatMul), which GPUs handle most efficiently.**

Three key pillars enable this.

---

# 1. SSD (Structured State Space Duality)

The fundamental principle of Mamba-2 is the proof that:

**State Space Models and Attention are mathematically equivalent.**

Previously:

* SSMs (RNN-like)
* Attention (Transformer)

were considered entirely different algorithms.

Mamba-2 shows that if certain structural constraints are applied to SSMs, they become equivalent to **linear attention**.

This equivalence is called **Structured State Space Duality (SSD)**.

---

# 2. Reduction to Matrix Multiplication

The original Mamba required **parallel scan operations**, which GPUs handle inefficiently.

Using SSD theory, Mamba-2 transforms the computation process:

1. **Sequence segmentation**
   Long sequences are divided into small blocks.

2. **Matrix computation within blocks**
   Each block is processed using **matrix multiplication**, similar to Transformers.

3. **SSM between blocks**
   State updates handle memory transfer across blocks.

This allows Mamba-2 to fully utilize **GPU Tensor Cores**, achieving up to **8× speedup** over the original Mamba.

---

# 3. Multi-Head State Structure

Inspired by multi-head attention, Mamba-2 reorganizes state management.

### Shared State

In the original Mamba, each input channel had an independent memory.

Mamba-2 instead introduces **shared states across channels**, similar to multi-value or multi-head attention.

**Advantages**

* Lower memory usage
* Higher representational power

---

# 4. Overall Architecture

The Mamba-2 block is simpler and closer to the Transformer architecture.

Processing steps:

1. **Input Projection**
   Split the input into multiple heads.

2. **SSD Layer**
   Core computation combining matrix operations and SSM.

3. **Gating and Activation**
   Filter the processed information.

4. **Output Projection**
   Return to the original dimensionality.

Mamba-2 enables:

* **parallel Transformer-like computation during training**
* **memory-efficient RNN-like inference**

---

# Architectural Improvements

### 1. Unified Linear Projection

In Mamba-1, several independent linear transformations were applied before SSM processing.

In Mamba-2, these transformations are merged into **one large projection**, from which parameters $(A, X, B, C)$ branch out.

**Purpose**

Combine multiple small matrix operations into one large operation to maximize GPU efficiency.

**Effect**

* fewer kernel launches
* improved throughput

---

### 2. Simplified Parameter Generation

In Mamba-1, parameters $A, B, C$ were generated through multiple transformations.

In Mamba-2, they are derived directly from the initial projection using simpler pathways.

This simplification is based on **SSD theory**, which allows $A$ to be treated as a scalar (or structured parameter).

---

### 3. Nonlinearity and Normalization

A new **N (Normalization/Nonlinearity)** component appears in Mamba-2.

Typically this includes **Group Normalization** applied after the SSM output.

This stabilizes outputs across multiple heads and ensures scalability similar to Transformer multi-head attention.

---

# Mathematical Foundation

The mathematical essence of Mamba-2 lies in redefining SSMs as **a single large matrix computation**.

The key concept is the formulation using **semiseparable matrices**, derived from SSD theory.

---

## 1. Matrix Representation of SSM

The discretized SSM equation

$$
h_t = \bar{A}*t h*{t-1} + \bar{B}_t x_t
$$

can be expanded across a sequence of length $L$.

The output vector

$$
y = (y_1, y_2, \dots, y_L)^\top
$$

can be written as

$$
y = Mx
$$

where $M$ is a lower triangular matrix:

$$
M =
\begin{pmatrix}
\bar{C}_1 \bar{B}_1 & 0 & 0 \
\bar{C}_2 \bar{A}_2 \bar{B}_1 & \bar{C}_2 \bar{B}_2 & 0 \
\bar{C}_3 \bar{A}_3 \bar{A}_2 \bar{B}_1 & \bar{C}_3 \bar{A}_2 \bar{B}_2 & \bar{C}_3 \bar{B}_3
\end{pmatrix}
$$

Each element satisfies

$$
M_{ij} = \bar{C}*i \left(\prod*{k=j+1}^{i} \bar{A}_k\right) \bar{B}_j
$$

This structure is called a **1-semiseparable matrix**.

---

## 2. SSD Theory: Equivalence to Attention

If $\bar{A}_t$ is treated as a scalar $a_t$ (specifically $e^{\Delta_t A}$), then

$$
M_{ij} = \bar{C}_i \bar{B}*j \prod*{k=j+1}^{i} a_k
$$

Compare this with attention scores:

$$
q_i^\top k_j
$$

The correspondence is:

* Query: $q_i = \bar{C}_i$
* Key: $k_j = \bar{B}_j$
* Decay mask: $\prod_{k=j+1}^{i} a_k$

Thus Mamba-2 computes:

**linear attention with an input-dependent exponential decay mask.**

---

## 3. Block Matrix Acceleration

Instead of computing the full matrix $M$, Mamba-2 performs **block decomposition**.

The sequence is divided into blocks of size $Q$.

Two computations are performed:

### Intra-Block Computation

Within each block:

**standard matrix multiplication (MatMul)** using Tensor Cores.

### Inter-Block Computation

State propagation between blocks is handled recursively via SSM.

Thus

$$
M = M_{\text{intra}} + M_{\text{inter}}
$$

This preserves **linear complexity**

$$
O(L)
$$

while achieving Transformer-like parallel efficiency.

---

## 4. GPU Execution Steps

Actual GPU execution proceeds as follows:

1. **Input segmentation**
   Split input $x$ from shape $B \times L \times D$ into blocks.

2. **Block scan**
   Compute the terminal state for each block using parallel scan.

3. **State propagation**
   Pass the state between blocks.

4. **Output synthesis**
   Combine block inputs and propagated states via matrix operations.

---

# Performance Impact

Mamba-2 maintains Transformer-level accuracy while dramatically improving efficiency.

---

## 1. Throughput and Speed

Training speed:
**2–8× faster** than the original Mamba.

Inference efficiency:
Linear complexity enables processing of **hundreds of thousands to millions of tokens** without memory explosion.

---

## 2. Language Model Benchmarks

Performance comparable to or better than Transformers.

Example:

A **2.7B parameter Mamba-2 model** outperforms similarly sized Transformer models such as **Pythia** on benchmarks like **MMLU**.

---

### Improved Information Retention

The state dimension increased from

* **16 → 128**

an **8× expansion**, improving the model's ability to retain contextual information.

---

# Conclusion

Mamba-2 solves many practical issues of the original Mamba architecture.

Key achievements include:

1. Improved GPU computational efficiency
2. A mathematical bridge between Transformer techniques and SSM
3. Improved efficiency for large-scale training

Transformers remain dominant today, but alternative architectures like Mamba-2 offer a promising path forward.

They may help overcome potential scaling limitations of Transformer-based models.

---

# References

Author's explanation blog
[https://tridao.me/blog/2024/mamba2-part1-model/](https://tridao.me/blog/2024/mamba2-part1-model/)

Author's Mamba blog article
[https://yoshishinnze.hatenablog.com/entry/2026/01/25/182406](https://yoshishinnze.hatenablog.com/entry/2026/02/24/000000)

