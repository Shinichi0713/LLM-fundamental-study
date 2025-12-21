## [purpose](https://github.com/Shinichi0713/LLM-fundamental-study)

this site exists to study LLM and technic of VLM.

<img src="image/README/welcom_llm.png" alt="代替テキスト" width="700" style="display: block; margin: 0 auto;">

## Related Repositories

Code is stored next URL:

[Shinichi0713/LLM-fundamental-study: this site is the fundamental page of LLM-mechanism](https://github.com/Shinichi0713/LLM-fundamental-study)

On the other hand, I have Reinforcement-Learning repository also.

[Shinichi0713/Reinforce-Learning-Study](https://github.com/Shinichi0713/Reinforce-Learning-Study)

## site map

1. Roadmap of LLM
   [indicate Roadmap of LLM](https://github.com/Shinichi0713/LLM-fundamental-study/blob/main/docs/roadmap_llm.md)
2. Roadmap of VLM
   [indicate Roadmap of VLM](https://github.com/Shinichi0713/LLM-fundamental-study/blob/main/docs/roadmap_vlm.md)
3. Sparce Attention
   [Explanation of Sparce Attention](https://github.com/Shinichi0713/LLM-fundamental-study/blob/main/docs/sparce_attention.md)

# Archives

## Basic of understanding LLM

the essence of basic of neuralnetwork is described below page.

the description:

- activate function
- graph theorem
- optimization
- stability method of training

<img src="image/README/climing_pinokio.png" alt="代替テキスト" width="500" style="display: block; margin: 0 auto;">

The result of investigateing is showed on below link.

[Basic of understanding LLM](https://github.com/Shinichi0713/LLM-fundamental-study/blob/main/basics_nn/basic_readme.md)

## Time Series Neural Networks

Handling Sequential Data and Word Relationships
To process textual data, it's crucial to be able to handle sequential data. Furthermore, understanding the relationships between words within a sentence is also essential.

In this context, we will be focusing on neural networks designed to address these challenges.

### Summary of Challenges in Conventional Models for Text Processing

| Feature                          | RNN / LSTM                                          | CNN                                         | Transformer (LLM)                                                               |
| :------------------------------- | :-------------------------------------------------- | :------------------------------------------ | :------------------------------------------------------------------------------ |
| **Computational Process**  | **Sequential (Slow, No Parallelization)**     | Parallelizable                              | **Ultra-Parallel (High Speed)**                                           |
| **Information Handling**   | Compresses all past information into a single point | Local (Within a fixed window only)          | **Self-Attention (Directly calculates relationships between all tokens)** |
| **Long-Term Dependency**   | Poor (Vanishing Gradient)                           | Poor (Requires extremely deep layers)       | **Excellent (Consistent regardless of distance)**                         |
| **Positional Information** | Implicitly maintained through sequential processing | Often lost with Pooling                     | **Explicitly defined with Positional Encoding**                           |
| **Scalability**            | Difficult                                           | Possible, but unsuitable for language tasks | **Extremely Easy (Laws of Scaling)**                                      |

[Sequential Neural Network](https://github.com/Shinichi0713/LLM-fundamental-study/blob/main/sequential_nn/README.md)

## LayerNorm Work

**Layer Normalization (LayerNorm)** helps stabilize neural network training by normalizing each layer’s activations per sample (not across the batch).

For an input vector $x$, it normalizes using:

$$
\text{LayerNorm}(x_i) = \gamma \frac{(x_i - \mu)}{\sigma + \epsilon} + \beta
$$

where $\mu$ and $\sigma$ are the mean and standard deviation of $x$, and $\gamma, \beta$ are learnable parameters.

**Why it stabilizes training:**

1. Keeps activations at a consistent scale → prevents exploding/vanishing gradients.
2. Maintains stable feature distributions across layers → smoother convergence.
3. Independent of batch size → works well with RNNs and Transformers.

**In short:**

LayerNorm acts like a “temperature regulator” for each layer, ensuring that signals remain well-scaled and learning stays stable and efficient.

LayerNorm is learnable layer.

Visualize the distribution of output changed by learning.

![layer_norm](image/README/layernorm_training.gif)

## Attention Mechanizm
### meaning
The **Q (Query)**, **K (Key)**, and **V (Value)** in the Attention mechanism are often compared to **"searching in a library"** or **"searching for a video on YouTube."**
This system is designed to efficiently extract "the most contextually relevant information" from a vast amount of data. Here is a beginner-friendly breakdown of each role.

__1. Analogizing with a Search System__

Imagine you are looking for a video on YouTube:

* **Query (Q): "The keywords you just typed into the search bar."**
* *Example:* "How to cook delicious curry."
* *Role:* This is exactly what you "want to know" or "are looking for" right now.


* **Key (K): "The titles and tags of the videos."**
* *Example:* "Ultimate Curry from Spices," "Quick 5-Minute Curry Recipe."
* *Role:* These are the "labels or indices" used to match against the Query. They are used to measure how well a video matches your Query.


* **Value (V): "The video file itself (the content)."**
* *Role:* This is the "actual information" you want to obtain in the end.


__2. How it Works within LLM Calculations__

In actual implementation code, these three components are processed through the following steps:

1. **Measuring Similarity between Q and K (Score Calculation)**
The system calculates how deeply the "current word (Q)" is related to the surrounding "words (K)."
* *Example:* For the word "I (Q)," which of the surrounding words—"school," "to," or "go"—is the most important?


2. **Determining the Weights (Attention Degrees)**
A higher score (weight) is assigned to the words where Q and K have high compatibility.
3. **Gathering V using Weighted Averaging**
The system collects more "information (V)" from words that have high weights.
* *Result:* The vector for the word "I" is strongly mixed with the information of the action "go."


### Caluculation Method
By taking the inner product between each Query and Key, we compute the relevance scores:

$$
\text{score}(Q, K) = Q K^\top
$$

→ This produces a score matrix indicating  **how much each word attends to every other word** .

(The shape is ( n \times n ))

### Scaling and Softmax Normalization

The scores are then normalized into a probability distribution as follows:

$$
\text{Attention Weights} = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)
$$

Dividing by ( \sqrt{d_k} ) prevents the inner product values from becoming too large,

which helps stabilize the gradients during training.

![1761459118097](image/README/1761459118097.png)

### Multi-Head attention

Here are the key characteristics of Multi-Head Attention:

- Multiple Perspectives (多様な視点): It's like having multiple specialized "heads" (attention mechanisms) analyzing the same input at the same time.
- Diverse Relationships (多様な関係性): Each head learns to focus on different types of relationships or patterns between words. For instance, one might focus on a verb and its subject, while another captures a pronoun's reference to a distant noun.
- Richer Understanding (より豊かな理解): By combining the insights from all the heads, the model gets a deeper, more nuanced, and richer understanding of the input data.
  Parallel Processing (並列処理): These attention calculations happen in parallel, making the process efficient.
- Enhanced Expressiveness (表現力の向上): This mechanism allows the model to capture complex information from the entire input sequence, which improves its overall ability to handle language tasks.

![1763166137173](image/README/1763166137173.png)

### Local Attention

Each token only attends to tokens within a**fixed, adjacent window** around itself. This mimics locality bias in CNNs.

![1763177663253](image/README/1763177663253.png)

### Global Attention

at the conceptual level, the CLS-token-based global attention in Vision Transformers (ViT) can be understood as playing the same functional role as Global Attention in Sparse Attention models.
However, they differ in purpose, design, and implementation.

CLS Token in ViT as Global Attention

In Vision Transformers:

```
[CLS] → [Patch₁, Patch₂, ..., Patchₙ]
```

- The CLS token attends to all image patches
- It aggregates global visual information
- The final CLS representation is used for classification or downstream tasks

From a functional perspective, the CLS token behaves as a global attention node.

![1766232218054](image/README/1766232218054.png)

### Local + GLobal Attention

I implemented this using Global + Local attention.While using Local Attention only yielded the previously mentioned result (referring to limited context), incorporating Global Attention allows attention to cover the entire sequence as shown below.This method, adopted by Big Bird (Big Bird: Transformers for Longer Sequences), is an approach that achieves both high computational speed (linear, $O(n)$ complexity) and strong performance.

<img src="image/README/1766232943951.png" alt="代替テキスト" width="420" style="display: block; margin: 0 auto;">


### attention of token toward the others

* The **i-th row** of the attention matrix shows  *“when the query is token i, which keys (tokens) it pays attention to (assigns weight to)”* .
* Main patterns:

  * **Diagonal dominance** — the token mainly attends to itself or nearby tokens → local processing (common in language models)
  * **Focus on CLS / [EOS]** — capturing overall sentence context or summary (typical in classification tasks)
  * **Attention following phrase structure** — e.g., verbs strongly attending to their objects, adjectives attending to the nouns they modify (semantic relationships)
  * **Specialization across heads** — some heads focus on local relationships, while others capture long-range dependencies

![1761463514970](image/README/1761463514970.png)

## Tokenizer

In this experiment, we used **SentencePiece** to train a tokenizer for Japanese text segmentation using the  **Livedoor News Corpus** .

**Output Results:**

Before training, every character was treated as a separate token.

```
🔹 Before Training (Character-level segmentation):
['私', 'は', '自', '然', '言', '語', '処', '理', 'を', '勉', '強', 'し', 'て', 'い', 'ま', 'す', '。']

['そ', 'の', 'ソ', 'フ', 'ト', 'ウ', 'ェ', 'ア', 'は', '多', 'く', 'の', 'ユ', 'ー', 'ザ', 'ー', 'か', 'ら', '賛', '同', 'を', '得', 'て', 'い', 'る', '。']
```

After training, the SentencePiece model produced the following segmentation results.
It learned that words like **“して” (shite)** and **“から” (kara)** should be kept together as single tokens.

However, terms like **“自然言語解析” (natural language analysis)** and **“ソフトウェア” (software)** were still split into multiple parts.
This is likely because such words appeared less frequently in the Livedoor dataset.

With a larger training corpus, we expect the model to recognize compound words like “自然言語解析” and “ソフトウェア” as unified tokens.

```
🔹 After Training (SentencePiece segmentation):
['▁', '私', 'は', '自', '然', '言', '語', '処', '理', 'を', '勉', '強', 'して', 'い', 'ます', '。']
['▁', 'その', 'ソ', 'フ', 'ト', 'ウ', 'ェ', 'ア', 'は', '多', 'く', 'の', 'ユ', 'ー', 'ザ', 'ー', 'から', '賛', '同', 'を', '得', 'ている', '。']
```

### sentence distribution

with using sentence bert, we can visualize sentence meaning visualization.
below is one example.

![1766270609958](image/README/1766270609958.png)

## Positional Encoding

### Absolute PE

Positional encoding serves the role of capturing positional relationships in language.

The logic is to generate sine and cosine waves with different frequencies depending on the embedding dimension, allowing the model to recognize both short-range and long-range dependencies through the wave values.

I implemented positional encoding in PyTorch and visualized the wave patterns according to the position and dimension.

![1761986434087](image/README/1761986434087.png)

### relative PE

Relative positional encoding teaches an AI model the positional relationships between tokens in a sentence.

One of its main characteristics is that it maintains stable performance even when sentence length varies.

In implementation, markers representing positional information are embedded inside the attention mechanism, so that this information is incorporated when the embedding vectors are processed.

Below figure the work of relative positional information.

![1762062689199](image/README/1762062689199.png)

After learning relation with using dataset, visualize the weight of head of relative, horizontal axis means the distance of token.

With this result, with using relative pe, basically, attention focus on nearside tokens. And the farer, attention become less focus.

![1762136383941](image/README/1762136383941.png)

### RoPE

with using RoPE, show transition of the attention score.

![layer_norm](image/README/attn_animation.gif)

I compared Absolute Positional Encoding (PE) and Rotary Position Embedding (RoPE).

While Absolute PE explicitly encodes the position of each token within a sentence, RoPE naturally incorporates the **relative distance between tokens** into the Attention computation.

I visualized the positional embeddings (PE) of each token for both methods.

![1762981322548](image/README/1762981322548.png)

## Inner Features

with VLM-Lens, we analysis the inner feature.

visualize as heat map.

![1760833255365](image/README/1760833255365.png)

### NewArchitecture

Now thinking new architecture.

Key feature is next.

- PE is used RoPE.
- Attention is Hybrid Attention.
- Normalize is LayerNorm.

![1763850597068](image/README/1763850597068.png)

### ModernBert

![1762669526990](image/README/1762669526990.png)

I performed MLM (Masked Language Modeling) with Modern-BERT on a small language dataset. After that, I visualized the attention scores.

## ViT

First, enact Patching toward original image.

![1760853369164](image/README/1760853369164.png)

the prediction of trained ViT.

ViT is trained with the dataset "CIFAR-10".

![1760993518301](image/README/1760993518301.png)

## tuning of LLM

investigate how to tune LLM.
summary is descripted as below.

mainly, intent to use these knowledge to train LLM, which generate sentence.

<img src="image/README/learning_llm.png" alt="代替テキスト" width="500" style="display: block; margin: 0 auto;">

overview:
[what is fine tuning?](https://shinichi0713.github.io/LLM-fundamental-study/docs/11-fine-tuning)

PEFT:
[retain-cost-fine-tuning](https://shinichi0713.github.io/LLM-fundamental-study/docs/12-retain-cost-fine-tuning)

LLM-learning:
[efficiently train llm](https://shinichi0713.github.io/LLM-fundamental-study/docs/13-LLM-learning)

## reference site

#### NLPコロキウム

this site supply LLM researchers insight.
we must study that.
https://www.youtube.com/watch?v=NatwshCTe_4

#### torch environment

at first, check the vesion respondance.
https://pytorch.org/get-started/locally/
