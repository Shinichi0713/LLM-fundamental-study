
Last time, we covered Graph Databases. By representing data correlations and causal relationships through graph structures, Graph DBs excel at analyzing complex connections that are often difficult to detect with traditional RDB searches.

Today, I will explain the mechanisms and effectiveness of causal analysis in Graph Theory—the foundational logic upon which Graph Databases are built.

# Mechanism

The reason graph networks—especially **causal graphs** and **causal Bayesian networks**—can discover causal relationships is that they do not merely rely on statistical **correlations**, but instead structurally model the **mechanisms by which data are generated**.

Why can we distinguish between **cause** and **effect** using graphs?
The key principles can be explained through the following three points.

---

## 1. Directed Structure (DAG: Directed Acyclic Graph)

Traditional correlations (such as scatter plots) have no direction.
Even if there is a correlation between **ice cream sales** and **drowning accidents**, we cannot determine which causes the other.

In graph networks, relationships are represented using **arrows (edges)**.

* $$X \to Y$$ : **X** causes a change in **Y**

**Causal assumption:**
The graph is defined as a one-way flow from **parent (cause)** to **child (effect)**. Because it contains **no cycles**, it becomes possible to mathematically trace chains of influence.

---

## 2. Conditional Independence and the Theory of d-Separation

This is the strongest mathematical foundation.

By looking only at the **structure of the graph**, we can determine which variables influence others and which variables are **independent**.

In particular, identifying the following three basic structures helps eliminate **spurious correlations**.

### Chain Structure

$$
X \to Z \to Y
$$

If **Z** is fixed (observed), the relationship between **X** and **Y** disappears.

---

### Fork Structure

$$
X \leftarrow Z \to Y
$$

When the **common cause Z** is fixed, the correlation between **X** and **Y** disappears.
This indicates that **Z was the source of the apparent correlation**.

---

### Collider Structure

$$
X \to Z \leftarrow Y
$$

Even if **X** and **Y** are originally unrelated, conditioning on their **common effect Z** creates a correlation between them (e.g., **selection bias**).

---

## 3. Simulation of Interventions

The concept that enables causal inference on graphs is the **do-operator**, proposed by **Judea Pearl**.

Two different probabilities must be distinguished.

### Correlation

$$
P(Y|X)
$$

This answers:

> “What happens to **Y** when we observe that **X** occurs?”

---

### Intervention

$$
P(Y|do(X))
$$

This answers:

> “What happens to **Y** if we **forcefully change X**?”

Using graph networks, we can mathematically predict what would happen under interventions using **historical data**, even without performing real-world experiments such as **A/B tests**.

This process is called **Causal Discovery**.

---

## 4. Algorithmic Discovery of Causality

Recently, algorithms capable of **automatically discovering causal directions** from data have advanced significantly.

### PC Algorithm

* Checks conditional independence between variables
* Constructs the **skeleton of the graph** using an elimination process

### LiNGAM

* Utilizes **non-Gaussianity (distortion)** in the data
* Statistically determines causal direction

Example:

$$
X \to Y \quad \text{or} \quad Y \to X
$$

---

# Mathematical Representation

Each level of the **Ladder of Causation** (observation, intervention, counterfactual) can be rigorously expressed using the mathematical framework known as the **Structural Causal Model (SCM)**.

Here we explain how a simple **graph of nodes and arrows** becomes a full mathematical model.

---

## 1. Definition of Structural Causal Models (SCM)

An arrow in a graph such as

$$
X \to Y
$$

is mathematically defined as the following **assignment equation**.

$$
Y := f_Y(X, U_Y)
$$

Where:

* **X** : direct cause (parent node)
* **$$U_Y$$** : exogenous variable representing external factors such as individual differences or measurement noise
* **$$f_Y$$** : mechanism function that determines **Y** from **X** and **U**

### Important distinction

Unlike relational database equations such as

$$
Y = X
$$

a reverse assignment

$$
X := Y
$$

is **not allowed**.

This asymmetry represents the **mathematical embodiment of causal direction**.

---

## 2. Second Layer: Intervention via the do-Operator

The question

> “What if we forcibly set $$X = x$$?”

is represented by **removing the original equation that determines X and replacing it with a constant**.

$$
P(Y | do(X=x))
$$

Mathematically, this corresponds to **removing all incoming edges into X in the graph** and computing the distribution of **Y** in the modified model.

This clearly distinguishes:

### Observation

$$
P(Y|X)
$$

Statistics of **Y when X happens to be x**.

### Intervention

$$
P(Y|do(X))
$$

Distribution of **Y when X is forcibly set to x**.

---

## 3. Third Layer: Counterfactual Reasoning

A counterfactual question such as

> “If X had been $$x'$$ instead of $$x$$, what would Y have been?”

is represented as

$$
Y_{X=x'}(u)
$$

This is solved through three mathematical steps.

---

### Step 1: Abduction

Use the observed data $$(x, y)$$ to infer the value of the latent background variable $$U$$ for that individual.

$$
P(U | X=x, Y=y)
$$

---

### Step 2: Action

Replace the structural equation for **X** with

$$
X := x'
$$

This represents the **intervention**.

---

### Step 3: Prediction

Using the modified model and the inferred value of **U**, compute the new value of **Y**.

---

## 4. Factorization of the Joint Probability Distribution

Given a graph structure $$G$$, the joint distribution

$$
P(x_1, \dots, x_n)
$$

can be decomposed into the product of conditional probabilities depending only on each variable’s **parents**.

$$
P(x_1, \dots, x_n) =
\prod_{i=1}^{n} P(x_i \mid pa_i)
$$

Where:

* $$pa_i$$ is the set of **parent nodes** of variable $$x_i$$.

This equation formally defines that:

> **If there is no arrow in the graph, the variables are conditionally independent.**

---

# Example

To develop an intuitive understanding of **SCM (Structural Causal Models)** and the **do-operator**, we will build a simulation using the Python causal inference library **DoWhy**.

First install DoWhy.

```
pip install DoWhy
```

This example demonstrates a typical case of **spurious correlation**:

Temperature causes both **ice cream sales** and **drowning accidents**.

---

# 1. Preparing the Simulation

We explicitly define the **mechanism (structural equations)** where **temperature affects both variables**.

```python
import numpy as np
import pandas as pd
import dowhy
from dowhy import CausalModel

# 1. Data generation (implementation of SCM)
np.random.seed(42)
num_samples = 1000

# Exogenous variables (U)
u_temp = np.random.normal(0, 1, num_samples)
u_ice = np.random.normal(0, 1, num_samples)
u_acc = np.random.normal(0, 1, num_samples)

# Structural equations
temp = 20 + 5 * u_temp

ice_sales = 2 * temp + 10 * u_ice

accidents = 0.5 * temp + 2 * u_acc

df = pd.DataFrame({
    'Temp': temp,
    'IceSales': ice_sales,
    'Accidents': accidents
})
```

---

# 2. Constructing the Mathematical Model

Next, we define the causal relationships as a graph.

```python
causal_graph = """
digraph {
    Temp -> IceSales;
    Temp -> Accidents;
}
"""

model = CausalModel(
    data=df,
    treatment='IceSales',
    outcome='Accidents',
    graph=causal_graph
)

model.view_model()
```

This produces the following structure.

Temperature directly affects both **IceSales** and **Accidents**.

---

# 3. Observation vs Intervention

## Correlation from Observational Data

If we only examine the correlation, it appears that **more ice cream sales lead to more accidents**.

```python
print(f"Correlation coefficient (IceSales vs Accidents): {df['IceSales'].corr(df['Accidents']):.3f}")
```

Example output:

```
Correlation coefficient (IceSales vs Accidents): 0.534
```

---

## Causal Intervention using the do-Operator

Now we calculate the effect of **forcing IceSales to change**.

```python
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.linear_regression"
)

print(f"Estimated causal effect (IceSales -> Accidents): {estimate.value:.3f}")
```

Example output:

```
Estimated causal effect (IceSales -> Accidents): 0.000
```

---

This confirms two important facts.

### 1. Correlation ≠ Causation

The observational probability

$$
P(Accidents | IceSales)
$$

contains the influence of the **common cause Temp** (a backdoor path).

---

### 2. Power of the do-Operator

The method `identify_effect` automatically detects that:

> Fixing **Temp** blocks the spurious path between IceSales and Accidents.

This corresponds to the **Backdoor Criterion**.

---

### 3. Identification of the Mechanism

As a result, the causal effect is estimated as

$$
0
$$

This means:

> **Even if ice cream sales were banned, drowning accidents would not decrease.**

The model correctly reflects the real-world mechanism.

---

# Conclusion

The greatest advantage of using graphs for causal analysis is that they allow us to **mathematically separate correlation from causation**.

Using **Directed Acyclic Graphs (DAGs)** provides four major practical benefits.

---

## 1. Simulation of Interventions

We can answer questions such as:

> “What will happen to sales B if policy A is implemented?”

without performing real-world experiments.

By combining **observational data and causal graphs**, we can mathematically perform **do-interventions**.

This enables safe simulation of costly or risky experiments such as:

* tax policy changes
* drug trials

---

## 2. Counterfactual Analysis

We can evaluate questions such as:

> “What would have happened if a different decision had been made?”

By defining a graph as a **Structural Causal Model**, we can estimate **potential outcomes** for individual cases.

Applications include:

* personalized medicine
* post-hoc policy evaluation


