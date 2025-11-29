# Purpose of this page
this page is created to show the basic of neural network, which helps to understand neural network. 


## GradientDescent

The core relationship between **Gradient Descent** and **Neural Networks** is that Gradient Descent is the **optimization algorithm** that enables the network to **learn**.

## Key Relationship Summary

Gradient Descent is the essential mechanism used to adjust the parameters (**weights** and **biases**) of a Neural Network to minimize its **Loss Function** (the measure of error).

### 1. The Goal: Minimizing Loss

* **Neural Network Learning** is fundamentally an **optimization problem**: finding the specific set of parameters that results in the lowest possible error when comparing the network's predictions to the true target values.
* This error is quantified by the **Loss Function** (e.g., Mean Squared Error or Cross-Entropy).

### 2. Computing the Gradient (Backpropagation)

* Before adjusting the parameters, the network must determine the **direction** and **magnitude** of the error gradient (the slope) at the current parameter values.
* **Backpropagation** (Error Backpropagation) is the highly efficient algorithm used specifically in Neural Networks to calculate this **gradient** across all weights and biases. It tells the network exactly how much the loss will increase or decrease if a specific weight is nudged.

### 3. Updating Parameters (Gradient Descent)

* The Gradient calculated by Backpropagation indicates the direction of **steepest ascent** (where the loss increases fastest).
* **Gradient Descent** then updates the parameters by moving in the **opposite direction** (steepest descent). 
* The **Learning Rate** ($\eta$) determines the step size taken in this opposite direction, according to the formula:
    $$
    \theta_{\text{new}} = \theta_{\text{old}} - \eta \nabla J(\theta)
    $$

### 4. Role of Optimizers

* In practical deep learning, basic Gradient Descent is often replaced by more advanced **Optimizers** (like **Adam** or **RMSprop**).
* These advanced methods are still fundamentally built on the principle of Gradient Descent—they calculate and use the gradient—but they incorporate techniques (like momentum or adaptive learning rates) to make the learning process faster and more stable.


<img src="doc/image/gradient_descent_animation.gif" alt="代替テキスト" width="500" style="display: block; margin: 0 auto;">
