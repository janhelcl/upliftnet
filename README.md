# UpliftNet

**UpliftNet** is a Python package that implements a causal loss function for neural networks, enabling them to directly optimize for uplift, also known as Conditional Average Treatment Effect (CATE). This approach is particularly useful in fields like marketing, medicine, and public policy, where the goal is to identify individuals who will respond most favorably to an intervention.

The core of this package is the `UpliftNet` class, a custom Keras model that optimizes a differentiable version of the **Promotional Cumulative Gain (PCG)** metric. This is achieved by leveraging the **LambdaLoss** framework, which allows for the direct optimization of ranking metrics.

## Key Features

- **Direct Uplift Optimization:** Instead of using proxy metrics, UpliftNet directly optimizes for the ranking of individuals by their predicted uplift.
- **Neural Network-Based:** Built on top of TensorFlow and Keras, allowing for flexible and powerful model architectures.
- **Efficient Training:** Supports mini-batch training, which elegantly solves the exponential complexity of ranking that traditional methods like gradient boosting struggle with.
- **Easy to Use:** The `UpliftNet` model can be used as a drop-in replacement for `tf.keras.Model`, with a familiar API.
- **Comprehensive Evaluation:** Includes tools for evaluating uplift models, such as the Cumulative Gain Curve, and various plots to visualize model performance.

## How it Works

UpliftNet is based on the principles of learning-to-rank, adapted for uplift modeling. It directly optimizes a ranking of individuals based on their predicted uplift, rather than using proxy metrics.

This approach is heavily inspired by the following two papers:

1.  **[Learning to rank for uplift modeling](https://arxiv.org/pdf/2002.05897):** This paper introduces the **Promotional Cumulative Gain (PCG)** metric, which is designed to directly optimize the Area Under the Uplift Curve (AUUC).
2.  **[The LambdaLoss Framework for Ranking Metric Optimization](https://storage.googleapis.com/gweb-research2023-media/pubtools/4591.pdf):** This paper describes the **LambdaLoss** framework, which provides a method for creating differentiable approximations of ranking metrics, allowing them to be used as loss functions in neural networks.

UpliftNet combines these two ideas to create a neural network that can directly optimize for the PCG metric. This is achieved by using a custom Keras model that implements a differentiable version of the PCG loss function.

By using neural networks, UpliftNet offers several advantages over traditional methods:

-   **Mini-batch Training:** This allows for efficient training on large datasets.
-   **Flexibility:** Neural networks can model complex relationships in the data and can be easily customized.
-   **End-to-End Training:** The entire model can be trained in a single process.

## Installation

```sh
pip install upliftnet
```

## Usage

The `UpliftNet` model can be used in the same way as a standard `tf.keras.Model`. Here's a basic example:

```python
import tensorflow as tf
from upliftnet.net import UpliftNet
from upliftnet.data import generate_logistic_data

# 1. Generate some synthetic data
X, y, treatment = generate_logistic_data(
    treatment_coefs=[0, 1, 1, 0, 0],
    control_coefs=[0, 0, 0, 1, 1],
    n_treatment=1000,
    n_control=1000
)

# 2. Create a tf.data.Dataset
dataset = tf.data.Dataset.from_tensor_slices((X, y, treatment)).batch(256)

# 3. Define your model architecture
inputs = tf.keras.Input(shape=(X.shape[1],))
x = tf.keras.layers.Dense(64, activation='relu')(inputs)
x = tf.keras.layers.Dense(32, activation='relu')(x)
outputs = tf.keras.layers.Dense(1, activation='linear')(x)

model = UpliftNet(inputs=inputs, outputs=outputs)

# 4. Compile and train the model
model.compile(optimizer='adam')
model.fit(dataset, epochs=10)

# 5. Evaluate the model
# ...
```

For more detailed examples, please refer to the `examples` directory.

## Evaluation

The package provides several tools for evaluating the performance of your uplift model:

-   **Cumulative Gain Curve:** `upliftnet.metrics.cumulative_gain_curve` and `upliftnet.plots.cumulative_gain_plot`
-   **Area Under the Cumulative Gain Curve (AUC):** `upliftnet.metrics.cgc_auc`
-   **Target Plot:** `upliftnet.plots.target_plot`
-   **True Lift Plot:** `upliftnet.plots.true_lift_plot`
-   **Calibration Plot:** `upliftnet.plots.calibration_plot`

