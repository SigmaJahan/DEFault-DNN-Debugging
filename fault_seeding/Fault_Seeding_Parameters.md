## Fault Seeding Parameters for DEFault
A comprehensive guide defining mutation parameters and ranges for systematic fault injection in neural networks.

### Layer Selection Criteria

#### Layer-based Operations:
* Change Activation Function: Only layers with non-linear activation functions
* Add Activation Function: Only layers with linear activation functions
* Add Weights Regularization: Only layers without weights regularization
* Remove Weights Regularization: Only layers with existing weights regularization
* Change Dropout Rate: Only dropout layers
* Layer Type Swap: Only LSTM and GRU layers
* Kernel Size Change: Only convolutional layers
* Filter Size Change: Only convolutional layers
* Pooling Size Change: Only pooling layers
* Stride Size Change: Only strided layers
* Padding Change: Only padded layers
* Layer Unit Modification: All parametrized layers

### Parameter Values

#### 1. CNN-specific Parameters:
* Kernel sizes: (x,x) where x ∈ [1, 7]
* Filter sizes: [1, 32]
* Pooling sizes: (x,x) where x ∈ [1, 5]
* Stride sizes: (x,x) where x ∈ [1, 5]
* Padding types: ['valid', 'same', 'causal']
* Neuron counts: [1, 256]

#### 2. Range-based Parameters Using Binary Search:

**Hyperparameter Operations:**
* Change Number of Epochs: [1, original_epochs]
* Decrease Learning Rate: [≈0, original_learning_rate]

**Regularization Operation:**
* Change Patience Parameter: [1, original_patience]

#### 3. List-based Parameters Using Exhaustive Search:

**Activation Function Options:**
Exponential Linear Unit (ELU), SoftMax, Scaled Exponential Linear Unit (SELU), SoftPlus, SoftSign, Rectified Linear Unit (ReLU), Hyperbolic Tangent (TanH), Sigmoid, Hard Sigmoid, Exponential, Linear

**Loss Function Options:**
Mean Squared Error (MSE), Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), Mean Squared Logarithmic Error (MSLE), Squared Hinge, Hinge, Categorical Hinge, Log-Cosh, Huber Loss, Categorical Cross-Entropy, Binary Cross-Entropy, Kullback-Leibler Divergence, Poisson

**Optimization Options:**
Stochastic Gradient Descent (SGD), RMSprop (Root Mean Square Propagation), AdaGrad (Adaptive Gradient Algorithm), Adam (Adaptive Moment Estimation), AdaMax, NAdam (Nesterov-accelerated Adaptive Moment Estimation)

**Weight Options:**

*Weight Initializers:*
Zeros, Ones, Constant, Random Normal, Random Uniform, Truncated Normal, Orthogonal, LeCun Uniform, Glorot Normal (Xavier Normal), Glorot Uniform (Xavier Uniform), He Normal, LeCun Normal, He Uniform

*Weight Regularizers:*
L1 (Lasso Regularization), L2 (Ridge Regularization), L1-L2 (Elastic Net Regularization)

#### 4. Specific Value Sets:

**Batch Size Operation:**
* Change Batch Size: [16, 512] with step size 16

**Dropout Rate Operation:**
* Change Dropout Rate: [0, original_learning_rate]

**Layer Insertion Options:**
* Dense Units: [32, 128] with step size of 16
* Dropout Rate: [0, original_learning_rate]
* Activation Functions: Exponential Linear Unit (ELU), SoftMax, Scaled Exponential Linear Unit (SELU), SoftPlus, SoftSign, Rectified Linear Unit (ReLU), Hyperbolic Tangent (TanH), Sigmoid, Hard Sigmoid, Exponential, Linear

#### No Parameters Required:
Remove Bias from Layer, Add Bias to Layer, Remove Activation Function, Layer Removal, Layer Type Swapping (LSTM/GRU)
