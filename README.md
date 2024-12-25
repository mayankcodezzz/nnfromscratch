# nnfromscratch
# **Creating a neural network from scratch using numpy to classify 2x2 matrices**
![Screenshot (18)](https://github.com/user-attachments/assets/c8de4b5e-59c7-49fd-acf6-22b4d2d48927)

# Problem Understanding
We have 2x2 matrices where each cell is black (1) or white (0). The possible classifications are:

Solid: All black or all white.

Vertical: Two columns are consistent (e.g., [[1, 0], [1, 0]]).

Horizontal: Two rows are consistent (e.g., [[1, 1], [0, 0]]).

Diagonal: Diagonal patterns (e.g., [[1, 0], [0, 1]]).

The input size is 4 (flattened 2x2 matrix), and the output size is 4 (one-hot encoded classification).

# Step-by-Step Plan
**Step 2.1: Data Representation**<br/>
Flatten the 2x2 matrix into a 1D array for simplicity. Each matrix will be represented as [x1, x2, x3, x4].<br/>

**Step 2.2: Initialize Parameters**<br/>
We'll use a single hidden layer neural network:<br/>
* Input layer: 4 neurons (for the flattened matrix).<br/>
* Hidden layer: Choose, say, 8 neurons with ReLU activation.<br/>
* Output layer: 4 neurons (Softmax for classification).<br/>

Weights and biases are randomly initialized.<br/>

**Step 2.3: Forward Propagation**<br/>
* Compute the activation of the hidden layer using ```Z1 = W1.X + b1``` and ```A1 = ReLU(Z1)```.<br/>
* Compute the output layer using ```Z2 = W2.A1 + b2``` and ```A2 = Softmax(Z2)```.<br/>

**Step 2.4: Loss Function**<br/>
Use cross-entropy loss:![Screenshot (17)](https://github.com/user-attachments/assets/e2c0f198-60d3-46e0-9f58-d2978ff0578e)
<br/>

**Step 2.5: Backward Propagation**<br/>
* Compute gradients for output and hidden layers.<br/>
* Update weights and biases using gradient descent.<br/>

**Step 2.6: Train the Model**<br/>
* Train the network on a set of examples.<br/>
* Monitor loss and accuracy during training.<br/>

### ***First we will import numpy to work with arrays in python***

### ***Creating an Activation Function***
This is a ReLU (Rectified Linear Unit) function. It takes a number and:
* If the number is negative, it makes it 0.
* If it’s positive, it keeps it as is.

## **The function relu_derivative(x) computes the derivative of the ReLU (Rectified Linear Unit) activation function.**

Explanation:<br/>
**1. ReLU Activation Function:**<br/>
   The ReLU function is defined as:<br/>
   `f(x)= x ​ if x>0  ​`<br/>
   `f(x)= 0 ​ if x≤0  ​`<br/>
 Its derivative is:<br/>
   `f'(x)= 1 ​ if x>0  ​`<br/>
   `f'(x)= 0 ​ if x≤0  ​`<br/>
What the function does:<br/>

* The expression (x > 0) creates a boolean array where each element is True if the corresponding element of x is greater than 0 and False otherwise.
* .astype(float) converts this boolean array to a float array, where True becomes 1.0 and False becomes 0.0.
* Thus, the function returns 1.0 for elements of x that are greater than 0 and 0.0 for elements less than or equal to 0, which corresponds to the derivative of the ReLU function.

### ***Make the numbers stable***
1. `np.max(x, axis=0, keepdims=True)`<br/>
* We find the biggest number in the list `x`.
* This helps to keep the math safe and avoid super big numbers (it’s called a "stability fix").
* Example: If `x = [2, 3, 5]`, the biggest number is `5`.

2. `x - np.max(x, axis=0, keepdims=True)`<br/>
* Subtract the biggest number from every number in `x`.
* Example: `[2, 3, 5] - 5` becomes `[-3, -2, 0]`.

3. `np.exp(...)`
* Now we take the exponential (a fancy math operation) of the adjusted numbers.
* Example: `np.exp([-3, -2, 0])` becomes `[0.05, 0.14, 1]`.

### ***Turn the numbers into probabilities***
1. `np.sum(exp_x, axis=0, keepdims=True)`
* Add up all the new numbers.
* Example: `[0.05, 0.14, 1]` adds up to `1.19`.

2. `exp_x / np.sum(...)`
* Divide each number by the total to make them into probabilities.
* Example:
   * 0.05 / 1.19 ≈ 0.04
   * 0.14 / 1.19 ≈ 0.12
   * 1 / 1.19 ≈ 0.84.

* Now you have probabilities: `[0.04, 0.12, 0.84]`.

## ***LOSS FUNCTION***
### ***1.Get the number of samples***<br/>

* We’re counting how many examples (or data points) we’re working with.
* `y_true.shape[1]`: The shape gives the dimensions of y_true.<br/>
 * If y_true is a matrix, the second number `(1)` tells us how many samples we have.
* Example: If y_true is a matrix with shape `(3, 5)`, it means there are `5` samples.
### ***2.Calculate the loss***<br/>
1. `np.log(y_pred)`
* Take the natural logarithm of each predicted probability in y_pred.
* This step is important for how cross-entropy works mathematically.
2. `y_true * np.log(y_pred)`
* Multiply the true labels `(y_true)` with the logarithm of the predicted probabilities `(np.log(y_pred))`.
* This ensures we only consider the predictions for the correct labels.
* Example: If `y_true = [1, 0, 0]` and `y_pred = [0.7, 0.2, 0.1]`, only 0.7 (the probability of the correct label) is used.
3. `np.sum(...)`
* Add up all the values from the previous step for all samples.
* Example: If you have predictions for 3 samples, you’ll sum the contributions from all 3.
4. `- (negative sign)`
* Cross-entropy involves taking the negative of the sum. This makes the loss a positive value.
5. `/ m`
* Divide by the number of samples `(m)` to get the average loss per sample.

# ***Initialization***
### 1. **Define the function**<br/>
* We’re defining a function called `initialize_parameters`.
It takes three inputs:
  * `input_size`: How many features go into the network (number of input neurons).
  * `hidden_size`: How many neurons are in the hidden layer.
  * `output_size`: How many outputs the network produces (number of output neurons).<br/>

### 2. **Set a random seed**<br/>
* `np.random.seed(42)`
* Setting a random seed ensures that the random numbers generated are always the same every time you run the code. This helps with reproducibility, so results don’t vary randomly.

### 3. **Initialize weights and biases**<br/>
## Weights for Layer 1 (𝑊1)
## `W1 = np.random.randn(hidden_size, input_size) * 0.01`
* np.random.randn(hidden_size, input_size) generates a random matrix of size `hidden_size x input_size` with values from a standard normal distribution (mean = 0, standard deviation = 1).
* Multiplying by 0.01 scales these values down to make them small. This helps the network start learning without large gradients that might destabilize training.
* Example: If hidden_size = 3 and input_size = 2, then
W1 will be a 3×2 matrix.


## Biases for Layer 1 (b1)
## `b1 = np.zeros((hidden_size, 1))`
* `np.zeros((hidden_size, 1))` creates a matrix of zeros with `hidden_size x 1` dimensions.
* Biases are initialized to zero because they don’t need random starting values.

## **Similarly we do it for layer 2 (weight and biase)**
## `W2 = np.random.randn(output_size, hidden_size) * 0.01`
## `b2 = np.zeros((output_size, 1))`

# ***Forward Propagation***
* `Z1 = np.dot(W1, X) + b1`: Multiply inputs (X) by weights (W1) and add biases (b1). This gives the first hidden layer's signals.
* `A1 = relu(Z1)`: Apply ReLU to make these signals cleaner.
* `Z2 = np.dot(W2, A1) + b2`: Multiply hidden layer signals (A1) by weights (W2) and add biases (b2). This gives the final layer's signals.
* `A2 = softmax(Z2)`: Apply Softmax to turn final signals into probabilities for classification.

# ***Backward Propagation***
* dZ2: Difference between the robot’s guess and the true answer.
* dW2: Adjustments for the final layer weights.
* db2: Adjustments for the final layer biases.
* dA1: Feedback to the first layer.
* dZ1: Adjustments for the first layer.
* dW1, db1: Adjustments for the first layer weights and biases.

# ***Training the Model***

* initialize_parameters: Give the neurons their starting rules.
* forward_propagation: Make guesses.
* cross_entropy_loss: Measure how wrong the guesses are.
* backward_propagation: Calculate how to improve.
* update_parameters: Teach the neurons better rules.
* Repeat for a set number of iterations.

# ***Training the Model***

* initialize_parameters: Give the neurons their starting rules.
* forward_propagation: Make guesses.
* cross_entropy_loss: Measure how wrong the guesses are.
* backward_propagation: Calculate how to improve.
* update_parameters: Teach the neurons better rules.
* Repeat for a set number of iterations.

