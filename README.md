# Neural Network from Scratch and MNIST Classification

### Introduction

In this project, I implemented a basic neural network from scratch using Python and applied it to the MNIST dataset to classify handwritten digits (0-9). The project is divided into three main parts:

1. **Building a Neural Network from Scratch:** Following Michael Nielsen’s "Neural Networks and Deep Learning" book, I implemented a simple feedforward neural network using core Python concepts.
2. **MNIST Classification:** I trained this neural network on the MNIST dataset and evaluated its performance on the test set.
3. **Comparison with PyTorch:** After building the network from scratch, I implemented the same model using the PyTorch framework and compared the performance and ease of use between both implementations.

### Project Structure
```
src/
├── mnist_loader.py         # Code to load MNIST dataset from mnist.pkl.gz
├── network.py              # Neural network implementation from scratch
├── pytorch_mnist.py        # PyTorch implementation of the neural network
data/
└── mnist.pkl.gz            # MNIST dataset in compressed pickle format
README.md                   # This file

```
### Files Explained

- mnist_loader.py: Contains utility functions for loading the MNIST dataset.
- network.py: Implements a neural network from scratch, including feedforward, backpropagation, and stochastic gradient descent.
- pytorch_mnist.py: Implements the equivalent neural network using the PyTorch library.
- mnist.pkl.gz: The MNIST dataset compressed into a gzip file, containing 60,000 training examples and 10,000 test examples.

## 1. Neural Network from Scratch

### Overview

The neural network is built based on the concepts from Michael Nielsen's "Neural Networks and Deep Learning," specifically Chapter 1. The key components of the implementation include:

- **Feedforward:** A process of passing inputs through the network to generate outputs.
- **Backpropagation:** An algorithm for updating the weights and biases of the network using gradient descent.
- **Stochastic Gradient Descent:** The optimization algorithm used to minimize the cost function by iteratively adjusting weights and biases.

### Key Concepts

- **Feedforward:** The input data is passed through layers of neurons to produce an output, typically in the form of probabilities for each digit (0-9).
- **Backpropagation:** The gradient of the loss function is calculated with respect to each weight and bias in the network, allowing the network to learn from its errors.
- **Gradient Descent:** This is used to update the parameters (weights and biases) of the network in the direction that minimizes the loss.

### Training Process

The model is trained using the MNIST dataset. The network is initialized with random weights and biases and trained over multiple epochs. The loss is computed using the cross-entropy function, and the weights and biases are updated using backpropagation and stochastic gradient descent.

***Hyperparameters***

- Learning rate: 3.0
- Number of epochs: 30
- Mini-batch size: 10

### Results

``` 
>>> import mnist_loader
>>> import network
>>> training_data, validation_data, test_data =mnist_loader.load_data_wrapper()
>>> net = network.Network([784, 30, 10])
>>> net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
Epoch 0: 8042 / 10000 (80.42% accuracy), took 7.13 seconds
Epoch 1: 8108 / 10000 (81.08% accuracy), took 7.07 seconds
...
...
...
Epoch 28: 8401 / 10000 (84.01% accuracy), took 11.33 seconds
Epoch 29: 8405 / 10000 (84.05% accuracy), took 11.48 seconds
```
- The network was able to classify MNIST digits with 84.05% accuracy after 30 epochs.
- Each epoch initially took around 7 seconds but increased to around 11 seconds by the final epochs. This slowdown is typical as the model continues learning and as more computations are required in later stages of training.

## 2. PyTorch Implementation

### Overview

After implementing the neural network from scratch, I replicated the same architecture using PyTorch, a popular deep learning framework. PyTorch simplifies many aspects of neural network construction and training, such as automatic differentiation, GPU acceleration, and model optimization.

### Model Architecture in PyTorch

The model architecture in PyTorch is similar to the scratch implementation:

- Input layer: 784 neurons (28x28 pixels of the MNIST images).
- Hidden layer: 30 neurons with ReLU activation.
- Output layer: 10 neurons corresponding to the digits 0-9.

### Training Process

PyTorch abstracts much of the low-level details (e.g., backpropagation) and provides built-in optimizers and loss functions, making the process more streamlined compared to the scratch implementation.

### Hyperparameters

- Learning rate: 0.5
- Number of epochs: 30
- Mini-batch size: 10
- Optimizer: SGD (Stochastic Gradient Descent)
- Loss function: CrossEntropyLoss

### Results
```
Epoch 1, Loss: 0.7328029925952843, Train Acc: 75.41%, Val Acc: 88.00%
Epoch 2, Loss: 0.39517677256039213, Train Acc: 87.47%, Val Acc: 91.59%
Epoch 3, Loss: 0.32325061021456075, Train Acc: 89.89%, Val Acc: 92.48%
Epoch 4, Loss: 0.28194762504240595, Train Acc: 91.11%, Val Acc: 93.38%
Epoch 5, Loss: 0.25238258407504827, Train Acc: 92.01%, Val Acc: 94.18%
Epoch 6, Loss: 0.2359788905320836, Train Acc: 92.56%, Val Acc: 94.44%
Epoch 7, Loss: 0.2221789840600892, Train Acc: 92.96%, Val Acc: 94.46%
Epoch 8, Loss: 0.2154833152056185, Train Acc: 93.17%, Val Acc: 95.25%
Epoch 9, Loss: 0.20774746079768328, Train Acc: 93.56%, Val Acc: 95.34%
Epoch 10, Loss: 0.19728110657373407, Train Acc: 93.93%, Val Acc: 95.15%
/Users/kareemaayman/task13.2/src/pytorch_mnist.py:107: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load('best_model.pth'))
Test Accuracy: 95.36%
```
- The network was able to classify MNIST digits with 93.93% train accuracy, 95.25% validation accuracy, and 95.36% test accuracy after only 10 epochs.

- Although the original plan was to train for 30 epochs, the training was stopped after 10 epochs because:

	- Sufficient Performance: By the 10th epoch, the model had already achieved a high test accuracy of 95.36%, and further improvements were expected to be minimal.
	- Avoiding Overfitting: Training for more epochs might have led to overfitting, where the model performs well on the training set but not as well on unseen data (the test set).
	- Time Considerations: Given the high performance at 10 epochs, continuing to train for 30 epochs would have increased the training time without a significant boost in accuracy.
	
## 3. Comparison Between Scratch and PyTorch Implementation

### Ease of Use
- From Scratch: Building the neural network from scratch allowed for a deeper understanding of how neural networks function. However, it required manually implementing feedforward propagation, backpropagation, gradient descent, and managing weight updates. While it was a valuable learning experience, the amount of code and complexity increased significantly as compared to using a deep learning library.
- PyTorch: PyTorch makes building and training models much easier by abstracting away these complexities. With just a few lines of code, you can define, train, and evaluate models.

### Performance
- Scratch Implementation: The neural network built from scratch achieved a test accuracy of 84.05% after 30 epochs. This performance, while decent, is lower than the PyTorch implementation. The scratch implementation also took more time to train, with each epoch taking around 7 seconds initially and increasing to over 11 seconds by the end.
- PyTorch Implementation: The PyTorch model achieved a test accuracy of 95.36% after 10 epochs. PyTorch makes use of better initialization schemes, optimizers, and faster backpropagation techniques, leading to improved performance.

### Flexibility
- From Scratch: Custom implementations allow for more flexibility in experimenting with different techniques (e.g., custom optimization methods, activation functions). However, this comes at the cost of more code and complexity.
- PyTorch: PyTorch provides a lot of flexibility while maintaining ease of use. You can still experiment with custom architectures and optimization methods, but with significantly less effort compared to scratch implementations.

### Summary of Results
| Implementation | Final Accuracy | Training Time per Epoch |
| -------------- | -------------- | ----------------------- |
| From Scratch   | 84.05% (after 30 epochs) | ~7 seconds initially, increasing to ~11 seconds by epoch 30 |
| -------------- | -------------- | ----------------------- |
| PyTorch        | 95.36% (after 10 epochs) | ~5-7 seconds per epoch |

### Key Takeaways
- The PyTorch implementation achieved higher accuracy in fewer epochs and required less code.
- Building a network from scratch was useful for understanding the fundamentals of neural networks, but PyTorch provided much better performance and ease of use.
- The scratch implementation is more suited for learning purposes, whereas PyTorch is highly efficient for real-world applications.

## Conclusion

In this project, I implemented a simple neural network from scratch to classify handwritten digits using the MNIST dataset. The scratch implementation helped solidify my understanding of key concepts like feedforward propagation, backpropagation, and stochastic gradient descent. I then implemented the same network using PyTorch and observed a significant improvement in both performance and ease of use.

While the scratch implementation achieved an accuracy of 84.05% after 30 epochs, the PyTorch model reached 95.36% accuracy in just 10 epochs. This demonstrates the power of deep learning libraries like PyTorch in simplifying model building and training while offering optimized performance.

## Next Steps
- Experiment with adding more layers or neurons to the network.
- Introduce techniques like regularization, dropout, or batch normalization to improve generalization.
- Use GPU acceleration with PyTorch for faster training.
- Explore convolutional neural networks (CNNs) for better performance on image data.

## How to Run
### Prerequisites
- Python 3.x
- Install required packages (for PyTorch implementation):
``` pip install torch torchvision ```

### Running the Neural Network from Scratch
To train and evaluate the neural network built from scratch, run the following commands:
```
python3
>>> import mnist_loader
>>> import network
>>> training_data, validation_data, test_data =mnist_loader.load_data_wrapper()
>>> net = network.Network([784, 30, 10])
>>> net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
```
This will load the MNIST dataset, train the network for 30 epochs, and display the accuracy after each epoch.

### Running the PyTorch Implementation
To train and evaluate the neural network built using PyTorch, run the following command:
```
cd task13.2/src
python3 pytorch_mnist.py
```
This script will automatically download the MNIST dataset (if not already downloaded), train the model, and display the training accuracy and validation accuracy after each epoch. After training, it will report the final test accuracy.
