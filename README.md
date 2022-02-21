# Explporation of Model Design and Model Generalization of Deep Newural Networks 

This model design choices of Deep Neural Networks is explored using the MNIST handwritten digits dataset, and then the generalization is tested on the fashion-MNIST dataset.
Deep learning is complicated by several design choices such as:
- Network Architecture: the number of layers, the type of layers (fully connected, convolutional, pooling, residual connections, etc.), and the activation functions
- Optimization: SGD, Adagrad, Adam, etc., minibatch size, number of epochs, the learning rate, etc.
- Initialization: initializing the network weights with independent normal or uniform draws, distributed with a pre-specified mean and variance
- Regularization: dropout, batch normalization, and weight decay, as well as implicit regularization such as early stopping or data augmentation
- Loss function:  in our case we will use cross-entropy loss for classification

These choices can have a substantial impact on the performance of trained neural network models. As a result, it is natural to ask how robust the models are to these design choices. Comparing the performance of these model design, we will choose the best model according to their test accuracy, and then test if the best model is still as robust on the fashion-MNIST dataset. We will investigate the relationship between model hyperparameters and test accuracy and training time using these two datasets.

## Dataset
1. MNIST hand-written digits data
  - 60,000 training set
  - 10,000 testing set
  - 10 labels (digits)
2. fashion-MNIST data
  - 60,000 training set
  - 10,000 testing set
  - 10 lables (pieces of clothing)
Validation split og 0.2 on testing data as velidation data

## Process
### 1. Baseline Model
The baseline model is a simple neural network with 2 hidden layers. We used the `tensorflow.keras` Sequential API to build the network. The baseline model is a fully connected network, using 2 dense layers of 16 units with ReLU activation as hidden layers, not including any regularization layers. We chose this model as our baseline, since while fully connected network is broadly applicable, both datasets are image datasets. CNNs usually perform better than a fully connected model on image recognition since CNNs can heavily reduce dimensionalities given the large number of parameters in an image. The input dimension is (28, 28, 1) and the output dimension is 10. The output would be a label from (0,
9), corresponding to the number.

### 2. Complex Models
We explore different network architectures, and several choices of hyperparameters.
For network choices, we avoid over-parametrization by incrementally adding a Conv2D layer, along with dense layers. Other regularization layers such as Batch Normalization, dropout, and max pooling were not included in deciding which model parameters(filter size, dense layer unit size). After comparing the number of hidden layers and model parameters, based on the best choice, we will also explore Adam with different learning rates, dropout percentages and regularization methods such as implementing batch normalization and maxpooling layers.

**Early Stopping**
We implemented early stopping since running all 100 epochs will be computationally expensive and leads to overfitting. In the baseline model above, it was evident that the model was overfitting since the validation accuracy was decreasing over time. To implement early stopping, we tested several patience parameters.

**Network Architecture**
Three different network architectures will be explored. For each network, we use 3 different number of convolution layers, 3 different filter sizes for the Conv2D layers, and 3 different unit sizes for the dense layer. The parameter choices are specified as follows:
- number of Conv2D layers [1,2,3]
- filter sizes: [16, 24, 32]
- dense layer unit sizes: [64, 128, 256]

After training 27 different models, we select the best model based on test accuracy and training time and compare regularization and optimizer parameters.

**Regularization**
- Dropout rates: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
- Batch Normalization: [1, 2, 3] layers
- Optimizer: Adam with learning rates [0.01, 0.001, 0.0001]
- Max-pooling

### 3. Test Best Model on fashion-MNIST Dataset
### 4. Re-run process on fashion-MNIST and compare best models for both datasets
