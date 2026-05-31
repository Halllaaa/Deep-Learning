# Beginner Full Explanation - Part I and Part II

This document explains Part I and Part II from the beginning, for someone who is new to deep learning and PyTorch.

The goal is not only to know that the code works, but to understand what the project is doing, why we made these choices, and how to explain the results in the final report.

---

# 1. How Many Data Elements Did We Use?

## Part I - Tabular Data

Part I uses the Breast Cancer Wisconsin dataset.

This is not an image dataset. Each element is one row/sample/patient observation.

Total dataset size:

```text
569 samples
```

In our code, the split was:

```text
397 training samples
86 validation samples
86 test samples
```

So:

```text
397 + 86 + 86 = 569 total samples
```

Each sample has:

```text
30 numerical features
```

These features describe properties of cell nuclei, such as radius, texture, perimeter, area, smoothness, and other measurements.

The target has 2 classes:

```text
malignant
benign
```

So Part I is:

```text
569 tabular samples, 30 features per sample, 2 classes
```

## Part II - Image Data

Part II uses Fashion-MNIST.

Fashion-MNIST contains grayscale images of clothes and fashion items.

The complete Fashion-MNIST dataset has:

```text
60,000 training images
10,000 test images
70,000 total images
```

But to make the script run faster on your laptop CPU, we used a smaller subset:

```text
4,000 training images
1,000 validation images
1,000 test images
```

So our experiment used:

```text
6,000 images total
```

Each image has:

```text
1 channel
28 x 28 pixels
```

Because the images are grayscale, they have 1 channel. A color image would usually have 3 channels: red, green, and blue.

Each image has:

```text
28 * 28 = 784 pixel values
```

The target has 10 classes:

```text
T-shirt/top
Trouser
Pullover
Dress
Coat
Sandal
Shirt
Sneaker
Bag
Ankle boot
```

So Part II is:

```text
6,000 images used in our run, 28x28 pixels per image, 10 classes
```

---

# 2. What Is Deep Learning?

Deep learning is a type of machine learning that uses neural networks.

A neural network learns a function that maps input data to an output prediction.

For example:

```text
Input: 30 medical measurements
Output: malignant or benign
```

or:

```text
Input: image of clothing
Output: class such as sneaker, shirt, bag, etc.
```

The network starts with random weights. During training, it changes these weights so that predictions become more correct.

---

# 3. What Is Supervised Learning?

Both Part I and Part II use supervised learning.

Supervised learning means that every training example has:

```text
input data + correct answer
```

Example from Part I:

```text
Input: 30 numerical features
Correct answer: benign
```

Example from Part II:

```text
Input: 28x28 image
Correct answer: sneaker
```

The model predicts an answer. Then we compare the prediction with the correct answer. The difference is called the loss.

---

# 4. Important Words You Must Know

## Dataset

A dataset is the full collection of examples used in the project.

Part I dataset:

```text
Breast Cancer Wisconsin
```

Part II dataset:

```text
Fashion-MNIST
```

## Sample

A sample is one element from the dataset.

In Part I, one sample is one tabular row.

In Part II, one sample is one image.

## Feature

A feature is an input variable.

In Part I, each sample has 30 features.

Example:

```text
mean radius
mean texture
mean perimeter
```

In Part II, the features are pixel values.

## Label

A label is the correct class.

Example:

```text
benign
malignant
```

or:

```text
shirt
sneaker
bag
```

## Model

A model is the neural network that learns from data.

## Parameters

Parameters are the learnable values inside the model.

The most common parameters are:

```text
weights
biases
```

Training changes these parameters.

## Epoch

One epoch means the model has gone through the full training set once.

Example:

```text
3 epochs = the model sees the training data 3 times
```

## Batch

Instead of giving all data to the model at once, we divide it into small groups called batches.

Example:

```text
batch size = 32
```

means the model sees 32 samples at a time.

## Loss

Loss is the error of the model.

Low loss is good.

High loss is bad.

During training, we want the loss to decrease.

## Accuracy

Accuracy means:

```text
number of correct predictions / total number of predictions
```

If accuracy is 0.82, that means:

```text
82% of predictions were correct
```

## Precision

Precision answers:

```text
When the model predicts a class, how often is it correct?
```

## Recall

Recall answers:

```text
Of all real examples of a class, how many did the model find?
```

## F1-score

F1-score combines precision and recall.

It is useful when we want one balanced metric.

## Confusion Matrix

A confusion matrix shows correct and incorrect predictions for each class.

It helps us see which classes are confused with each other.

---

# 5. Part I - MLP For Tabular Data

## 5.1 What Is The Task?

Part I asks us to classify tabular data using an MLP.

Our dataset:

```text
Breast Cancer Wisconsin
```

Goal:

```text
Predict whether a tumor is malignant or benign
```

Input:

```text
30 numerical features
```

Output:

```text
2 classes
```

## 5.2 What Is Tabular Data?

Tabular data is data organized like a table.

Example:

```text
sample | feature 1 | feature 2 | feature 3 | target
1      | 12.3      | 15.2      | 78.1      | benign
2      | 19.1      | 22.4      | 120.5     | malignant
```

Each row is one sample.

Each column is one feature.

## 5.3 What Is An MLP?

MLP means Multi-Layer Perceptron.

It is a neural network made of fully connected layers.

Fully connected means each neuron in one layer is connected to every neuron in the next layer.

Our MLP structure:

```text
30 input features
-> Linear layer
-> ReLU
-> Dropout
-> Linear layer
-> ReLU
-> Output layer with 2 values
```

The final output has 2 values because there are 2 classes:

```text
malignant
benign
```

## 5.4 Why Normalize The Data?

The 30 features do not all have the same scale.

Some values may be small, others large.

A neural network trains better when features are on a similar scale.

So we used:

```python
StandardScaler()
```

It transforms features so that they have approximately:

```text
mean = 0
standard deviation = 1
```

Important rule:

We fit the scaler only on the training data.

Then we transform validation and test data using the same scaler.

Why?

Because validation and test data must stay unseen. We should not use them to learn preprocessing parameters.

## 5.5 Why Train, Validation, Test?

We split data into three parts:

```text
training set
validation set
test set
```

Training set:

```text
Used to update model weights
```

Validation set:

```text
Used to compare models during experimentation
```

Test set:

```text
Used only at the end for final evaluation
```

Our Part I split:

```text
397 train
86 validation
86 test
```

## 5.6 What Does PyTorch Do?

PyTorch is the library used to build and train neural networks.

Important PyTorch objects:

```text
Tensor
nn.Module
DataLoader
Loss function
Optimizer
state_dict
```

## 5.7 What Is A Tensor?

A tensor is like an array or matrix used by PyTorch.

Example:

```text
one sample with 30 features -> tensor of shape (30,)
batch of 32 samples -> tensor of shape (32, 30)
```

## 5.8 What Is nn.Module?

`nn.Module` is the base class for PyTorch models.

When we create a custom model, we write:

```python
class CustomMLP(nn.Module):
```

Inside the class, we define layers and the forward pass.

## 5.9 What Is nn.Sequential?

`nn.Sequential` is a simple way to stack layers one after another.

Example:

```python
nn.Sequential(
    nn.Linear(30, 64),
    nn.ReLU(),
    nn.Linear(64, 2)
)
```

It is clean when the model is simple.

## 5.10 Difference Between nn.Sequential And Custom Class

`nn.Sequential`:

```text
simple, short, good for straight models
```

Custom `nn.Module`:

```text
more flexible, better for complex models
```

The project asks for both, so we implemented both.

## 5.11 What Is ReLU?

ReLU is an activation function.

Formula:

```text
ReLU(x) = max(0, x)
```

It allows the network to learn nonlinear relationships.

Without activation functions, many layers would behave almost like one linear layer.

## 5.12 What Is Dropout?

Dropout randomly disables some neurons during training.

It helps reduce overfitting.

Overfitting means the model memorizes the training data but performs badly on new data.

## 5.13 What Is CrossEntropyLoss?

Cross-entropy loss is used for classification.

Our model outputs raw scores called logits.

CrossEntropyLoss compares these logits with the true class labels.

Lower cross-entropy means better predictions.

## 5.14 What Is Adam?

Adam is an optimizer.

The optimizer updates the model weights after backpropagation.

In simple terms:

```text
loss tells us how wrong the model is
gradients tell us how to change the weights
optimizer performs the change
```

## 5.15 What Is Backpropagation?

Backpropagation computes how much each parameter contributed to the error.

In code:

```python
loss.backward()
```

This calculates gradients.

Then:

```python
optimizer.step()
```

updates the weights.

## 5.16 What Is Initialization?

Initialization means choosing the starting values of model weights.

We tested:

```text
Gaussian initialization
constant initialization
Xavier initialization
```

Gaussian:

```text
weights start with random values from a normal distribution
```

Constant:

```text
all weights start with the same value
```

Xavier:

```text
weights are initialized to keep signal variance stable between layers
```

Constant initialization is usually weaker because neurons start too similarly.

## 5.17 What Is state_dict?

`state_dict` is a dictionary containing model weights and biases.

It is what we save when we want to store a trained model.

In code:

```python
torch.save(...)
```

To reload:

```python
model.load_state_dict(...)
```

This satisfies the assignment requirement to save and reload the best model.

## 5.18 Part I Results

The best Part I results were:

```text
custom + gaussian:
accuracy = 0.9419
F1-score = 0.9524

sequential + xavier:
accuracy = 0.9419
F1-score = 0.9524
```

Accuracy 0.9419 means:

```text
about 94.19% of test samples were classified correctly
```

The test set had 86 samples.

So about:

```text
81 out of 86 samples were correct
```

## 5.19 Part I Interpretation

The MLP worked well on this dataset because the input is already a vector of numerical features.

The best models reached around 94% accuracy.

The constant initialization performed slightly worse, which shows that initialization affects learning.

The conclusion:

```text
A well-configured MLP can be effective for tabular classification, but it depends on normalization, architecture, initialization, and dataset size.
```

---

# 6. Part II - CNN For Images

## 6.1 What Is The Task?

Part II asks us to classify images using CNNs.

Our dataset:

```text
Fashion-MNIST
```

Goal:

```text
Predict the clothing class from an image
```

Input:

```text
1 grayscale image of size 28 x 28
```

Output:

```text
10 classes
```

## 6.2 What Is An Image For A Neural Network?

A grayscale image is a matrix of pixel values.

Fashion-MNIST image shape:

```text
1 x 28 x 28
```

This means:

```text
1 channel
28 rows
28 columns
```

Each pixel is a number.

Dark pixels have smaller values.

Bright pixels have larger values.

## 6.3 Why Is An MLP Limited For Images?

An MLP needs a vector.

So for Fashion-MNIST:

```text
28 x 28 image -> 784 values
```

This operation is called flattening.

Problem:

```text
flattening destroys spatial structure
```

The model no longer naturally knows which pixels were next to each other.

Images depend strongly on local patterns:

```text
edges
corners
textures
shapes
```

An MLP can learn from images, but it does not have a built-in image structure.

## 6.4 What Is A CNN?

CNN means Convolutional Neural Network.

A CNN is designed for images.

It uses convolution filters that slide over the image.

Instead of looking at all pixels at once, each filter looks at a small local region.

This is useful because visual patterns are local.

## 6.5 Three Big Ideas Of CNNs

## Locality

Pixels near each other are related.

A CNN uses small filters like:

```text
3 x 3
5 x 5
```

This lets the model learn local patterns.

## Weight Sharing

The same filter is applied across the whole image.

This means the model can detect the same pattern anywhere.

Example:

```text
a vertical edge can appear on the left, center, or right of the image
```

The same filter can detect it everywhere.

## Hierarchy

Early CNN layers learn simple patterns.

Examples:

```text
edges
corners
dark/light transitions
```

Deeper layers combine simple patterns into more complex shapes.

## 6.6 What Is Cross-Correlation?

In CNNs, the main operation is often called convolution, but technically many libraries perform cross-correlation.

The idea:

1. Take a small filter.
2. Put it over a patch of the image.
3. Multiply filter values by image patch values.
4. Add the results.
5. Move the filter to the next position.

This produces a feature map.

## 6.7 What Is A Feature Map?

A feature map is the output of a convolution filter.

It shows where a certain pattern was detected in the image.

Example:

```text
one filter may activate strongly on vertical edges
another may activate on dark regions
another may activate on curves
```

## 6.8 What Is Padding?

Padding adds extra pixels around the image border.

Example:

```text
padding = 1
```

adds one border around the image.

Why use padding?

```text
to preserve spatial size
to avoid losing border information too quickly
```

## 6.9 What Is Stride?

Stride is how far the filter moves each step.

Stride 1:

```text
move one pixel at a time
```

Stride 2:

```text
move two pixels at a time
```

Bigger stride reduces the output size faster.

This can make training faster but may lose information.

## 6.10 Output Size Formula

For convolution:

```text
output = floor((input + 2 * padding - kernel) / stride) + 1
```

Examples:

```text
28x28 input, kernel 5, padding 0, stride 1 -> 24x24 output
28x28 input, kernel 5, padding 2, stride 1 -> 28x28 output
28x28 input, kernel 3, padding 1, stride 2 -> 14x14 output
```

This is one of the theory requirements in the assignment.

## 6.11 What Is Pooling?

Pooling reduces spatial size.

Max-pooling:

```text
takes the maximum value in a region
```

Average-pooling:

```text
takes the average value in a region
```

Pooling helps reduce computation and makes the model more robust to small shifts.

## 6.12 What Is A 1x1 Convolution?

A 1x1 convolution uses a filter of size:

```text
1 x 1
```

It does not look at neighboring pixels.

Instead, it mixes information across channels.

It is useful for:

```text
combining feature maps
changing number of channels
adding non-linearity with low cost
```

## 6.13 What Models Did We Test?

We tested:

```text
mlp_flattened_images
cnn_lenet_max_padding
cnn_avg_pooling
cnn_more_filters
cnn_with_1x1
cnn_strided
```

## 6.14 What Is The Image MLP?

The image MLP first flattens the image:

```text
1 x 28 x 28 -> 784
```

Then it uses fully connected layers.

This is the baseline model.

Baseline means:

```text
a simple model used for comparison
```

## 6.15 What Is LeNet?

LeNet is one of the classic CNN architectures.

It follows this idea:

```text
convolution
activation
pooling
convolution
activation
pooling
fully connected classifier
```

Our CNN is inspired by LeNet.

## 6.16 Why Compare CNN Variants?

The assignment asks us to study different architectural choices.

So we tested:

```text
max-pooling vs average-pooling
more filters
1x1 convolution
stride-based convolution
padding
```

This lets us explain how architecture affects performance.

## 6.17 Part II Results

The final results were:

```text
cnn_strided:
accuracy = 0.819
macro F1 = 0.822

mlp_flattened_images:
accuracy = 0.805
macro F1 = 0.806

cnn_lenet_max_padding:
accuracy = 0.794
macro F1 = 0.801

cnn_with_1x1:
accuracy = 0.791
macro F1 = 0.798

cnn_more_filters:
accuracy = 0.773
macro F1 = 0.782

cnn_avg_pooling:
accuracy = 0.765
macro F1 = 0.766
```

The best model was:

```text
cnn_strided
```

It reached:

```text
81.9% test accuracy
82.2% macro F1-score
```

## 6.18 How To Interpret Part II Results

The best CNN beat the MLP:

```text
CNN strided accuracy = 81.9%
MLP accuracy = 80.5%
```

The difference is not huge, but it supports the theory that CNNs are better suited to images.

Why is the difference not very large?

Because we used:

```text
only 4,000 training images
only a few epochs
CPU training
simple CNN architectures
```

With the full dataset and more epochs, CNNs usually improve more clearly.

## 6.19 Common Confusions In Fashion-MNIST

Some classes are visually similar.

For example:

```text
shirt, T-shirt, coat, pullover
```

These classes can be confused because their shapes are close.

Other classes are easier:

```text
trouser
bag
sandal
ankle boot
```

They have more distinctive shapes.

## 6.20 Feature Map Interpretation

The script saves feature map visualizations.

These show internal activations inside the CNN.

They help us see that the CNN transforms the image into learned representations.

Early feature maps may highlight:

```text
edges
dark regions
shape borders
texture patterns
```

This is useful for the report because the assignment asks to visualize and interpret feature maps.

---

# 7. How To Read The Terminal Output

When you run the script, you see something like:

```text
epoch=01 train_loss=1.1932 val_loss=0.7306 val_accuracy=0.7210
```

Meaning:

```text
epoch=01
```

The model has gone through the training data once.

```text
train_loss=1.1932
```

Training error.

```text
val_loss=0.7306
```

Validation error.

```text
val_accuracy=0.7210
```

Validation accuracy is 72.10%.

Good behavior:

```text
loss decreases
accuracy increases
```

---

# 8. How To Read A Classification Report

Example:

```text
              precision    recall  f1-score   support

T-shirt/top       0.82      0.81      0.82       107
Trouser           0.94      0.96      0.95       105
```

## Support

Support is the number of real examples of that class.

Example:

```text
107 T-shirt/top images
105 Trouser images
```

## Precision

Precision means:

```text
when the model predicted this class, how often was it right?
```

## Recall

Recall means:

```text
of all real examples of this class, how many did the model find?
```

## F1-score

F1-score balances precision and recall.

It is often the easiest metric to compare models.

---

# 9. How To Read The Results CSV Files

Part I results:

```text
outputs/tables/part1_mlp_results.csv
```

Part II results:

```text
outputs/tables/part2_cnn_results.csv
```

You can open them in Excel.

Important columns:

```text
model
test_accuracy
precision
recall
f1_score
macro_f1
model_path
```

For Part I, look mainly at:

```text
test_accuracy
precision
recall
f1_score
```

For Part II, look mainly at:

```text
test_accuracy
macro_f1
```

---

# 10. What To Say In The Final Report

## Part I Short Explanation

The MLP is suitable for the Breast Cancer Wisconsin dataset because each example is represented by a fixed-size vector of numerical features. After normalization, the model can learn nonlinear relationships between the input variables and the target class. The comparison between initialization strategies shows that the training process depends on the initial values of the weights. Gaussian and Xavier initializations performed better than constant initialization.

## Part II Short Explanation

The CNN is more appropriate than an MLP for images because it preserves spatial structure. Instead of flattening the image immediately, it applies local filters that detect visual patterns. Padding, stride, pooling, number of filters, and 1x1 convolution all affect how information is extracted and compressed. In our experiment, the strided CNN gave the best performance, showing that convolutional models can exploit image geometry better than a simple flattened MLP.

---

# 11. Simple Final Comparison Between Part I And Part II

Part I:

```text
Data type: tabular
Input shape: 30 features
Model: MLP
Classes: 2
Samples used: 569
Best accuracy: about 94.19%
```

Part II:

```text
Data type: image
Input shape: 1 x 28 x 28
Model: CNN
Classes: 10
Images used in our run: 6,000
Best accuracy: about 81.9%
```

Main idea:

```text
The architecture must match the structure of the data.
```

For tabular data:

```text
MLP is natural because the input is a vector.
```

For image data:

```text
CNN is natural because the input has spatial structure.
```

---

# 12. Commands To Run The Work

Open VS Code terminal and run:

```powershell
cd "C:\Users\halas\OneDrive\Bureau\DL"
```

Run Part I:

```powershell
python .\src\part1_mlp_tabular.py
```

Run Part II:

```powershell
python .\src\part2_cnn_images.py
```

Part II takes longer than Part I because CNNs process images and have more computation.

---

# 13. What You Should Remember

If you remember only the most important points, remember these:

1. Part I used 569 tabular samples.
2. Part II used 6,000 images from Fashion-MNIST for faster training.
3. MLP is good for vector/tabular data.
4. CNN is good for image data because it keeps spatial structure.
5. Training means adjusting weights to reduce loss.
6. Validation is used to compare models during experiments.
7. Test data is used at the end to estimate final performance.
8. Accuracy tells how many predictions are correct.
9. F1-score balances precision and recall.
10. The final report must connect theory, implementation, results, and interpretation.

