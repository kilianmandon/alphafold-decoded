# Machine Learning

In this lesson, we’ll give an introduction to machine learning. Specifically, we’ll implement a two-layer feed-forward neural network, to predict handwritten digits. We'll do so from scratch, using nothing but the basic tensor operations we've learned in the last section.

To start with that, let’s recall how we did the handwritten digit recognition before, in our tensor introduction: In it, we first calculated mean images for each number, ending up with a tensor of shape (10, 28, 28). This tensor held values between 0 for black pixels and 1 for white pixels. For each of the 10 digits, we normalized the images by subtracting their mean (of shape (10,)) and their standard deviation (also of shape (10,)). After normalization, values in the tensor were distributed around 0 where negative values represented black pixels, and positive values represented white pixels.

To do inference, we multiplied the image we wanted to test against each of these templates and summed the values up. If the image was similar to the template, it held values close to 1 where the template had positive values, and values close to 0 where the template had negative values. After multiplication and addition, the image would get a high score for this template, as only the positive values in the template were multiplied against high numbers from the test image, while negative values were multiplied against low values.

We can flatten both the templates and the test images without affecting the calculation. This way, the templates are a tensor of shape (10, 784) and the test images are vectors of shape (784,). The multiplication-and-summing-up operation is just matrix-vector multiplication between the template matrix and the test vectors, and we end up with vectors of shape (10,), containing high scores were the test image highly agreed with the templates, and low values when the test image was different from the templates.

This is exactly what Machine Learning does, with two differences: 

1. Features (like the agreement with the templates) are calculated hierarchically, not in a single step.
2. The values for the weighting aren’t calculated by some classic algorithm (like we did here with using mean images as weights for the pixels) but are learned to fit the data.

## Hierachical Features
Let’s start with the first point: Hierarchical features. The idea is simple. Looking at the digits, many of them share characteristic features, like a circle in the top-half of the image, or a diagonal line in the center. We could come up with any number H of these features, with a template for each of them. The templates would be flattened images, where pixels that should be white in the matched features have positive values, while pixels that should be black in the matched features have negative values. These templates would be gathered in a matrix $W_1$ of shape (784, H). Multiplication of the test images with the matrix would result in a vector of shape (H,) with high values for features that are present in the image, and low values for features that are not present. 

These can then be combined in a second stage to get scores for actual numbers: An eight corresponds to a circle in the upper-half and one in the lower-half, all other features shouldn’t be contained. If the hidden dimension had features like (‘upper-half-circle’, ‘diagonal-line’, ‘straight-horizontal-line’, ’lower-half-circle’,…), then we could test if the feature vector represents an eight, by multiplying it against a vector like (1, -1, -1, 1, …). If the feature vector was calculated for an image of an eight, this would lead to a high score. We can come up with these weightings for the other digits and combine them into a matrix $W_2$ of shape (10, H). Now, we can multiply the feature vector against it to get scores for the digits. 

This hierarchical approach to the classification isn’t only helpful for reusing shared features, it can also be used to cover alternative ways of drawing digits: A four for example, could be drawn in a closed or open form. The hidden dimension could include both as templates, resulting in two different four-features. The matrix for calculating the end-scores could give both of them a high weighting, so that both features result in a strong vote for 4. 

## Non-Linearities
There is a little catch to it: What we’ve done so far is calculating $h=W_1\cdot x$ to calculate the feature vector, then $z=W_2\cdot h$ to calculate the final scores. In total, we have $z = W_1\cdot (W_2\cdot x) = (W_1\cdot W_2)\cdot x$. But the matrix product $W_1\cdot W_2$ is just a (10, 784) matrix. The two (10,H) and (H, 784) matrices collapsed together into a single matrix, and we lost all our imagined possibilities of hierarchical features. 

But our ideas weren’t stupid. All the benefits of hierarchical features we came up with are real, and the collapse of the two layers is just a technical issue. A simple linear weighing of the features simply can’t keep the information we gained (which we see in the way the two matrices collapsed together). But there is an easy fix. It is to introduce any reasonable, non-linear function in between. There are many good options (Sigmoid, Leaky ReLU, tanh), and we’ll use the ReLU function, which is the most abundantly used one. It simply clamps all negative values to zero. Our new formulation looks like $W_2 \cdot  relu(W_1\cdot x)$, and because of the relu function, the matrices don’t fold together anymore. 

This formulation of a layer (matrix multiplication) is called a linear layer, or a linear layer without bias. A bias is a vector that we add to the result of the matrix multiplications. Our final formulation is $W_2 \cdot  relu(W_1\cdot x+b_1) + b_2$. Including the option of a bias can be helpful sometimes: If, for example, the dataset contained way more images of 7s than of 1s, it might be helpful to learn a positive bias for the score of 7s and a negative bias for the score of 1s to increase the number of correct guesses. This type of layer (matrix multiplication + bias) is called an affine linear layer, or a linear layer with bias. Schematically, our model looks like this: affine -> relu -> affine

## The Loss Function
We are done with the formulation of our network:
$$z = W_2 \cdot  relu(W_1\cdot x+b_1) + b_2$$
All that’s left to do now is to find good values for the parameters $W_2, W_1, b_2$ and $b_1$. To do that, we first need a way to determine if a set of parameters is good, a measure for the quality of our model. This is done with a loss function. A loss function gives a high value for bad predictions, and a low value for high predictions. We’ll use a simple loss function: We first map our scores z (which are any real numbers) to the range [0, 1]. This way, a good prediction would be a one-hot vector, meaning that only the value for the correct index is 1, while all others are 0. A good prediction for a four would look like (0, 0, 0, 0, 1, 0, 0, 0, 0, 0). A suitable function to squish the real numbers (from negative infinity to positive infinity) into the range [0, 1] is the sigmoid function:
$$\sigma(x) = \frac{1}{1+e^{-x}}$$
It's also a working non-linearity that can be used instead of ReLU.

Now, we create these “should-be results” for all images in the training data from the labels of the images. The labels (so far) were just numbers, the numbers shown in the images. The one-hot encoded versions $\hat{y_i}$ (which can be compared directly to the model outputs) are 10-element vectors. They are also called labels. For the comparison, we want a function that gives high values when the guesses and labels are really different, and low values if they are similar. There are many different options here. We’ll choose the so-called L2-loss:
$$\frac{1}{N} \sum_{i=1}^N (y_i – \hat{y_i})^2$$
where $y_i = \sigma(z_i)$ are the [0,1]-mapped model outputs. The L2-loss works great for this simple model, and it’s a good educational loss function because it’s easy to understand and easy to differentiate. It’s not the most common loss function for discrete classification (which is what we’re doing here), but it has some real uses for trying to predict continuous outputs (like the positions of atoms in AlphaFold).

## Gradient Descent
We found a criterium for model optimization: Minimizing the L2-loss of our model on the training set. We could, theoretically, just try out random parameters and select the combination with the lowest loss, but given the number of parameters, it is infeasible to use this method for parameter optimization. The idea that's actually being used is this: For every image, we nudge the parameters by a very small amount into the direction that improves the prediction for this specific image. 

Finding out how to enhance the single prediction might seem difficult at first, but it's actually easy. We can (and will do so in a minute) calculate the derivative of the loss with respect to every single parameter of our model. That means every single number in the tensors $W_1, W_2, b_1$ and $b_2$. If this derivative is positive, we know that stepping to the right (increasing the parameter) increases the loss, while stepping to the left (decreasing the parameter) decreases the loss. If the derivative is negative, it's the other way around. The actual value of the derivative also plays a role: If the absolute value is high this parameter drastically affects the loss. If it's low, it's insignificant. 

The collection of the loss with respect to all parameters is called the gradient, and it is written as $\nabla L$. We've seen that changing every parameter in the opposite direction of the derivatives reduces the loss. Also, changes in parameters where the derivative has a high absolute value affect the loss stronger, so we might want to focus on them. There's an actual mathematical theorem to consolidate this: Given a gradient for a function (the derivatives for all parameters), subtracting the gradient from the parameters is the locally optimal step towards minimizing the function.


Summing everything up:
- We calculate the forward pass for our model as 
    $$z_i = W_2\cdot relu(W_1\cdot x_i+b_1) + b_2$$
    $$y_i = \sigma(z_i)$$
    where $W_j, b_j$ are parameters that are initialized randomly and improved during training.
-  For a batch of $N$ train images $x_i$ with labels $\hat{y}_i$, we calculate the loss of the model as 
    $$L = \frac{1}{N} \sum_i (y_i-\hat{y_i})^2$$
    Our goal is to minimize this loss. 
- For optimization, we calculate the derivatives $\frac{\partial L}{\partial W_j}$ abnd $\frac{\partial L}{\partial b_j}$. If we decrease the parameters by them a tiny bit, like
    $$W_j \leftarrow W_j - \alpha \frac{\partial L}{\partial W_j}$$
    with a small value $\alpha$ (the learning rate), we decrease the loss.

All that's left to do is calculating the derivatives.

## Calculating the Derivatives
We'll go through the calculation of the derivatives rather quickly. If you aren't super involved in math, this might be challenging for you, but it's not too important that you understand the individual steps anyway. Calculating the derivatives is something that Pytorch does for you automatically, and we'll just do it by hand this once in the introduction to Machine Learning. It's cool if you can follow the equations (and you might want to watch a slower-paced video for this), but it's more important to understand the principle of gradient descent itself. 

With that out of the way, let's jump into the math. We can use the chain rule to trace the derivatives through the network layer by layer. For each layer, we'll talk about _upstream gradients_ and _downstream gradients_. For a layer $z = F(x)$ that converts an input $x$ to an output $z$, the _upstream gradient_ $\nabla_z L$ is the derivative of the loss with respect to the layer's output. The _downstream gradient_ $\nabla_x L$ is the derivative of the loss with respect to the layer's input. For each layer in the network, we receive the upstream gradient, and use it to calculate the downstream gradient and the gradient of the layer's parameters, if it has any.

We have four layers that we need to calculate derivatives for:
1) L2-loss
2) Affine linear layer
3) ReLU
4) Sigmoid

### L2-Loss
The derivative for the L2-loss is calculated like this:
$$L = \frac{1}{N}\sum_i (y_i-\hat{y_i})^2, \;\frac{\partial L}{\partial y_i} = \frac{1}{N}\sum_i 2 (y_i-\hat{y_i})$$
We shouldn't forget, that $y_i$ and $\hat{y_i}$ aren't numbers here, but 10-element vectors (the index is not for the individual numbers, but for the individual samples in the batch). So, writing this out as single numbers, it would look like
$$L = \frac{1}{N}\sum_{i=1}^{N} \sum_{j=1}^{10} (y_{ij}-\hat{y_{ij}})^2$$
but that doesn't change the calculation of the derivative.

### Affine Linear Layer
Written out, the formulation of an affine linear layer looks like this:
$$\begin{pmatrix}z_1 \\ \vdots \\ z_n\end{pmatrix} = \begin{pmatrix}w_{11} & w_{12} & \cdots & w_{1m} \\ \vdots & \ddots & \ddots & \vdots \\ w_{n1} & w_{n2} & \cdots & w_{nm} \end{pmatrix} \begin{pmatrix}x_1 \\ \vdots \\ x_m \end{pmatrix}+ \begin{pmatrix}b_1\\\vdots \\ b_n\end{pmatrix}$$
or
$$\begin{pmatrix}z_1 \\ \vdots \\ z_n\end{pmatrix} = \begin{pmatrix}w_{11} x_1 & w_{12} x_2 & \cdots & w_{1m} x_m \\ \vdots & \ddots & \ddots & \vdots \\ w_{n1} x_1 & w_{n2} x_2 & \cdots & w_{nm} x_m \end{pmatrix}+ \begin{pmatrix}b_1\\\vdots \\ b_n\end{pmatrix}$$

From this form, we can read off the derivatives by using the chain rule:
$$\frac{\partial L}{\partial b_j} = \frac{\partial L}{\partial z_j}$$
$$\frac{\partial L}{\partial x_j} = \sum_{i=1}^{n} \frac{\partial L}{\partial z_i}\frac{\partial z_i}{\partial x_j}= \sum_{i=1}^n \frac{\partial L}{\partial z_i} w_{ij} = W^T\cdot \nabla_z L$$
$$\frac{\partial L}{\partial w_{ij}} = \frac{\partial L}{\partial z_i} x_j$$

If the loss was calculated on a batch, $z_i$ and $x_i$ have an additional batch dimension and the calculations $W^T \cdot  \nabla_z L$ and $\frac{\partial L}{\partial z_i}x_j$ will also include the batch dimension. As $W$ and $b$ contribute in all iterations in the batch, we get their gradients from summing up the batch dimension.

Basically, the gradients for the affine linear layer are calculated in the following way:
- The gradient of the bias is just the upstream gradient $\nabla_z L$
- The downstream gradient for the layer input $x$ is the matrix product of $W$ and the upstream gradient (after moving the dimensions of $W$ around so that the shape of the output matches $x$)
- The gradient of the weight matrix is the outer product of the upstream gradient and the layer input $x$.
- If the loss is batched, the weight and bias gradients are summed up along the batch dimension.
  
### ReLU
This is easy again: If the input was negative, it was set to 0 and no small changes to the input can change the output. So, for items where the input was negative, the gradient of the input is 0. Where the input was positive, ReLU didn't change it, so for positive inputs, the downstream gradient is just the upstream gradient. In theory, ReLU isn't differentiable in 0, but this isn't a problem in practice. We'll treat the derivative in 0 as 0.

### Sigmoid
The Sigmoid function $z = \sigma(x) = \frac{1}{1+e^{-x}}$ follows the differential equation 
$$\sigma'(x) = z(1-z)$$
Note that $z$ isn't the layer input, but the layer output. By the chain rule, the downstream gradient is 
$$\nabla_x L = \nabla_z L \cdot z (1-z)$$

With these calculations for the propagation of the gradients through the individual layers, we can calculate the derivatives for every parameter in the model, by chaining them together.

## Conclusion
The actual formulation of Machine Learning is really similar to our naive, filter-based approach we used in the tensor introduction for image classification, with two differences: The introduction of hierarchical features by introducing more layers to the network, and the idea of automatically adjusting the parameters by gradient descent.

We've seen three types of layers:
- Affine linear layers: Matrix multiplication with a weight matrix plus bias addition
- ReLU: max(0, x), setting all negative values to 0.
- Sigmoid: $\sigma(x) = 1/(1+e^{-x})$

We constructed our two-layer classifier as 
affine -> relu -> affine -> sigmoid

This gives rise to outputs in the range [0, 1]. We use one-hot encoding to encode our labels as 10-element tensors, where only the index of the number that is shown is set to 1 and all others are set to 0. We use these as targets for our training.

We introduced the L2-loss $L = \frac{1}{N} \sum_i (y_i-\hat{y_i})^2$, where $N$ is the batch size. For training, we sample a small batch of $N$ train images, calculate the prediction with the model's forward pass, then optimize the parameters by gradient descent.

For gradient descent, we need to calculate the derivatives for all of the layers. The computations of these is a little messy, but we derived terms that we can translate into python code.

In the tutorial, we'll put all of these steps together and implement a ready-to-use machine learning model for handwritten digit recognition. When you're ready, head over to the Notebook and start the actual implementation.

This was a really quick dive into Machine Learning. If you found the material to fast or think that you didn't understand important details, there are great alternative tutorials that will guide you through the topic in a more slow-paced, visually rich fashion. I really enjoyed watching [this](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&si=K9H1t44OgDMBidQG) series by 3Blue1Brown and [this](https://youtu.be/hfMk-kjRv4c?si=WbyaHb36XXPWXUaO) video by Sebastian Lague. Both are comprehensive, but the first one goes a little deeper into the mathematical details. 

All in all, remember that modern machine learning frameworks protect you from thinking about most of these details, and many guides to machine learning just don't walk you through this part (doing ML by hand) at all. Personally, I think it's really educative, but you don't have to worry about the rest of this series on Alphafold just because you didn't like the vector calculus for the derivatives.