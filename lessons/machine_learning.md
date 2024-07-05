# Machine Learning

Hi everyone, and welcome to our introduction to machine learning. This video is the second in our series where we'll be implementing alphafold from scratch. 
In this video, you'll learn how to build a two-layer feed-forward neural network for the classification of handwritten digits ground-up. We'll use nothing but the basic tensor operations we've learned in the last section.

There are already a bunch of great videos out there that teach you basically the same thing. I particularly enjoyed the videos from 3blue1brown and Sebastian Lague. Like all of them, we'll try to be concise while still giving you the full theoretical background and appreciating the beauty of the algorithm. There are three points that make this video a little different and potentially more suitable for you: First, we'll leverage our knowledge on tensors from the last lesson. This means that we don't have to build the theory up using single numbers and neurons; instead, we can directly use matrices and vectors. This makes the formulation a lot cleaner and closer to how it's done in reality. Second, as with all videos in this series, we'll provide you with a Jupyter Notebook where you can do the whole implementation by yourself, ensuring you truly understand it. And, third, it just fits a little more snugly into the rest of the series, aligning with the methods we use and preparing you for the next steps in completing AlphaFold.

So, with that out of the way, let's dive right in!

Machine Learning is about detecting patterns in data. We've got a simple pattern we want to look for on the left, and you can see that it's pretty similar to the one at the top right and almost complementary to the one at the bottom. But how can we quantify that? As a first step, we need to quantify the patterns themselves, for example with values between 0 and 1 based on their color. Now, a simple metric for estimating the similarity of the patterns is to calculate the dot product of the vector on the left with the ones on the right. We can see that this works pretty well for the first and last test, but it's odd that the full white pattern has the highest scores. It's easy to see what went amiss here. We're only rewarding white pixels where they should be, but we don't penalize them where they should be black. The solution is to normalize either the template, the test images or both, so that they are distributed around 0. This normalized dot works really well as a similarity check, and assigns reasonable scores to all four test images. 

If you already finished the tutorial notebook from the last lesson, this is exactly what we did in the last section of it, where we tried to classify handwritten digits without machine learning. First, we gathered all train images with the same label and calculated their mean, so that we have a template to search for. After that, we took each of these templates and normalized it by subtracting the mean value and dividing by its standard deviation, to make the values evenly distributed around 0. Then, to test the validation images, them against the templates pointwise and summed the values up over the different pixels. This is equivalent to the dot-product for the one-dimensional images we saw before. In fact, we could also flatten the images and templates so that they are one-dimensional themselves, making the operation between them a simple matrix multiplication of the templates against the transposed validation images. And with this simple method, we already reached an accuracy of about 80%! 

Going to Machine Learning, we do exactly the same, with two small differences:

1. Features (like the agreement with the templates) are calculated hierarchically, not in a single step.
2. The values for the weighting aren’t calculated by some classic algorithm (like we did here with using mean images as weights for the pixels) but are learned to fit the data.

## Hierachical Features
Let’s start with the first point: Hierarchical features. The idea is simple. Looking at the digits, many of them share characteristic features, like a circle in the top-half of the image, or a diagonal line in the center. We could come up with any number H of these features, with a template for each of them. The templates would be flattened images, where pixels that should be white in the matched features have positive values, while pixels that should be black in the matched features have negative values. These templates would be gathered in a matrix $W_1$ of shape (784, H). Multiplication of the test images with the matrix would result in a vector of shape (H,) with high values for features that are present in the image, and low values for features that are not present. 

These can then be combined in a second stage to get scores for actual numbers: An eight corresponds to a circle in the upper-half and one in the lower-half, all other features shouldn’t be contained. If the hidden dimension had features like (‘upper-half-circle’, ‘diagonal-line’, ‘straight-horizontal-line’, ’lower-half-circle’,…), then we could test if the feature vector represents an eight, by multiplying it against a vector like (1, -1, -1, 1, …). If the feature vector was calculated for an image of an eight, this would lead to a high score. We can come up with these weightings for the other digits and combine them into a matrix $W_2$ of shape (10, H). Now, we can multiply the feature vector against it to get scores for the digits. 

This operation is often visualized using a graph like shown here. The circles represent individual neurons, which are just numbers, and the layers of neurons are, from left to right, the 784 element input image, the 16 element feature vector, and the 10 element output vector. The next layer is computed from the previous by multiplying it against the associated weight matrix. Equivalently, we could say that each neuron in a layer is computed by multiplying each neuron value from the previous layer with the weight that's connecting it, then summing these values up.

This two-layered, hierarchical approach to the classification isn’t only helpful for reusing shared features, it can also be used to cover alternative ways of drawing digits: A four for example, could be drawn in a closed or open form. The hidden dimension could include both versions as templates, resulting in two different four-features. The matrix for calculating the final scores from the features could give both of them a high weighting, so that both features result in a strong vote for 4 if present. 

## Non-Linearities
So summing up, here is what we did so far: We started out with our 784 element input vector. We multiplied it against a weight matrix W1 to compute our hidden feature vector h. Here, the rows of W1 represent flattened images of patterns we might be interested in, with negative values at pixels that should better be black to match the template and positive values at pixels that should better be white to match the template. Next, we multiplied the feature vector against a second matrix W2 to get our ten-element output. But there's a catch: this chain of matrix multiplication is, by the rules of math, equivalent to multiplying with a single matrix, which can be computed as W2@W1. The two matrices collapse together, and we loose all our imagined improvements from using hierarchical features.

But our ideas weren’t stupid. All the benefits of hierarchical features we came up with are real, and the collapse of the two layers is just a technical issue. A simple linear weighing of the features simply can’t keep the information we gained (which we see in the way the two matrices collapsed together). Luckily, there is an easy fix. It is to introduce any reasonable, non-linear function in between. 

There are many good optioners here. The one we'll use for the hidden layer is the ReLU function, which is the most abundantly used activation function. It simply clamps all negative values to zero. 

Another common non-linearity that we'll use later in the model is the sigmoid function, which squishes all inputs between zero and one. A downside of the sigmoid function is that when saturated, meaning for large positive or negative inputs, changes to the inputs don't really affect the output, which makes it difficult to adjust the weights properly.

No matter which one we choose, it prevents the matrices from folding together. Including the non-linearity, our architecture now looks like this.

The layers we are using here, which are nothing but matrix multiplication, are called linear layers, or linear layers without bias. A bias is a vector that we add to the result of the matrix multiplications. With a bias in each layer, our final formulation now looks like this. Including the option of a bias can be helpful sometimes: If, for example, the dataset contained way more images of 7s than of 1s, it might be helpful to learn a positive bias for the score of 7s and a negative bias for the score of 1s to increase the number of correct guesses. This type of layer (matrix multiplication + bias) is called an affine linear layer, or a linear layer with bias, or often just a linear layer as well. Schematically, our model looks like this: An affine layer, going from 784 dimensions to 16, followed by a ReLU activation and another affine layer, going a dimension of 16 to 784.

What comes next is the magic of machine learning. We've talked a lot about what good values for the weight matrices might be. Truth is, we don't need to decide on them. We simply choose them as the values that lead to the best result, using an algorithm that's astonishingly simple.

To get to that point, we first need a way to tell our computer what the best result is, or at least, what makes a result good. That's not to difficult; for a specific input image, we know what we would want the output to be like: a vector where only the output at the correct index is 1, while all other outputs are zero. This is called a one-hot encoding of the label. With this perfect output in mind, we need a measure to calculate how much it differs from the actual output, and this metric is called a loss function. 

For our implementation here, we’ll choose the so-called L2-loss:
$$\frac{1}{N} \sum_{i=1}^N (y_i – \hat{y_i})^2$$
Here, N is the number of images that we passed through the model, and the different y_k  are the model outputs for the images, squished to the range [0, 1] by the sigmoid function. The L2-loss works great for this simple model, and it’s a good educational loss function because it’s easy to understand and easy to differentiate. It’s not the most common loss function for discrete classification (which is what we’re doing here), but it has some real uses for trying to predict continuous outputs (like the positions of atoms in AlphaFold).

Looking at some code, this is what we came up so far: First, we compute the desired output as a one-hot encoding of the labels. Then, we calculate the prediction of our model for a batch of images. Last, we compute the L2-loss between the predictions and the labels. At the start of training, we initialize our weights randomly, leading to a bad prediction and a high loss.

## Gradient Descent
So, we found a criterion for model optimization: Minimizing the L2-loss of our model on the training set. We could, theoretically, just try out random parameters and select the combination with the lowest loss, but given the number of parameters, that's infeasible. The idea that's actually being used is this: For every image, we nudge the parameters by a very small amount into the direction that improves the prediction for this specific image. 

Finding out how to enhance the single prediction might seem difficult at first, but it's actually easy. While holding all other parameters constant, we can take a look at the loss as a function in one individual parameter, and do that for every single parameter in the model. That means for every single number in the tensors $W_1, W_2, b_1$ and $b_2$. For these functions, we can explicitly calculate the derivative of the loss with respect to the parameter - we'll see how in a minute. If this derivative is positive, we know that stepping to the right (increasing the parameter) increases the loss, while stepping to the left (decreasing the parameter) decreases the loss. If the derivative is negative, it's the other way around. The actual value of the derivative also plays a role: If the absolute value is high this parameter drastically affects the loss. If it's low, it's insignificant. 

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
$$\begin{align*}L &= \frac{1}{N}\sum_i (y_i-\hat{y_i})^2 \\ \frac{\partial L}{\partial y_i} &= \frac{1}{N}\cdot 2 \cdot (y_i-\hat{y_i})\end{align*}$$
We shouldn't forget, that $y_i$ and $\hat{y_i}$ aren't numbers here, but 10-element vectors (the index is not for the individual numbers, but for the individual samples in the batch). So, writing this out as single numbers, it would look like this:
$$\begin{align*}
L &= \frac{1}{N}\sum_{i=1}^{N} \sum_{j=1}^{10} (y_{ij}-\hat{y}_{ij})^2 \\
\frac{\partial L}{\partial y_{ij}} &= \frac{1}{N}\cdot 2 \cdot (y_{ij}-\hat{y}_{ij})
\end{align*}
$$
As we can see, it doesn't change the form of the derivative.

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