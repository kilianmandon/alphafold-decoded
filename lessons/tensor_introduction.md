# Tensor Introduction

## Intro

In this lesson, we will give an Introduction to Tensors in PyTorch.
Tensors are the fundamental building block that drives Machine Learning. 
Now, you have probably worked with tensors before. Vectors for example are one dimensional tensors. Matrices are two-dimensional tensors. Operations like the dot-product between vectors, or matrix vector multiplication, are classic examples of operations on tensors.

Tensors are a generalization of this: A tensor is an n-dimensional array. 
Going beyond two or three dimensions might sound a little odd or technical, but it actually comes up really naturally. 

If you start of working with black-and-white images, like we will do when classifying handwritten digits in the tutorials, you are totally fine with two dimensions. A black and white image is basically a matrix of values between 0 for black and 1 for white. You can specify a row and a column and access the pixel value at this location.

If you think of a coloured image however, at each pixel, there already are three values: The amound of red, green, and blue at the pixel location. At each pixel, we have a feature vector consisting of three values. The whole image is a feature volume, a three-dimensional tensor of shape (H, W, C), where H is the number of rows, or the height of the image, W is the number of columns, or the width of the image, and C is the number of channels, in this case three. To get the redness of the top left pixel, we would index into the image like img[0, 0, 0].

But three dimensions often aren't enough: In Machine Learning, we are usually working with batches of data, for example, we might compute the forward pass through our model for a set of N images of different dogs. Now, we would need a 4-dimensional tensor of shape (N, H, W, C) to process all the images in a joint fashion. This is the typical form of tensors in machine learning: A batch of feature volumes, where the dimensions of the feature volume have actual semantic meaning (like the location in the image and the color channel) and the batch dimension is just being carried along in a 'for-each' manner.

So, this is what tensors are, but what is PyTorch? PyTorch is a Machine Learning framework, and at the core, it has two tasks: 
First, it needs to implement a multitude of operations on tensors, like matrix multiplication, and it needs to do so very efficiently. You may have worked with numpy before, which basically does the same thing. PyTorch however can be orders of magnitudes faster, as it can harness the power of your Graphics Processing Unit.
The second task is a little less obvious, and it's called Automatic Differentiation. We will take a closer look at why this is so important, but basically, training a Machine Learning is about minimizing the error of your model. And the way that this is done is by computing the derivative of the error with respect to every single parameter in your model. Knowing the derivative, we know in which direction we need to change the parameter to decrease the error, and this is the whole magic of Machine Learning. Calculating these derivatives by hand is cumbersome and error-prone, and luckily, PyTorch does it automatically. 
[We won't be training AlphaFold ourselves in this series, so aside of the Introduction to Machine Learning, you will probably only see this feature coming up as a spike in your used memory when you enable the feature by accident, as keeping track of the whole computational graph for the calculation of derivatives requires a lot of compute resources.]

~ 3 Minutes

## Tensor Creation

So, now that we know what tensors are, let's create one. The simplest method for tensor creation is `torch.tensor`. It takes a nested list as an argument and creates a tensor of the same shape. Most of the time, you will use this syntax to create a vector or a matrix. This, for example, is a 4-element vector. This would be a 3x2 matrix. You can see that the outer pair of brackets has three elements, while the inner ones have two each. This is why the tensor has shape (3, 2), and not (2, 3). You can use this syntax to create any shape of tensor. This for example, would be a 4x1 tensor. This concept of one-dimensions is really important. Even though this tensor has the same values as the one-dimensional vector we have seen earlier, it has significantly different properties when used in tensor operations. 
There are some other important tensor creation routines, that we'll get to know better in the tutorials. `torch.linspace` for example is a method that is used very often. It creates evenly spaced values between a specified start and end. `torch.linspace(0, 2, steps=5)` for example would create the tensor [0, 0.5, 1, 1.5, 2]. Another important method is `torch.ones`, which takes a desired shape, and creates a tensor of this shape where every value is 1.0. You can scale this tensor to any other value. This line for example, creates a 4x3x4 tensor of 5s.

Tensors support most basic operations, like + or minus, but also logical operations like >, < or ==. For example, the comparison of these two vectors returns the boolean tensor [True, False, True]. As we can see here, tensors can have different datatypes, like `long` for whole numbers, `float` for fractional numbers and `bool` for booleans. You can cast a tensor to a different type. We can cast our boolean vector back to floats for example, which would return [1.0, 0.0, 1.0].

~ 2 Minutes

## Tensor Manipulation

One of the most important concepts for tensor manipulation is understanding the shapes of tensors, and how we change them. To properly understand reshaping, we need to understand how a tensor is stored in computer memory. Although we are defining tensors with a specific, n-dimensional shape, in reality, all tensors are one-dimensional: A sequence of all the values in the tensor. If we have a 3x4 matrix for example, this is a 12-element one dimensional tensor. And understanding how reshaping works is understanding how this flattening process works.

By default, PyTorch uses a row-major order. This means that elements that are next to each other in a row are also next to each other in the flattened, 1D version of the tensor. The 1D version is constructed by taking all the rows, and concatenating them. This also means that the element one row further down has an offset of W to our element, where number of columns of the matrix. 

This generalizes to more than two dimensions: Let's say we have a batch of matrices, with shape (N, H, W). Flattening this tensor means flattening all individual matrices, which will have shape (H*W) when flattened, and then concatenating them. Flattening happens from right to left: We first go through all the different columns before jumping to the next row. For that we go through all the columns again, go to the next row, and so on. After we are done flattening all the rows, we go to the next matrix in the batch.

A proper understanding of this flattening process makes it easy to understand the concept of reshaping: Reshaping means going to the flattened version of a tensor, then changing the subdivision.
Let's look at an example: 
torch.tensor([[0, 1, 2], [3, 4, 5]])
This is a 2x3 tensor. It's flattened version looks like this:
[0, 1, 2, 3, 4, 5]
If we are reshaping it shape 3x2 now, this means that we are only taking two elements for each row, so the rows are
[0, 1], [2, 3,], [4, 5]
The full tensor now looks like this:
[[0, 1], [2, 3], [4, 5]]
This use of "Rearrangement Reshaping" is not the most common. More often, we will use reshape to flatten some dimensions, like reshaping (N, H, W) to (N, H*W), or for unflattening, which is just the reverse. Another common use-case for reshaping is adding 1-dimensions.

~ 2:30 Min

Another fundamental task when working with tensors is *indexing* and *slicing*. Indexing is about accessing an element at a specific position in the tensor. If we have this matrix for example
A = torch.tensor([[0,1,2],[3,4,5]])
and we wanted to access the elemt 3, we would index to it using A[1, 0]. As it is in the second row (index 1) and first column (index 0). One thing to note is that when indexing elements in pytorch, you don't directly get the elements as numbers, but as one-element tensors. This is because PyTorch wants to keep track of values in it's specific format, for example to allow for automatic differentiation. If you want to access the actual value as a number, you will need to write this as A[1, 0].item()

Slicing lets you extract specific portions of a tensor. Let's look at an example we'll meet again in AlphaFold:
In the Structure Module, we will come across the need to represent transforms, 3D motions, which consist of a Rotation and a Translation. They are in the format of 4x4 matrices of the following form:

$$ T = \left(\begin{array}{c|c} R & t \\\ \hline 0\;0\;0 & 1\end{array}\right), \; \tilde{x} = \begin{pmatrix}x\\\ \hline 1 \end{pmatrix}$$

Here, $R$ is a 3x3 matrix, and $t$ is a 3-element vector. If we want to crop the 3x3 matrix from this transform, we would index it as 
`R = T[0:3, 0:3]`, to specify that for the rows and columns, we want to go from index 0 (inclusive) to index 3 (exclusive). Starting at 0 is the default, so we could rewrite this as `R = T[:3, :3]`.
For the translation, we want the first 3 rows, but only the fourth column. We can write this as `t = T[:3, 3]`. As the fourth row is the last row, we could also write this as `t = T[:3, -1]`, using negative indexing. With this syntax, t would be a one-dimensional tensor. But we might want to concatenate $R$ and $t$ again later into a 3x4 tensor. For this, R and t would need the same number of dimensions, R would be a 3x3 tensor and t a 3x1 tensor. For this, we could also use slicing in the following way: `t = T[:3, 3:4]`, or `t=T[:3, 3:]`, as going to the end is the default. This pattern of using one-element slices, like 3:4, is common to conserve one-dimensions.

~ 2:30 Min