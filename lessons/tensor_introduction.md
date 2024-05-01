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

So, now that we know what tensors are, let's create one. The simplest method for tensor creation is `torch.tensor`. It takes a nested list as an argument and creates a tensor of the same shape. Most of the time, you will use this syntax to create a vector or a matrix. This, for example, is a 4-element vector: `torch.tensor([2.0, 1.45,-1.0, 0.0])`. This would be a 3x2 matrix: `torch.tensor([[1,2],[1,2],[3,4]])`. You can see that the outer pair of brackets has three elements, while the inner ones have two each. This is why the tensor has shape (3, 2), and not (2, 3). You can use this syntax to create any shape of tensor. This for example, would be a 4x1 tensor: `torch.tensor([[2.0],[1.45],[-1.0],[0.0]])`. This concept of one-dimensions is really important. Even though this tensor has the same values as the one-dimensional vector we have seen earlier, it has significantly different properties when used in tensor operations. 
There are some other important tensor creation routines, that we'll get to know better in the tutorials. `torch.linspace` for example is a method that is used very often. It creates evenly spaced values between a specified start and end. `torch.linspace(0, 2, steps=5)` for example would create the tensor `[0, 0.5, 1, 1.5, 2]`. Another important method is `torch.ones`, which takes a desired shape, and creates a tensor of this shape where every value is 1.0. You can scale this tensor to any other value. This line for example, creates a 4x3x4 tensor of 5s: `torch.ones((4,3,4)) * 5`

Tensors support most basic operations, like + or minus, but also logical operations like >, < or ==. For example, the comparison of these two vectors - `torch.tensor([3,5,1]) < torch.tensor([4, 5, 8])` - returns the boolean tensor [True, False, True]. As we can see here, tensors can have different datatypes, like `long` for whole numbers, `float` for fractional numbers and `bool` for booleans. You can cast a tensor to a different type. We can cast our boolean vector back to floats for example, which would return [1.0, 0.0, 1.0].

~ 2 Minutes

## Tensor Manipulation

One of the most important concepts for tensor manipulation is understanding the shapes of tensors, and how we change them. To properly understand reshaping, we need to understand how a tensor is stored in computer memory. Although we are defining tensors with a specific, n-dimensional shape, in reality, all tensors are one-dimensional: A sequence of all the values in the tensor. If we have a 3x4 matrix for example, this is a 12-element one dimensional tensor. And understanding how reshaping works is understanding how this flattening process works.

By default, PyTorch uses a row-major order. This means that elements that are next to each other in a row are also next to each other in the flattened, 1D version of the tensor. The 1D version is constructed by taking all the rows, and concatenating them. This also means that the element one row further down has an offset of W to our element, where number of columns of the matrix. 

This generalizes to more than two dimensions: Let's say we have a batch of matrices, with shape (N, H, W). Flattening this tensor means flattening all individual matrices, which will have shape (H*W) when flattened, and then concatenating them. Flattening happens from right to left: We first go through all the different columns before jumping to the next row. For that we go through all the columns again, go to the next row, and so on. After we are done flattening all the rows, we go to the next matrix in the batch.

A proper understanding of this flattening process makes it easy to understand the concept of reshaping: Reshaping means going to the flattened version of a tensor, then changing the subdivision.
Let's look at an example: 
`torch.tensor([[0, 1, 2], [3, 4, 5]])`
This is a 2x3 tensor. It's flattened version looks like this:
`[0, 1, 2, 3, 4, 5]`
If we are reshaping it shape 3x2 now, this means that we are only taking two elements for each row, so the rows are
`[0, 1], [2, 3,], [4, 5]`
The full tensor now looks like this:
`[[0, 1], [2, 3], [4, 5]]`
This use of "Rearrangement Reshaping" is not the most common. More often, we will use reshape to flatten some dimensions, like reshaping (N, H, W) to (N, H*W), or for unflattening, which is just the reverse. Another common use-case for reshaping is adding 1-dimensions.

~ 2:30 Minutes

Another fundamental task when working with tensors is *indexing* and *slicing*. Indexing is about accessing an element at a specific position in the tensor. If we have this matrix for example
`A = torch.tensor([[0,1,2],[3,4,5]])`
and we wanted to access the element 3, we would index to it using `A[1, 0]`, as it is in the second row (index 1) and first column (index 0). One thing to note is that when indexing elements in pytorch, you don't directly get the elements as numbers, but as one-element tensors. This is because PyTorch wants to keep track of values in it's specific format, for example to allow for automatic differentiation. If you want to access the actual value as a number, you will need to write this as `A[2, 0].item()`.

Slicing lets you extract specific portions of a tensor. Let's look at an example we'll meet again in AlphaFold:
In the Structure Module, we will come across the need to represent transforms, 3D motions, which consist of a Rotation and a Translation. They are in the format of 4x4 matrices of the following form:

$$T = \left(\begin{array}{c|c} R & t \\ \hline 0\;0\;0 & 1\end{array}\right), \; \tilde{x} = \begin{pmatrix}x\\ \hline 1 \end{pmatrix}$$

Here, $R$ is a 3x3 matrix, and $t$ is a 3-element vector. If we want to crop the 3x3 matrix from this transform, we would index it as 
`R = T[0:3, 0:3]`, to specify that for the rows and columns, we want to go from index 0 (inclusive) to index 3 (exclusive). Starting at 0 is the default, so we could rewrite this as `R = T[:3, :3]`.
For the translation, we want the first 3 rows, but only the fourth column. We can write this as `t = T[:3, 3]`. As the fourth row is the last row, we could also write this as `t = T[:3, -1]`, using negative indexing. With this syntax, t would be a one-dimensional tensor. But we might want to concatenate $R$ and $t$ again later into a 3x4 tensor. For this, R and t would need the same number of dimensions, R would be a 3x3 tensor and t a 3x1 tensor. For this, we could also use slicing in the following way: `t = T[:3, 3:4]`, or `t=T[:3, 3:]`, as going to the end is the default. This pattern of using one-element slices, like 3:4, is common to conserve one-dimensions.

Now, we might not work with a single transform `T` of shape (4, 4), but with a batch of trasforms of shape (N, 4, 4). Or we might even have one transform for each backbone in the batch, so we would have shape (N, N_res, 4, 4). To account for this, PyTorch provides the ellipsis operator `...` which is really useful. It expands to as many `:` as are necessary to match the dimensions. The syntax `R = T[..., :3, :3]` correctly extracts the rotation matrices. For the batched case, `...` would expand to `:`. For the double-batched case, it would be `:, :` and for non-batch use, it isn't considered at all.

As a last note on indexing, we will talk on left-hand side and right-hand side indexing. The terms left-hand side and right-hand side refer to the left and right side of an equal sign in programming. So far, we have only used right-hand side indexing, where we index to access a slice of a tensor, and assign it to a variable or process it further.

Left-hand side indexing works just as well: Let's say we have a 3x3 tensor `A`, and we want to replace it's middle column with the tensor [0, 1, 2]. We  can do that like this:

`A[:, 1] = torch.tensor([0, 1, 2])`

Here, we used left-hand side indexing to select the slice of `A` and assign new values to it. This also works with all other indexing techniques.

~ 3:45 Minutes


----------- Part One -----------

Topics for Part 2: Computations and reductions along axes, broadcasting, torch.einsum

Welcome to part 2 of our Introduction to Tensors! In the first part, we covered many of the basics regarding tensors, like the creation of tensors, indexing and reshaping. In this part, we want to highlight some more advanced concepts, in particular computations along axes, broadcasting and the einsum method.

## Computations Along Axes
Let's start with computations along axes. Say you have got a 4x3 matrix and you want to sum up the elements. You have three  different options: Summing up all elements (resulting in a 1-element tensor), summing up all the rows (resulting in a 3-element tensor) or summing up all the colunmns (resulting in a 4-element tensor). You can specify the behaviour by setting the `dim` argument in the method `torch.sum`. If set to `dim=0`, summation will happen along the row dimension. If set to `dim=1` summation will happen along the column dimension. If set to `None`, all elements will be summed up. You can also set it to a tuple: Let's say you had a batch of matrices, of shape (N, 4, 3). You can calculate the sum of all elements for the matrices individually as `torch.sum(A, dim=(1, 2))` or `torch.sum(A, dim=(-1,-2))` using negative indexing.

There are many such operations: With `torch.argmax` you can compute the index of the largest element (total or per row/column), with `torch.mean` and `torch.std` you can compute the mean and standard deviation, and with `torch.linalg.vector_norm` you can compute the standard L2-norm along a dimension. All of these examples are reducing, that means that the dimension you specified as `dim` will be missing in the output shape. You can set the parameter `keepdim=True` to keep it as a one-dimension. This can be really useful, for example to allow for concatenation or broadcasting. 

There are also computations along axes that aren't reducing. One that we will use in the tutorials is the softmax function. The sofmax function takes a vector $\bold{x}$ and creates a new vector according to
$$\operatorname{softmax}(x)_i = \frac{\exp{x_i}}{\sum_j\exp{x_j}}$$
It has three important properties: 
- The entries sum up to 1, so the results can be interpreted as a probability distribution
- If one value in the input is significantly larger than the others (say larger by ~6), the result will be close a one-hot vector where only this entry is 1 and the others are 0, since the exponential function boosts large values overproportionally strong
- In comparison to a 'hard' max function that sets the largest value to 1 and all others to 0, it distributes the mass more smoothly if the large values are close to each other. 

Even though the softmax function is no reduction, it needs a specified dimension, as it is a vector-to-vector calculation. Along the specified dimension, it will create values that sum up to one. Along the other dimensions, it doesn't have this property.

Another similar case where you need to think about axes is when concatenating or stacking tensors. Let's say you have three 4-element tensors `u`, `v`, and `w`. You can use `torch.stack` to stack them to a matrix. If you stack them as `torch.stack((u, v, w), dim=0)`, you will get a 3x4 matrix where the vectors are the individual rows. If you use `torch.stack((u, v, w), dim=1)` or `torch.stack((u, v, w), dim=-1)`, you will get a 4x3 matrix with the vectors as the individual column. The position of the newly introduced dimension 3 is given by `dim`. For `torch.stack`, all tensors to be stacked need to have the same shape, so that the resulting tensor has block form. The similar `torch.cat` doesn't introduce a new dimension but glues the tensors together along an existing dimension. We've mentioned before, when we extracted the 3x3 matrix `R` and the translation vector `t` from the 4x4 tensor `T`, that it is beneficial to extract `t` with slicing so that it has shape (3, 1) instead of (3,). This is practical here: If we wanted to concatenate `R` and `t` again, we can do so using `torch.cat((R, t), dim=-1)` since they have the same number of dimensions.

## Broadcasting
We have hinted on broadcasting quite a bit when talking about the importance of one-dimensions. Here, we will finally see what all the fuzz is about. 

In it's simplest form, broadcasting follows this idea: Let's say we have a 4x3 matrix `A` and a vector `B = torch.tensor([1, 2, 3])`. `B` has the same number of elements as the rows of `A`, and we might want to add it to each row of `A`. To do so, we would need to broadcast it from it's current shape `(3,)` to the shape `(4, 3)`, i.e. to `torch.tensor([[1,2,3],[1,2,3],[1,2,3],[1,2,3]])`. Broadcasting in PyTorch allows us to do just that. First, we would introduce a new one-dimension to `B` and get it to shape (1, 3). This means interpreting the vector `B` as a matrix with one row. We have several options to do so: `B=B[None, :]`, `B = B.unsqueeze(0)`, or `B = B.reshape(1, 3)` are all equally effective in creating the new shape for `B`. Now, if we try to add `A` and `B`, PyTorch automatically broadcasts `B` to shape (4, 3) by duplicating the row. 

If we had `B = torch.tensor([1,2,3,4])` instead, we could do the same trick to add `B` to all columns of `A`. First, we'd compute `B = B.reshape(4, 1)` so that it is a matrix with one column, and when computing `A + B` now, it is implicitly broadcasted to shape (4, 3) to match the shape of `A`. 

In general, the rules for broadcasting are the following: 
* the dimensions of A and B are aligned with each other
* to be broadcastable, the individual dimensions must be equal, or set to one for one of the tensors
* along these one-dimensions, the tensor is expanded by duplicating it, until it matches the shape of the other

There actually is one more rule:
* dimensions are aligned from right to left. If one tensor has fewer dimensions than the other, ones are prepended to match the number of dimensions

With this last rule, we could have left out the reshaping when adding `B` as a row vector to `A`. The shapes (4, 3) and (3,) are broadcastable, as they are aligned from left to right, and ones are prepended to match the number of dimensions.

Let's look at another example: Say we have a batch V of vectors, a tensor of shape (N, 3), and we wanted to normalize them. We can compute the norms for each vector by using `norms = torch.linalg.vector_norm(V, dim=-1)`. The result is of shape (N,). We can unsqueeze it with `norms=norms.reshape(N, 1)`. Now, it is broadcastable and we can compute 
`V_normalized = V / norms`
However, we could have saved the intermediate step by using `keepdim`:
`V_normalized = V / torch.linalg.vector_norm(V, dim=-1, keepdim=True)`
This way, the calculated norms directly have shape (N, 1) and are broadcastable against `V`. 

Another usecase for broadcasting are 'each-with-each' operations. Let's say we have a 3-element vector `v` and a 2 element vector `w`. If we want to compute the product of each element of `v` with each element of `w`, we would get six elements in total. We can organize them in a 3x2 matrix, where the entry i,j is made up of `v[i] * w[j]`. This is called the outer product of `v` and `w`. We can calculate this using broadcasting, by reshaping `v` to a 3x1 column vector and `w` to a 1x2 row vector, followed by a broadcast multiplication of the two. 

## torch.einsum
Broadcasting is an incredibly powerful tool. Together with the indexing, reshaping and reduction techniques we have seen so far, we already have sufficient tools to solve almost any problem in the whole series. 

If you try to do so however, you will see that these methods can quickly get a little cumbersome. Let's look at the most classic tensor-tensor operation: Matrix multiplication. Given two matrices `A` of shape (i,k) and `B` of shape (k,j), the matrix product can formulated like this:

For each row of `A` and for each column of `B`, calculate the pointwise product of the row and the column. After that, sum up along the row-dimension of `A` (which is the column dimension of `B`)

We have seen before that 'each-with-each' calculations can be computed by expanding with one-dimensions and doing a broadcast multiplication. For the first step of this problem, we would calculate `C = A.reshape(i, k, 1) * B.reshape(1, k, j)`. The result would be a tensor of shape (i,k,j). The slices `C[i, :, j]` consist of the pointwise multiplication of the i-th row of `A` with the j-th column of `B`. For matrix multiplication, we would calculate `C = torch.sum(C, dim=1)` to sum these elements up.

We might also want to compute batched matrix multiplication, where `A` has shape (N, i, k) and `B` has shape (N, k, j). We could compute this as 
`C = torch.sum(A.reshape(N, i, k, 1) * B.reshape(N, 1, k, j), dim=-2)`

In this example, we see all typical elements of a tensor-tensor operation: Some of the dimensions (i and j) are present in one of the educts but not the other, and are worked on in an 'each-with-each' fashion. One dimension (k) is present in both and aligned. After calculating the product of the aligned vectors, the dimension is contracted by summation. One dimension (N) is present in both and is just for a batched, 'for-each' calculation. 

PyTorch has a method to cover all these cases in a really concise manner, by using the Einstein notation with `torch.einsum`. The method takes an equation string and the tensors that are used in the operation. For our case of matrix multiplication, the operation would be `torch.einsum('ik,kj->ij', A, B)`. The batched matrix multiplication would be computed as `torch.einsum('Nik,Nkj->Nij', A, B)`. The equation string directly follows the rules we have seen above:
* if a letter is present in one of the educts, but not in the other and kept in the product (like i and j), the other tensor is unsqueezed and broadcasted to include the dimension
* if a letter is present in both educts, but not in the product (like k), it is contracted 
* if a letter is present in both educts and the product, it is treated as a batched, 'for-each' dimension

In the same way we used the ellipsis operator before during indexing, we can use it here as well to account for any number (or none at all) of prepended batch dimensions. With `torch.einsum('...ik,...kj->...ij', A, B)` we can cover both the batched and non-batched case simultaneously.

Another example is the outer product we computed earlier. Without explicitly reshaping the tensors ourselves, it can be written as `torch.einsum('i,j->ij', v, w)`. 

`torch.einsum` is incredibly flexible. You will use it all the time in the tutorials, and you will have quite a bit of examples in the tutorial for this tensor introduction as well. The construction of the equation string can be a bit tricky at the start, but you will quickly get used to it. 

## Conclusion
With this, we are done with the introduction to tensors. In total, this was a quite extensive introduction, but it has the benefit that you've already seen almost all of the operations you will need during the series. There are of course some more miscallaneous operations we will need during the way, but with all the tools discussed in here, you have enough foundation to look up new methods online or with ChatGPT. We have prepared a tutorial Jupyter Notebook where you are guided through using all the methods we have shown you. After that, you will be well prepared to start with the next part of the series: The Introduction to Machine Learning.