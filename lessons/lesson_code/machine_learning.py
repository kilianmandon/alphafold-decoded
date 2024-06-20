import torch
from torch import nn

train_loader, validation_loader = None, None
N_val = None
split_train_images = None
split_by_label = None

# train_images has shape (N_train, 28, 28)
train_images, train_labels = next(iter(train_loader))
# validation_images has shape (N_val, 28, 28)
validation_images, validation_labels = next(iter(validation_loader))

# 10 items of shape (N_digit, 28, 28) each
split_train_images = split_by_label(train_images, train_labels)

# mean_images has shape (10, 28, 28)
# mean_images has values in the range [0, 1]
mean_images = [torch.mean(images, dim=0) for images in split_train_images]

# Normalization:
means = torch.mean(mean_images, dim=(-1, -2), keepdim=True)
stds = torch.std(mean_images, dim=(-1, -2), keepdim=True)

# templates has standard normally distributed values
# i.e. ~ range [-1, 1]
templates = (mean_images - means) / stds

# agreement has shape (N_val, 10)
agreement = torch.einsum('Nhw,Chw->NC', validation_images, templates)

# Working with flattened images:
# templates has shape (10, 784)
templates = templates.reshape(10, -1)
# validation_images has shape (N_val, 784)
validation_images = validation_images.reshape(N_val, -1)

agreement = torch.einsum('Cd,Nd->CN', validation_images, templates)
# Note: This is matrix multiplication: templates @ validation_images.T

W1, W2 = None, None
b1, b2 = None, None
def relu(x):
    return x
def sigmoid(x):
    return x

# x has shape (784,)
x = validation_images[0]

# W1 has shape (16, 784)
# h has shape (16,)
h = W1 @ x + b1
h = relu(h)

# W2 has shape (10, 16)
# y_guess has shape (10,)
y_guess = W2 @ h + b2
# Transforming to range [0, 1]
y_guess = sigmoid(y_guess)

# W2@W1 has shape (10, 784)
y_guess = (W2@W1) @ x

model = None
y_labels = None
N_train = None

# x has shape (N_train, 784)
x = train_images
# y has shape (N_train, 10)
y = nn.functional.one_hot(train_labels, num_classes=10)

# y_pred has shape (N_train, 10)
y_pred = model(x)

loss = 1/N_train * torch.sum((y_pred - y) ** 2)

def calculate_loss():
    pass
def calculate_grads():
    pass

y_pred = model(x)

loss = calculate_loss(y, y_pred)

grads = calculate_grads(y, y_pred)

W1 = W1 - 0.001 * grads['W1']
W2 = W2 - 0.001 * grads['W2']
b1 = b1 - 0.001 * grads['b1']
b2 = b2 - 0.001 * grads['b2']

W = None
dz = None

# dx is a linear operation between W and dz

# W has shape (c_out, c_in)
# dz has shape (N, c_out)
# dx should have shape (N, c_in)

dx = torch.einsum('oi,No->Nc', W, dz)
