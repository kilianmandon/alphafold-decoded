import torch

## Tensor Creation

# Shape (4,)
torch.tensor([2.0, 1.45, -1.0, 0.0])

# Shape (3, 2)
torch.tensor([[1, 2],[1, 2],[3, 4]])

# Shape (3, 2)
torch.tensor([
    [1, 2],
    [1, 2],
    [3, 4],
])

# Shape (4, 1)
torch.tensor([[2.0], [1.45], [-1.0], [0.0]])

# Shape (5,)
print(torch.linspace(0, 2, steps=5))
# Result: torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0])

# Shape (4, 3, 4)
torch.ones((4, 3, 4)) * 5

A = torch.tensor([3, 5, 1])
B = torch.tensor([4, 5, 8])
comparison = A < B

print(comparison)
# Result: torch.tensor([True, False, True])

as_float = comparison.float()

print(as_float)
# Result: torch.tensor([1.0, 0.0, 1.0])

## Tensor Manipulation
### Shapes
matrix = torch.tensor([
    [0,  1,  2],
    [3,  4,  5],
    [6,  7,  8],
    [9, 10, 11]
])

torch.flatten(matrix)
# Result: torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

# batch has shape (2, 2, 3)
batch = torch.tensor([
    [
        [0, 1, 2],
        [3, 4, 5]
    ],
    [
        [6,  7,  8],
        [9, 10, 11]
    ]
])

# batch has shape (2, 6)
batch = torch.tensor([
    [0, 1, 2, 3,  4,  5],
    [6, 7, 8, 9, 10, 11]
])

# batch has shape (12,)
batch = torch.tensor(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
)

# Shape (2, 3)
matrix = torch.tensor([
    [0, 1, 2],
    [3, 4, 5]
])

matrix.reshape(3, 2)

# Shape (6,)
matrix = torch.tensor([0, 1, 2, 3, 4, 5])

# Shape (3, 2)
matrix = torch.tensor([
    [0, 1],
    [2, 3],
    [4, 5]
])
### Indexing and Slicing

A = torch.tensor([
    [0, 1, 2],
    [3, 4, 5]
])

A[1, 0]
# Result: torch.tensor(3)

A[1, 0].item()
# Result: 3

T = torch.zeros((4, 4))

#      [      |   ]
# T =  [  R   | t ]
#      [      |   ] 
#      [0 0 0 | 1 ]

# R has shape (3, 3)
R = T[0:3, 0:3]
# R = T[:3, :3]

# t has shape (3,)
t = T[:3, 3]
# t = T[:3, -1]

# t has shape (3, 1)
t = T[:3, 3:4]
# t = T[:3, 3:]

T = torch.zeros((1, 4, 4))

# One transform per batch item:
# T.shape = (N, 4, 4)
R = T[:, :3, :3]

T = torch.zeros((1, 1, 4, 4))

# One transform per batch per residue:
# T.shape = (N, N_res, 4, 4)
R = T[:, :, :3, :3]

# With ellipsis:
# Works for batched, double-batched and non-batched case
R = T[..., :3, :3]

A = torch.zeros((3, 3))
# Left-hand side indexing:
# Assignment to the second column of A
A[:, 1] = torch.tensor([0, 1, 2])

# [Left-hand side] = [Right-hand side]

# Right-hand side indexing:
# Retrieval of the second column of A
col = A[:, 1]

# Left-hand side indexing:
# Assignment to the second column of A
A[:, 1] = torch.tensor([0, 1, 2])



## Computations along Axes

# Shape (4, 3)
A = torch.tensor([ #
    [0,  1,  2],   #  3
    [3,  4,  5],   # 12
    [6,  7,  8],   # 21
    [9, 10, 11]    # 30
])                 # ##
#   18  22  26     # 66

# Summing up the rows:
A.sum(dim=0)
# Result: torch.tensor([18, 22, 26])
# Shape (3,)

# Summing up the columns:
A.sum(dim=1)
# Result: torch.tensor([3, 12, 21, 30])
# Shape (4,)

# Summing up all elements:
A.sum(dim=None)
# Result: torch.tensor(66)
# Shape = (,)

A = torch.arange(12).reshape(4, 3)
A.sum(dim=(1, 2))
# A.sum(dim = (-1, -2))

# Shape (2, 3)
A = torch.tensor([
    [0, 3, 5],
    [0, 4, 1],
])

# Index of maximal elements:
A.argmax(dim=1)
# Result: torch.tensor([2, 1])
# Shape (2,)
torch.argmax(A, dim=1)

A = A.float()

torch.linalg.vector_norm(A, dim=1)

torch.mean(A, dim=1)
torch.std(A, dim=1)

# A has shape (4, 3)

# Out shape: (4, 1)
torch.argmax(A, dim=1, keepdim=True)

# Out shape: (4, 1)
torch.mean(A, dim=1, keepdim=True)
torch.std(A, dim=1, keepdim=True)

# Out shape: (4, 1)
torch.linalg.vector_norm(A, dim=1, keepdim=True)

torch.cat((R, t), dim=-1)

# Shape (4, 3)
A = torch.tensor([ #
    [0,  1,  2],   #  3
    [3,  4,  5],   # 12
    [6,  7,  8],   # 21
    [9, 10, 11]    # 30
])                 # ##
#   18  22  26     # 66

# Summing up the rows:
A.sum(dim=0)
# Result: torch.tensor([18, 22, 26])
# Shape (3,)

# With keepdim:
A.sum(dim=0, keepdim=True)
# Result: torch.tensor([[18, 22, 26]])
# Shape (1, 3)

# softmax(x)_i = exp(x_i) / sum_j exp(x_j)

# Shape (4, 3)
A = torch.tensor([ #
    [0.,  1.,  2.],   #  1 / (1 + e + e^2), e / (1 + e + e^2), e^2 / (1 + e + e^2)
    [3.,  4.,  5.],   #  e^3 / (e^3 + e^4 + e^5), ...
    [6.,  7.,  8.],   #  e^6 / (e^6 + e^7 + e^8), ...
    [9., 10., 11.]    #  e^9 / (e^9 + e^10 + e^11), ...
])

# Shape (4, 3)
res = A.softmax(dim=1)

# Shape (4,)
res.sum(dim=1)
# Result: torch.tensor([1., 1., 1., 1.])

# Shape (3,)
print(res.sum(dim=0))
# Result: torch.tensor([0.3601, 0.9789, 2.6610])

### Stacking and Concatenating
u = torch.arange(3)
v = torch.arange(3)
w = torch.range(3)

# Shape (3, 4)
torch.stack((u, v, w), dim=0)
# Result:
# torch.tensor([
#    [0, 1, 2],
#    [0, 1, 2],
#    [0, 1, 2],
# ])

# Shape (4, 3)
torch.stack((u, v, w), dim=1)
# Result:
# torch.tensor([
#    [0, 0, 0],
#    [1, 1, 1],
#    [2, 2, 2],
#    [3, 3, 3]
# ])

#      [      |   ]
# T =  [  R   | t ]
#      [      |   ] 
#      [0 0 0 | 1 ]

# R has shape (3, 3)
R = T[0:3, 0:3]
# R = T[:3, :3]

# t has shape (3, 1)
t = T[:3, 3:4]
# t = T[:3, 3:]

# Rt has shape (3, 4)
Rt = torch.cat((R, t), dim=1)



## Broadcasting

# A has shape (4, 3)
A = torch.tensor([
    [0, 0, 0],
    [0, 0, 0]
    [0, 0, 0],
    [0, 0, 0]
])

# B has shape (3,)
B = torch.tensor([1, 2, 3])

# B has shape (1, 3)
B = B.reshape(1, 3)
# B = B.unsqueeze(0)
# B = B[None, :]

# B has shape (4, 3)
B = B.broadcast_to((4, 3))

# B = torch.tensor([
#     [1, 2, 3],
#     [1, 2, 3],
#     [1, 2, 3],
#     [1, 2, 3]
# ])

# A has shape (4, 3)
A = torch.zeros((4, 3))

# B has shape (3,)
B = torch.tensor([1, 2, 3])

# B has shape (1, 3)
B = B.reshape(1, 3)

# Adding B to all Rows:
C = A+B # Implicit Broadcasting of B to (4, 3)

# A has shape (4, 3)
A = torch.zeros((4, 3))

# B has shape (4,)
B = torch.tensor([1, 2, 3, 4])

# B has shape (4, 1)
B = B.reshape(4, 1)

# Adding B to all columns:
C = A+B # Implicit Broadcasting of B to (4, 3)

# Broadcasting Rules:
#
# - All dimensions need to be identical or 1
#
# - Along these 1-dimensions, the tensors are duplicated 
#    to match the other shape
#
# - Dimensions are aligned from right to left. 
#    If one tensor has less dimensions,
#    1-dimensions are added on the left.
#
# Shape A:  4   3
# Shape B:      3
# Implicit: 1   3

N = 15
# Working on a batch of 3D vectors:
V = torch.randn((N, 3))

# norms has shape (N,)
norms = torch.linalg.vector_norm(V, dim=-1)

# norms has shape (N, 1)
norms = norms.unsqueeze(-1)

V_normalized = V / norms # Implicit broadcsting to (N, 3)

# Keeping the dimension during norm calculation
V_normalized = V / torch.linalg.vector_norm(V, dim=-1, keepdim=True) 

v = torch.tensor([0, 5, 3])
w = torch.tensor([2, 4, 0])

# Outer product:
# [v0*w0 v0*w1]    [v0 v0]   [w0 w1]
# [v1*w0 v1*w1] =  [v1 v1] * [w0 w1]
# [v2*w0 v2*w1]    [v2 v2]   [w0 w1]

v = v.reshape(3, 1)
w = w.reshape(1, 2)

outer = v*w

## torch.einsum
N = 3
i, j, k = 5, 5, 5

# A has shape (i, k) = (2, 3)
A = torch.tensor([
    [0, 1, 3],
    [4, 5, 6]
])

# B has shape (k, j) = (3, 2)
B = torch.tensor([
    [0, 3],
    [5, 2],
    [4, 5]
])

A = torch.randn(i, k)
B = torch.randn(k, j)

# Matrix Multiplication: 
C=  A @ B
# C has shape (i, j)

C = A.reshape(i, k, 1) * B.reshape(1, k, j)
# C[i, k, j] = A[i, k, j] * B[i, k, j]
#            = A[i, k]    * B[k, j]

C = torch.sum(C, dim=1)

# Matrix Multiplication:
C = torch.sum(A.reshape(i, k, 1) * B.reshape(1, k, j), dim=-2)

# Batched Matrix Multiplication:
C = torch.sum(A.reshape(N, i, k, 1) * B.reshape(N, 1, k, j), dim=-2)

# With einsum notation:
# Non-batched:
C = torch.einsum('ik,kj->ij', A, B)
# Batched:
C = torch.einsum('Nik,Nkj->Nij', A, B)
# Both:
C = torch.einsum('...ik,...kj->...ij', A, B)

# Outer product:
# [v0*w0 v0*w1]    [v0 v0]   [w0 w1]
# [v1*w0 v1*w1] =  [v1 v1] * [w0 w1]
# [v2*w0 v2*w1]    [v2 v2]   [w0 w1]

v = v.reshape(3, 1)
w = w.reshape(1, 2)

outer = v*w

# With einsum notation:
outer = torch.einsum('i,j->ij', v, w)