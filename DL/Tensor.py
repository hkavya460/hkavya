import torch
import numpy as np
from torch import Tensor


#we use tensors to encode the inputs and outputs of a model,
# as well as the model’s parameters.

data = [[1,2],[3,4]]
x_data = torch.tensor(data)
print(x_data.shape)

#from numpy cretae tensor data
np_data = np.array(data)
x_np = torch.from_numpy(np_data)
# print(x_np)

other = torch.ones_like(x_data) # creating ones in the tensor created data
# print(other)

shape = (2,3)
rand_data = torch.rand(shape)
ones_data = torch.ones(shape)
# print(ones_data)

#operations on tensor

tensor = torch.ones(4,4)
tensor[:,1] = 0
print(tensor)
print(f"first Row :{tensor[0]}")
print(f"first col :{tensor[:,0]}")

#concta using cat
t1 = torch.cat([tensor,tensor,tensor])
t2 = torch.cat([tensor,tensor,tensor],dim=1)
print(t1.shape)
print(t2.shape)

#arithmetic operation "@"

y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(tensor)
print(y3.shape)
y4 = torch.matmul(tensor,tensor.T,out=y3)
print(y4)

#multiplication element wise
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
z4 = torch.mul(tensor,tensor,out=z3)
# print(z1,z2,z3,z4)

#agg function to aggregate

s = tensor.sum()
agg_item = s.item()
print(agg_item)