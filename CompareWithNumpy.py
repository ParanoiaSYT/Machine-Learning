import torch
import numpy as np


print(np.arange(6))
np_data=np.arange(6).reshape(2,3)
torch_data=torch.from_numpy(np_data)

print(
    '\ntorch_data',torch_data,
     '\nnp_data', np_data,
)


data=[1,2,-1,-2]
tensor=torch.FloatTensor(data)     #转化为32bit,torch里运算只认tensor形式
print(tensor)

print(
    np.abs(data),
    torch.abs(tensor),
)

data1=[[2,3],[5,9]]
tensor1=torch.FloatTensor(data1)
print(np.matmul(data1,data1))       #numpy里矩阵乘法是matmul
print(torch.mm(tensor1,tensor1))    #torch里矩阵乘法是mm


# 下面是.dot的小技巧，这里两者不同！注意甄别（矩阵乘法和卷积）
data2=np.array(data1)
print(data2.dot(data2))

print(tensor.dot(tensor))
print(tensor1.dot(tensor1))