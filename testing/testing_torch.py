import torch
from tools.tools import batch_duplication
t = torch.FloatTensor([[1,2,3],[4,5,6],[7,8,9]])
print(t)
#
print(batch_duplication(t,5).shape)
# t = t.repeat(2,4)
# print(t)

# t = torch.rand(12, 128, 128, 64)
# t[0] = torch.ones(128, 128, 64)
# t = t.repeat(2, 4)
# print(t[0])
# batch_size = 10
#
# x = torch.ones([128, 128, 64])
# # x = x.repeat(batch_size, 1, 1, 1)
#
# y = torch.zeros([batch_size, 128, 128, 64])
# y[0] = x
#
# atlas = torch.split(y,[1,batch_size-1])
# z = y + atlas
# print(z.shape)

