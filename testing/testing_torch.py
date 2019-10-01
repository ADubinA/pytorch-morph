import torch
from tools.tools import batch_duplication

import matplotlib.pyplot as plt


def test_batch_duplication():

    t = torch.FloatTensor([[1,2,3],[4,5,6],[7,8,9]])
    print(t.shape)
    s = batch_duplication(t,5)
    print(s.shape)
    assert t.all == s[0].all
    assert t.shape == s.shape[1:]


def test_grad():
    # create vector field
    t = torch.FloatTensor([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                           [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                           [[1, 2, 3], [4, 5, 6], [7, 8, 9]]],

                         [[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                          [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                          [[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]
    )

    # create batch of 2 vector fields
    new_shape = list(t.shape)
    new_shape.insert(0, 2)
    new_shape = tuple(new_shape)
    t = torch.cat((t, t)).view(new_shape)
    print(t.shape)

    from metrics import grad as grad
    s = t+1
    assert (grad(t) - grad(s)).bool().all()


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

if __name__ == "__main__":
    test_grad()