import torch.nn as nn
import torch
import torch.nn.functional as F

# a = torch.randn(2,3)
# liner = nn.Linear(3,3)
# b = liner(a)
# result = F.log_softmax(b,dim=1)
# label = torch.Tensor([2,0]).long()
# loss = F.nll_loss(result,label)
# loss.backward()
# print(liner.parameters())

a = torch.rand(2,3)
b =a.transpose(0,1)
c = b[0]
d = a.view(-1)
f = F.log_softmax(torch.cat((c, d),dim=0),dim=0)
f = f.unsqueeze(0)
# print(c)
# print(d)
print(f)
label = torch.Tensor([2]).long()
print(label)
loss = F.nll_loss(f,label)
loss.backward()