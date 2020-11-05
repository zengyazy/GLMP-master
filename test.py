import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

a = torch.tensor([[[1]]])
b = a.squeeze(0)
print(b)