import torch
from torch import nn
import math
import numpy as np
from core.config import config

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

# extracted from : https://github.com/pytorch/pytorch/issues/19808#
# new sequental layer that can handle multiple inputs as a single tuple
class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                if len(inputs) == 1:
                    #print(len(inputs[0]), inputs[0][0].shape, inputs[0][1].shape)
                    inputs = module(*inputs[0])
                else:
                    #print(len(inputs), inputs[0].shape, inputs[1].shape)
                    inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

class pos_embedding(nn.Module):
    def __init__(self,seq_len, dim):
        super(pos_embedding, self).__init__()
        self.pos = np.arange(seq_len)/seq_len
        d = np.arange(dim)
        d = (2 * ((d//2) / dim))
        self.pos = np.expand_dims(self.pos, axis=1)
        self.pos = self.pos / (1e4 ** d)
        self.pos[:, ::2] = np.sin(self.pos[:, ::2]*2*math.pi)
        self.pos[:, 1::2] = np.cos(self.pos[:, 1::2]*2*math.pi)
        self.pos = torch.FloatTensor(self.pos).unsqueeze(0)
    def forward(self,embeddings):
        return embeddings + self.pos[:, :embeddings.size(1)].to(embeddings.device)

