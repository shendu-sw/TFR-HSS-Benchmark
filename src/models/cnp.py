import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


__all__ = ["ConditionalNeuralProcess"]


class DeterministicEncoder(nn.Module):
    def __init__(self, sizes):
        super(DeterministicEncoder, self).__init__()
        self.linears = nn.ModuleList()
        for i in range(len(sizes) - 1):
            self.linears.append(nn.Linear(sizes[i], sizes[i + 1]))

    def forward(self, context_x, context_y):
        """
        Encode training set as one vector representation

        Args:
            context_x:  batch_size x set_size x feature_dim
            context_y:  batch_size x set_size x 1

        Returns:
            representation:
        """
        encoder_input = torch.cat((context_x, context_y), dim=-1)
        batch_size, set_size, filter_size = encoder_input.shape
        x = encoder_input.view(batch_size * set_size, -1)
        for i, linear in enumerate(self.linears[:-1]):
            # x = torch.relu(linear(x))
            x = F.gelu(linear(x))
        x = self.linears[-1](x)
        x = x.view(batch_size, set_size, -1)
        representation = x.mean(dim=1)
        return representation


class DeterministicDecoder(nn.Module):
    def __init__(self, sizes):
        super(DeterministicDecoder, self).__init__()
        self.linears = nn.ModuleList()
        for i in range(len(sizes) - 1):
            self.linears.append(nn.Linear(sizes[i], sizes[i + 1]))

    def forward(self, representation, target_x):
        """
        Take representation representation of current training set, and a target input x,
        return the probability of x being positive

        Args:
            representation: batch_size x representation_size
            target_x: batch_size x set_size x d
        """
        batch_size, set_size, d = target_x.shape
        representation = representation.unsqueeze(1).repeat([1, set_size, 1])
        input = torch.cat((representation, target_x), dim=-1)
        x = input.view(batch_size * set_size, -1)
        for _, linear in enumerate(self.linears[:-1]):
            # x = torch.relu(linear(x))
            x = F.gelu(linear(x))
        x = self.linears[-1](x)
        out = x.view(batch_size, set_size, -1)
        mu, log_sigma = torch.split(out, 1, dim=-1)
        sigma = 0.1 + 0.9 * torch.nn.functional.softplus(log_sigma)
        dist = torch.distributions.normal.Normal(loc=mu, scale=sigma)
        return dist, mu, sigma


class ConditionalNeuralProcess(nn.Module):
    def __init__(self, input_dim=None, output_dim=None, encoder_sizes=[2+1, 128, 128, 128, 256], decoder_sizes=[256+2, 256, 256, 128, 128, 2]):
        super(ConditionalNeuralProcess, self).__init__()
        self._encoder = DeterministicEncoder(encoder_sizes)
        self._decoder = DeterministicDecoder(decoder_sizes)

    def forward(self, x_context, y_context, x_target, y_target=None):
        representation = self._encoder(x_context, y_context)
        dist, mu, sigma = self._decoder(representation, x_target)

        log_p = None if y_target is None else dist.log_prob(y_target)
        #return log_p, mu, sigma
        return mu
        

def input_mapping(x, B):
    if B is None:
        return x
    else:
        x_proj = (2. * np.pi * x) @ B.t()
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class ConditionalNeuralProcessFourier(nn.Module):
    def __init__(self, encoder_sizes, decoder_sizes):
        super(ConditionalNeuralProcessFourier, self).__init__()
        self._encoder = DeterministicEncoder(encoder_sizes)
        self._decoder = DeterministicDecoder(decoder_sizes)
        self.mapping_size = 64
        self.B_gauss = torch.randn((self.mapping_size, 2)).cuda() * 10

    def forward(self, x_context, y_context, x_target, y_target=None):
        x_context = input_mapping(x_context, self.B_gauss)
        x_target = input_mapping(x_target, self.B_gauss)
        representation = self._encoder(x_context, y_context)
        dist, mu, sigma = self._decoder(representation, x_target)

        log_p = None if y_target is None else dist.log_prob(y_target)
        return log_p, mu, sigma