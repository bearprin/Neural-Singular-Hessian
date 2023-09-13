import numpy as np

import torch
import torch.nn as nn


def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')


def frequency_init(freq):
    def init(m):
        with torch.no_grad():
            # if hasattr(m, 'weight'):
            if isinstance(m, nn.Linear):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-np.sqrt(6 / num_input) / freq, np.sqrt(6 / num_input) / freq)

    return init


def first_layer_film_sine_init(m):
    with torch.no_grad():
        # if hasattr(m, 'weight'):
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


class MappingNet(nn.Module):
    def __init__(self, dim: int, out_dim_per_layer: int, n_out_layers: int = 1, hidden_size: int = 256):
        super().__init__()
        self.dim: int = dim
        self.out_dim_per_layer: int = out_dim_per_layer
        self.n_out_layers: int = n_out_layers
        self.hidden_size: int = hidden_size

        last_length = self.out_dim_per_layer * 2 * self.n_out_layers
        self.net = nn.Sequential(nn.Linear(self.dim, self.hidden_size), nn.LeakyReLU(0.2, inplace=True),
                                 nn.Linear(self.hidden_size, self.hidden_size), nn.LeakyReLU(0.2, inplace=True),
                                 nn.Linear(self.hidden_size, self.hidden_size), nn.LeakyReLU(0.2, inplace=True),
                                 nn.Linear(self.hidden_size, last_length)
                                 )

        self.net.apply(kaiming_leaky_init)
        with torch.no_grad():
            self.net[-1].weight *= 0.25

    def forward(self, x):
        """
        Args:
            x (B, ..., dim)
        Returns:
            frequencies: (B, ..., n_out_layers, out_dim_per_layer)
            biases (B, ..., n_out_layers, out_dim_per_layer)
        """
        output = self.net(x).view(x.shape[:-1] + (self.n_out_layers, self.out_dim_per_layer * 2))
        freqs, biases = output.split(self.out_dim_per_layer, dim=-1)

        return freqs, biases


class FiLMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)

    def forward(self, x, freq, phase_shift):
        """
        sin(freq * lin(x) + phase_shift)
        Args:
            x (B, ..., input_dim)
            freq: (B,...,hidden_dim)
            phase_shift: (B,...,hidden_dim)
        Returns
            output (B, ..., hidden_dim)
        """
        x = self.layer(x)
        assert (freq.shape == phase_shift.shape)
        if freq.ndim == (x.ndim - 1):
            freq = freq.unsqueeze(-2)
            phase_shift = phase_shift.unsqueeze(-2)

        return torch.sin(freq * x + phase_shift)


class FilmSiren(nn.Module):

    def __init__(self, hidden_size, n_layers, c_dim=32):
        super().__init__()
        self.hidden_size: int = hidden_size
        self.mapping_hidden_size: int = hidden_size
        self.n_layers: int = n_layers
        self.dim: int = 3
        self.c_dim: int = c_dim
        self.out_dim: int = 1
        self.share_frequencies: bool = False
        self.base_omega: int = 30

        self._initialize()

    def _initialize(self):
        if self.c_dim > 0:
            self.frequency_net = MappingNet(self.c_dim, self.hidden_size,
                                            n_out_layers=(1 if self.share_frequencies else self.n_layers),
                                            hidden_size=self.mapping_hidden_size)

        sirens = []
        for i in range(self.n_layers):
            if i == 0:
                lin = FiLMLayer(self.dim, self.hidden_size)
            else:
                lin = FiLMLayer(self.hidden_size, self.hidden_size)
            sirens.append(lin)

        self.sirens = nn.ModuleList(sirens)

        self.final_linear = nn.Linear(self.hidden_size, self.out_dim)

        self.sirens.apply(frequency_init(self.base_omega))
        self.final_linear.apply(frequency_init(self.base_omega))
        self.sirens[0].apply(first_layer_film_sine_init)
        self.epoch = 0

    def forward(self, query_points, query_feat=None):
        coords = query_points
        c = query_feat
        # mappingnet (B,...,n_layers,hidden_size)
        frequencies, biases = self.frequency_net(c)
        frequencies = frequencies * self.base_omega / 2 + self.base_omega

        if frequencies.shape[-2] != self.n_layers:
            shp = frequencies.shape
            new_shp = shp[:-2] + (self.n_layers,) + shp[-1:]
            frequencies = frequencies.expand(new_shp)
            biases = biases.expand(new_shp)

        x = coords
        _it = 0
        for layer, gamma, beta in zip(self.sirens, frequencies.unbind(dim=-2), biases.unbind(dim=-2)):
            x = layer(x, gamma, beta)
            _it += 1

        x = self.final_linear(x)

        return x  # sdf
