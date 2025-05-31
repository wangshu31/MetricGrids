"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

from typing import Callable, List, Union

import numpy as np
import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
from termcolor import colored

import torch.nn as nn
from .siren import SirenLayer, Sine
import util_misc

try:
    import tinycudann as tcnn
except ImportError as e:
    print(
        f"Error: {e}! "
        "Please install tinycudann by: "
        "pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
    )
    exit()

class modulator(nn.Module):

    def __init__(self,
            in_dim,
            out_dim,
            use_bias=True,
            w0=30.0
                 ):
        super().__init__()  
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_bias = use_bias
        self.act_fn = Sine(w0=w0)
        self.linear = nn.Linear(in_dim, out_dim, bias=use_bias)

    def forward(self, x):
        return self.act_fn(self.linear(x))

class MertricEmbedding(torch.nn.Module):
    def __init__(self, in_channels, N_freqs=0, logscale=True):

        super(MertricEmbedding, self).__init__()
        # original p.e.
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin] # , torch.cos
        self.out_channels = in_channels*(len(self.funcs)*N_freqs+1)

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)
    

    def forward(self, x):
        out = [(x+1.0)/2.0] # range[0,1] for different storage lengths
        sin_1 = torch.sin(torch.tensor(1.0))
        out += [(torch.sin(x)-sin_1)/(2.0 * sin_1)] # range[-1,0]
        out += [(torch.arcsin(x) - (3.0*torch.pi / 2.0)) / (2.0 * torch.pi)] # range[-1,-0.5]
        return torch.stack(out, dim=0)

class NGP(torch.nn.Module):
    """Instance-NGP Radiance Field"""

    def __init__(self,
                 cmin,
                 cmax, 
                 encoding="hashgrid", 
                 num_layers=3, 
                 skips=[], 
                 hidden_dim=64, 
                 in_dim=2, 
                 out_dim=3, 
                 num_levels=16, 
                 level_dim=2, 
                 base_resolution=16, 
                 log2_hashmap_size=24, 
                 desired_resolution=2048, 
                 act='relu', 
                 lc_act='relu', 
                 lc_init=1e-4, 
                 lca_init=None, 
                 w_init=None, 
                 b_init=None, 
                 a_init=None,
                 pe_freqs=[],
                 levels_omit=[],
                 ):
        super().__init__()

        if not isinstance(cmin, list):
            cmin = [cmin] * in_dim
        if not isinstance(cmax, list):
            cmax = [cmax] * in_dim
        self.register_buffer('cmin', torch.tensor(cmin))  # ... y x
        self.register_buffer('cmax', torch.tensor(cmax))  # ... y x

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.base_resolution = base_resolution
        self.max_resolution = desired_resolution
        self.n_levels = num_levels
        self.F = level_dim
        self.log2_hashmap_size = log2_hashmap_size
        self.num_layers = num_layers
        self.dim_hidden = hidden_dim
        
        per_level_scale = np.exp(
            (np.log(self.max_resolution) - np.log(self.base_resolution)) / (self.n_levels - 1)
        ).tolist()

        self.embedding = MertricEmbedding(self.in_dim)
        print(
            f'hash INFO: base_reso={self.base_resolution} '
            f'max_reso={self.max_resolution} up_sacle={per_level_scale:5f}'
            f'per_channels={2} hash_lengh=2^{self.log2_hashmap_size} '
            f'levels={self.n_levels} '
        )
        self.encoder = tcnn.Encoding(
                n_input_dims=self.in_dim,
                encoding_config={
                    "otype": "Grid",
                    "type": "Hash",
                    "n_levels": self.n_levels,
                    "n_features_per_level": self.F,
                    "log2_hashmap_size": self.log2_hashmap_size,
                    "base_resolution": self.base_resolution,
                    "per_level_scale": per_level_scale,
                    "interpolation": "Linear"
                },
                dtype=torch.float
            )

        # Decoder
        self.w0 = 30.0
        self.use_bias = True
        self.omegas = 30
        layers = []
        for ind in range(self.num_layers - 1):
            is_first = ind == 0
            layer_in_dim = self.encoder.n_output_dims if is_first else self.dim_hidden
            layers.append(
                SirenLayer(
                    in_dim=layer_in_dim,
                    out_dim=self.dim_hidden,
                    w0=self.w0,
                    use_bias=self.use_bias,
                is_first=is_first,
                )
            )
        self.net = nn.Sequential(*layers)
        self.last_layer = SirenLayer(
            in_dim=self.dim_hidden, out_dim=self.out_dim, w0=self.w0, use_bias=self.use_bias, is_last=True
        )
        self.modulators = nn.ModuleList(
            [modulator(in_dim=self.in_dim if i==0 else self.encoder.n_output_dims,
                                   out_dim=self.dim_hidden, w0=self.omegas)
             for i in range(self.num_layers-1)])
        
        print(colored("INFO:Model Information", "cyan", None, ['bold']))
        print(self)

    def get_optparam_groups(self, lr_init_grid = 1e-3, lr_init_network = 1e-4):
        grad_vars = []
        grad_vars += [{'params': self.encoder.parameters(), 'lr': lr_init_grid}]
        grad_vars += [{'params': self.net.parameters(), 'lr': lr_init_network}]
        grad_vars += [{'params': self.last_layer.parameters(), 'lr':lr_init_network}]
        grad_vars += [{'params': self.modulators.parameters(), 'lr':lr_init_network}]
        return grad_vars

    def count_params(self):
        params = {}
        params['hash_grid'] = util_misc.count_parameters(self.encoder)
        params['decoder'] = [a + b + c for a, b, c in zip(util_misc.count_parameters(self.net), 
                                                                util_misc.count_parameters(self.modulators), 
                                                                util_misc.count_parameters(self.last_layer))]
        params['total'] = sum([v for _, v in params.items()])
        for k, v in params.items():
            print(f'{k} params: {v}')
        return params

    def forward(self, x,step=None, exp_name=None, **kwargs):
        out_other = {}
        x = x/self.cmax.flip(-1)[None]
        x_flatten = x.view(-1, self.in_dim)
        x_flatten = (x_flatten+1.0)/2.0
        coord_pe = self.embedding(x)
        encoded_outputs = torch.stack([self.encoder(coord_pe[i]) for i in range(coord_pe.size(0))], dim=1)
        x = torch.sum(encoded_outputs, dim=1).float()
        encoded_outputs = encoded_outputs.permute(1, 0, 2).contiguous() # [num_grid, b, F*num_levels]
        for i, layer in enumerate(self.net):
            if i==0:
                modulate = self.modulators[i](x_flatten)
            elif encoded_outputs.shape[0]>=i:
                modulate = self.modulators[i]((encoded_outputs[i-1].float()))
            else:
                modulate = self.modulators[i]((encoded_outputs[-1].float()))
            backbone = layer(x)
            modulate = modulate*modulate
            if i==0:
                x = modulate*backbone
            else:
                x = modulate*backbone

        out = self.last_layer(x)
        return out, out_other