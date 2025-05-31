"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

from typing import Callable, List, Union

import numpy as np
import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

try:
    import tinycudann as tcnn
except ImportError as e:
    print(
        f"Error: {e}! "
        "Please install tinycudann by: "
        "pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
    )
    exit()


class _TruncExp(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))


trunc_exp = _TruncExp.apply


def contract_to_unisphere(
    x: torch.Tensor,
    aabb: torch.Tensor,
    ord: Union[str, int] = 2,
    #  ord: Union[float, int] = float("inf"),
    eps: float = 1e-6,
    derivative: bool = False,
):
    aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
    x = (x - aabb_min) / (aabb_max - aabb_min)
    x = x * 2 - 1  # aabb is at [-1, 1]
    mag = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
    mask = mag.squeeze(-1) > 1

    if derivative:
        dev = (2 * mag - 1) / mag**2 + 2 * x**2 * (
            1 / mag**3 - (2 * mag - 1) / mag**4
        )
        dev[~mask] = 1.0
        dev = torch.clamp(dev, min=eps)
        return dev
    else:
        x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
        x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
        return x

class Embedding(torch.nn.Module):
    def __init__(self, in_channels, N_freqs=0, logscale=True):
        """Using sin and cos as the nolinear metrics
        """
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos] # , torch.cos
        self.out_channels = in_channels*(len(self.funcs)*N_freqs+1)

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def __repr__(self):
        funcs_names = ', '.join(func.__name__ for func in self.funcs)
        return f"positional embedding:{funcs_names}({self.freq_bands.tolist()}*x)"

    def forward(self, x):
        out = [x]
        out += [torch.sin(x)]
        out += [torch.cos(x)]
        # for freq in self.freq_bands:
        #     for func in self.funcs:
        #         out += [func(freq*x)]
                # out += [(func(freq*x)+1.0)/2.0]
        # return torch.cat(out, -1)
        return torch.stack(out, dim=0)
        

class NGPRadianceField(torch.nn.Module):
    """Instance-NGP Radiance Field"""

    def __init__(
        self,
        aabb: Union[torch.Tensor, List[float]],
        num_dim: int = 3,
        use_viewdirs: bool = True,
        density_activation: Callable = lambda x: trunc_exp(x - 1),
        unbounded: bool = False,
        base_resolution: int = 16, # 16
        max_resolution: int = 4096, # 4096
        geo_feat_dim: int = 15,
        n_levels: int = 16, # 16
        log2_hashmap_size: int = 19, # 19
    ) -> None:
        super().__init__()
        if not isinstance(aabb, torch.Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32)

        center = (aabb[..., :num_dim] + aabb[..., num_dim:]) / 2.0
        size = (aabb[..., num_dim:] - aabb[..., :num_dim]).max()
        aabb = torch.cat([center - size / 2.0, center + size / 2.0], dim=-1)

        self.register_buffer("aabb", aabb)
        self.num_dim = num_dim
        self.use_viewdirs = use_viewdirs
        self.density_activation = density_activation
        self.unbounded = unbounded
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution
        self.geo_feat_dim = geo_feat_dim
        self.n_levels = n_levels
        self.log2_hashmap_size = log2_hashmap_size

        per_level_scale = np.exp(
            (np.log(max_resolution) - np.log(base_resolution)) / (n_levels - 1)
        ).tolist()
        

        self.embedding = Embedding(3)
        print(
            f'hash info: base_reso={base_resolution} '
            f'up_scale={per_level_scale:.5f} '
            f'per_channels={2} hash_lengh=2^{log2_hashmap_size} '
            f'levels={n_levels}'
        )
        self.xyz_encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "Grid",
	            "type": "Hash",
                "n_levels": n_levels,
                "n_features_per_level": 2,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale,
                "interpolation": "Linear"
            },
            # dtype=torch.float32
        )
        self.sigma_net = tcnn.Network(
            n_input_dims=self.xyz_encoder.n_output_dims,
            n_output_dims=1 + self.geo_feat_dim, #
            network_config={
                "otype": "FullyFusedMLP", # FullyFusedMLP
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            }
        )

        from .nrff import RenderingNet
        self.mlp_head=RenderingNet(8, 16, data_dim_color=self.xyz_encoder.n_output_dims, featureC=256, 
                                   device='cuda', view_pe=3, 
                                   btn_freq=[3e-1, 1e1],) # NRFF
        
        print("Model Information:")
        print(self)
    
    def get_optparam_groups(self, lr_init_grid = 1e-2, lr_init_network = 1e-3):
        grad_vars = []
        grad_vars += [{'params': self.xyz_encoder.parameters(), 'lr': lr_init_grid}]
        grad_vars += [{'params': self.sigma_net.parameters(), 'lr': lr_init_grid}]
        grad_vars += [{'params': self.mlp_head.parameters(), 'lr':lr_init_network}]
        return grad_vars

    def query_density(self, x, return_feat: bool = False):
        if self.unbounded:
            x = contract_to_unisphere(x, self.aabb)
        else:
            aabb_min, aabb_max = torch.split(self.aabb, self.num_dim, dim=-1)
            x = (x - aabb_min) / (aabb_max - aabb_min)
        selector = ((x > 0.0) & (x < 1.0)).all(dim=-1)
        x_viewed = x.view(-1, self.num_dim)
        batch_size = x_viewed.shape[0]
        coord_pe = self.embedding(x_viewed)

        encoded_outputs = torch.stack([self.xyz_encoder(coord_pe[i]) for i in range(coord_pe.size(0))], dim=1)
        encoded_outputs = torch.sum(encoded_outputs, dim=1).float()

        h = self.sigma_net(encoded_outputs)
        x = h.view(list(x.shape[:-1]) + [1 + self.geo_feat_dim]).to(x)

        density_before_activation, base_mlp_out = torch.split(
            x, [1, self.geo_feat_dim], dim=-1
        )
        density = (
            self.density_activation(density_before_activation)
            * selector[..., None]
        )
        if return_feat:
            return density, encoded_outputs # for NRFF 
        else:
            return density

    def _query_rgb(self, dir, embedding, apply_act: bool = True):
        if self.use_viewdirs:
            dir = (dir + 1.0) / 2.0
            d = self.direction_encoding(dir.reshape(-1, dir.shape[-1]))
            h = torch.cat([d, embedding.reshape(-1, self.geo_feat_dim)], dim=-1)
        else:
            h = embedding.reshape(-1, self.geo_feat_dim)
        rgb = (
            self.mlp_head(h)
            .reshape(list(embedding.shape[:-1]) + [3])
            .to(embedding)
        )
        if apply_act:
            rgb = torch.sigmoid(rgb)
        return rgb


    def forward(
        self,
        positions: torch.Tensor,
        directions: torch.Tensor = None,
    ):
        if self.use_viewdirs and (directions is not None):
            assert (
                positions.shape == directions.shape
            ), f"{positions.shape} v.s. {directions.shape}"
        density, embedding = self.query_density(positions, return_feat=True)
        rgb, normals = self.mlp_head(positions, directions, embedding) # NRFF
        # rgb = self._query_rgb(directions, embedding=embedding) # MLP
        return rgb, density  # type: ignore


class NGPDensityField(torch.nn.Module):
    """Instance-NGP Density Field used for resampling"""

    def __init__(
        self,
        aabb: Union[torch.Tensor, List[float]],
        num_dim: int = 3,
        density_activation: Callable = lambda x: trunc_exp(x - 1),
        unbounded: bool = False,
        base_resolution: int = 16,
        max_resolution: int = 128,
        n_levels: int = 5,
        log2_hashmap_size: int = 17,
    ) -> None:
        super().__init__()
        if not isinstance(aabb, torch.Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        self.register_buffer("aabb", aabb)
        self.num_dim = num_dim
        self.density_activation = density_activation
        self.unbounded = unbounded
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution
        self.n_levels = n_levels
        self.log2_hashmap_size = log2_hashmap_size

        per_level_scale = np.exp(
            (np.log(max_resolution) - np.log(base_resolution)) / (n_levels - 1)
        ).tolist()
        '''self.mlp_base = tcnn.NetworkWithInputEncoding(
            n_input_dims=num_dim,
            n_output_dims=1,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": 2,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale,
            },
            network_config={
                "otype": "CutlassMLP", # FullyFusedMLP
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        )
        '''
        self.embedding = Embedding(3)
        print(
            f'hash info: base_reso={base_resolution} '
            f'up_scale={per_level_scale:.5f} '
            f'per_channels={2} hash_lengh=2^{log2_hashmap_size} '
            f'levels={n_levels}'
        )
        self.xyz_encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "Grid",
	            "type": "Hash",
                "n_levels": n_levels,
                "n_features_per_level": 2,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale,
                "interpolation": "Linear"
            }
        )
        self.sigma_net = tcnn.Network(
            n_input_dims=self.xyz_encoder.n_output_dims,
            n_output_dims=1,
            network_config={
                "otype": "CutlassMLP", # FullyFusedMLP
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            }
        )
        print("DensityField Information:")
        print(self)

    def get_optparam_groups(self, lr_init_grid = 1e-2, lr_init_network = 1e-3):
        grad_vars = []
        grad_vars += [{'params': self.xyz_encoder.parameters(), 'lr': lr_init_grid}]
        grad_vars += [{'params': self.sigma_net.parameters(), 'lr':lr_init_network}]
        return grad_vars

    def forward(self, positions: torch.Tensor):
        if self.unbounded:
            positions = contract_to_unisphere(positions, self.aabb)
        else:
            aabb_min, aabb_max = torch.split(self.aabb, self.num_dim, dim=-1)
            positions = (positions - aabb_min) / (aabb_max - aabb_min)
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions_viewed = positions.view(-1, self.num_dim)
        coord_pe = self.embedding(positions_viewed)
        encoded_outputs = torch.stack([self.xyz_encoder(coord_pe[i]) for i in range(coord_pe.size(0))], dim=1)
        encoded_outputs = torch.sum(encoded_outputs, dim=1).float()
        h = self.sigma_net(encoded_outputs)
        density_before_activation = h.view(list(positions.shape[:-1]) + [1]).to(positions)

        density = (
            self.density_activation(density_before_activation)
            * selector[..., None]
        )
        return density
