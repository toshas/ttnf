import math

import torch

from .helpers import positional_encoding
from ..core.qttnf import QTTNF
from ..core.spherical_harmonics import spherical_harmonics_bases


class ShaderBase(torch.nn.Module):
    def forward(self, coords_xyz, viewdirs, feat_color):
        """
        Takes color features, view directopns, and optionally coordinates at which the features were sampled,
        and produces rgb values for each point.
        :param coords_xyz:
        :param viewdirs:
        :param feat_color:
        :return:
        """
        pass


class ShaderSphericalHarmonics(ShaderBase):
    def __init__(self, sh_basis_dim, checks=False):
        super().__init__()
        self.sh_basis_dim = sh_basis_dim
        self.checks = checks

    def forward(self, coords_xyz, viewdirs, feat_rgb):
        """
        :param coords_xyz (torch.Tensor): sampled points of shape [batch x ray x 3]
        :param viewdirs (torch.Tensor): directions corresponding to inputs rays of shape [batch x 3]
        :param feat_rgb (torch.Tensor): directions corresponding to inputs rays of shape [batch x ray x rgb_feat_dim]
        :return:
        """
        if self.checks:
            assert feat_rgb.dim() == 3
            B, R, X = feat_rgb.shape
            assert X == 3 * self.sh_basis_dim
            assert viewdirs.shape == (B, 3)
        else:
            B, R, _ = feat_rgb.shape

        rgb = feat_rgb.view(B, R, 3, self.sh_basis_dim)  # B x R x 3 x SH
        sh_mult = spherical_harmonics_bases(self.sh_basis_dim, viewdirs)  # B x SH
        sh_mult = sh_mult.view(B, 1, 1, self.sh_basis_dim)  # B x 1 x 1 x SH
        rgb = (rgb * sh_mult).sum(dim=-1)  # B x R x 3

        return rgb


class ShaderTensorF(torch.nn.Module):
    def __init__(self, rgb_feature_dim, posenc_viewdirs=2, posenc_feat=2, dim_latent=128, checks=False):
        super().__init__()
        self.posenc_viewdirs = posenc_viewdirs
        self.posenc_feat = posenc_feat
        self.checks = checks

        in_mlpC = (2 * posenc_viewdirs + 1) * 3 + (2 * posenc_feat + 1) * rgb_feature_dim
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_mlpC, dim_latent),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(dim_latent, dim_latent),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(dim_latent, 3),
        )
        torch.nn.init.constant_(self.mlp[-1].bias, 0)
        with torch.no_grad():
            mag_scale = 10
            num_linear = 3
            scale_linear = math.pow(mag_scale, 1 / num_linear)
            for layer in self.mlp:
                if isinstance(layer, torch.nn.Linear):
                    layer.weight *= scale_linear


    def forward(self, coords_xyz, viewdirs, feat_rgb):
        """
        :param coords_xyz (torch.Tensor): sampled points of shape [batch x ray x 3]
        :param viewdirs (torch.Tensor): directions corresponding to inputs rays of shape [batch x 3]
        :param feat_rgb (torch.Tensor): directions corresponding to inputs rays of shape [batch x ray x rgb_feat_dim]
        :return:
        """
        if self.checks:
            assert feat_rgb.dim() == 3
            B, R, X = feat_rgb.shape
            assert X == 3 * self.sh_basis_dim
            assert viewdirs.shape == (B, 3)
        else:
            B, R, _ = feat_rgb.shape

        viewdirs = viewdirs.view(-1, 1, 3).repeat(1, R, 1)
        indata = [feat_rgb, viewdirs]
        if self.posenc_feat > 0:
            indata += [positional_encoding(feat_rgb, self.posenc_feat)]
        if self.posenc_viewdirs > 0:
            indata += [positional_encoding(viewdirs, self.posenc_viewdirs)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)

        return rgb


class RadianceFieldTT(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        if args.shading_mode == 'spherical_harmonics':
            self.shader = ShaderSphericalHarmonics(args.sh_basis_dim, checks=args.checks)
        elif args.shading_mode == 'mlp':
            self.shader = ShaderTensorF(args.rgb_feature_dim)
        else:
            raise ValueError(f'Invalid shading mode "{args.shading_mode}"')
        self.shader_num_params = sum(torch.tensor(a.shape).prod().item() for a in self.shader.parameters())

        kwargs = dict(
            tt_rank_equal=args.tt_rank_equal,
            tt_minimal_dof=args.tt_minimal_dof,
            init_method=args.init_method,
            outliers_handling='zeros',
            expected_sample_batch_size=args.N_rand * (args.N_samples + args.N_importance),
            version_sample_qtt=args.sample_qtt_version,
            dtype={
                'float16': torch.float16,
                'float32': torch.float32,
                'float64': torch.float64,
            }[args.dtype],
            checks=args.checks,
            verbose=True,
        )

        if args.grid_tt_type == 'fused':
            # opacity + 3 * (# sh or a float per channel)
            dim_payload = 1 + 3 * (args.sh_basis_dim if args.use_viewdirs else 1)
            self.vox_fused = QTTNF(
                args.dim_grid, dim_payload, args.tt_rank_max, sample_by_contraction=args.sample_by_contraction, **kwargs
            )
        elif args.grid_tt_type == 'separate':
            # 3 * (# sh or a float per channel)
            dim_payload = 3 * (args.sh_basis_dim if args.use_viewdirs else 1)
            self.vox_rgb = QTTNF(
                args.dim_grid, dim_payload, args.tt_rank_max, sample_by_contraction=args.sample_by_contraction, **kwargs
            )
            self.vox_sigma = QTTNF(args.dim_grid, 1, args.tt_rank_max, **kwargs)  # opacity
        else:
            raise ValueError(f'Invalid voxel grid type "{args.grid_tt_type}"')

    def forward(self, coords_xyz, viewdirs):
        if self.args.checks:
            assert coords_xyz.dim() == 3 and coords_xyz.shape[2] == 3

        B, R, _ = coords_xyz.shape
        coords_xyz = coords_xyz.view(B * R, 3)

        if self.args.grid_tt_type == 'fused':
            tmp = self.vox_fused(coords_xyz)
            rgb, sigma = tmp[..., :-1], tmp[..., -1]  # B x R x 3 * (SH or 1), B x R
        elif self.args.grid_tt_type == 'separate':
            rgb = self.vox_rgb(coords_xyz)
            sigma = self.vox_sigma(coords_xyz)
        else:
            raise ValueError(f'Invalid grid type: "{self.args.grid_tt_type}"')

        rgb, sigma = rgb.view(B, R, -1), sigma.view(B, R)
        rgb = self.shader(coords_xyz, viewdirs, rgb)
        return rgb, sigma

    def get_param_groups(self):
        out = []
        if self.args.grid_tt_type == 'fused':
            out += [
                {'tag': 'vox', 'params': self.vox_fused.parameters(), 'lr': self.args.lrate},
            ]
        elif self.args.grid_tt_type == 'separate':
            out += [
                {'tag': 'vox', 'params': self.vox_rgb.parameters(), 'lr': self.args.lrate},
                {'tag': 'vox', 'params': self.vox_sigma.parameters(), 'lr':
                    self.args.lrate * self.args.lrate_sigma_multiplier},
            ]
        out += [
            {'tag': 'shader', 'params': self.shader.parameters(), 'lr': self.args.lrate_shader},
        ]
        return out

    @property
    def num_uncompressed_params(self):
        if self.args.grid_tt_type == 'fused':
            return self.vox_fused.num_uncompressed_params
        elif self.args.grid_tt_type == 'separate':
            return self.vox_rgb.num_uncompressed_params + self.vox_sigma.num_uncompressed_params

    @property
    def num_compressed_params(self):
        out = self.shader_num_params
        if self.args.grid_tt_type == 'fused':
            out += self.vox_fused.num_compressed_params
        elif self.args.grid_tt_type == 'separate':
            out += self.vox_rgb.num_compressed_params + self.vox_sigma.num_compressed_params
        return out

    @property
    def sz_uncompressed_gb(self):
        if self.args.grid_tt_type == 'fused':
            return self.vox_fused.sz_uncompressed_gb
        elif self.args.grid_tt_type == 'separate':
            return self.vox_rgb.sz_uncompressed_gb + self.vox_sigma.sz_uncompressed_gb

    @property
    def sz_compressed_gb(self):
        if self.args.grid_tt_type == 'fused':
            return self.vox_fused.sz_compressed_gb
        elif self.args.grid_tt_type == 'separate':
            return self.vox_rgb.sz_compressed_gb + self.vox_sigma.sz_compressed_gb

    @property
    def compression_factor(self):
        return self.num_uncompressed_params / self.num_compressed_params
