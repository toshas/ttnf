import contextlib
import copy
import difflib
import json
import os
import sys
import zipfile

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.utils.tensorboard import SummaryWriter


def torch_save_atomic(what, path):
    path_tmp = path + '.tmp'
    torch.save(what, path_tmp)
    os.rename(path_tmp, path)


def add_filetree_to_zip(zip, dir_src, filter_filename=None, filter_dirname=None):
    dir_src = os.path.abspath(dir_src)
    dir_src_name = os.path.basename(dir_src)
    dir_src_parent_dir = os.path.dirname(dir_src)
    zip.write(dir_src, arcname=dir_src_name)
    for cur_dir, _, cur_filenames in os.walk(dir_src):
        if filter_dirname is not None and filter_dirname(os.path.basename(cur_dir)):
            continue
        if cur_dir != dir_src:
            zip.write(cur_dir, arcname=os.path.relpath(cur_dir, dir_src_parent_dir))
        for filename in cur_filenames:
            if filter_filename is not None and filter_filename(filename):
                continue
            zip.write(
                os.path.join(cur_dir, filename),
                arcname=os.path.join(os.path.relpath(cur_dir, dir_src_parent_dir), filename)
            )


def pack_source_and_configuration(cfg, dir_src, path_zip):
    dir_src = os.path.abspath(dir_src)
    cfg = copy.deepcopy(cfg.__dict__)
    del cfg['log_root']
    cfg_str = json.dumps(cfg, indent=4)
    with zipfile.ZipFile(path_zip, 'w', compression=zipfile.ZIP_DEFLATED) as zip:
        add_filetree_to_zip(
            zip,
            dir_src,
            filter_filename=lambda f: not (f.endswith('.py') or f.endswith('.txt')),
            filter_dirname=lambda d: d in ('__pycache__',),
        )
        zip.writestr('cfg.txt', cfg_str)


def pack_directory(path_dir, path_zip, filter_filename):
    path_dir = os.path.abspath(path_dir)
    with zipfile.ZipFile(path_zip, 'w', compression=zipfile.ZIP_DEFLATED) as zip:
        add_filetree_to_zip(zip, path_dir, filter_filename=filter_filename)


def diff_source_dir_and_zip(cfg, dir_src, path_zip):
    dir_src = os.path.abspath(dir_src)
    with zipfile.ZipFile(path_zip) as zip:
        for file in zip.namelist():
            if file == 'cfg.txt':
                continue
            file_info = zip.getinfo(file)
            if file_info.is_dir():
                continue
            path_src = os.path.join(os.path.dirname(dir_src), file)
            if not os.path.isfile(path_src):
                raise FileNotFoundError(path_src)
            with open(path_src) as f:
                lines_src = f.read().split('\n')
            lines_zip = zip.read(file).decode('utf-8').split('\n')
            lines_diff = list(difflib.unified_diff(lines_zip, lines_src))
            if len(lines_diff) > 0:
                raise Exception(
                    f'Source ({file}) changed - will not resume. Diff:\n' +
                    f'\n'.join(lines_diff)
                )
        cfg = copy.deepcopy(cfg.__dict__)
        del cfg['log_root']
        cfg_str = json.dumps(cfg, indent=4).split('\n')
        cfg_zip = zip.read('cfg.txt').decode('utf-8').split('\n')
        cfg_diff = list(difflib.unified_diff(cfg_zip, cfg_str))
        if len(cfg_diff) > 0:
            raise Exception(
                f'Configuration changed - will not resume. Diff:\n' +
                f'\n'.join(cfg_diff)
            )


def verify_experiment_integrity(cfg):
    dir_src = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    log_dir = os.path.join(cfg.log_root, cfg.expname)
    path_zip = os.path.join(log_dir, 'source.zip')
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    if not os.path.isfile(path_zip):
        pack_source_and_configuration(cfg, dir_src, path_zip)
    else:
        diff_source_dir_and_zip(cfg, dir_src, path_zip)


def tb_add_scalars(tb, main_tag, tag_scalar_dict, global_step=None):
    # unlike SummaryWriter.add_scalars, this function does not create a separate FileWriter per each dict entry
    for k, v in tag_scalar_dict.items():
        tag = main_tag + '/' + k
        if isinstance(v, dict):
            tb_add_scalars(tb, tag, v, global_step=global_step)
        else:
            tb.add_scalar(tag, v, global_step=global_step)


def fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd


@contextlib.contextmanager
def stderr_redirected(to=os.devnull, stderr=None):
    # https://stackoverflow.com/a/22434262/411907
    if stderr is None:
        stderr = sys.stderr

    stderr_fd = fileno(stderr)
    # copy stderr_fd before it is overwritten; `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stderr_fd), 'wb') as copied:
        stderr.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stderr_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stderr_fd)  # $ exec > to
        try:
            yield stderr  # allow code to be run with the redirected stdout
        finally:
            # restore stderr to its previous value; dup2 makes stderr_fd inheritable unconditionally
            stderr.flush()
            os.dup2(copied.fileno(), stderr_fd)  # $ exec >&copied


class SilentSummaryWriter(SummaryWriter):
    def __init__(self, *args, **kwargs):
        with stderr_redirected():
            super().__init__(*args, **kwargs)


# Misc
img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10. * torch.log(x.detach().cpu()) / torch.log(torch.Tensor([10.]))
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)


class ScramblingSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = 0
        self.ids = torch.LongTensor(np.random.permutation(self.total))

    def __iter__(self):
        return self

    def __next__(self):
        if self.curr + self.batch > self.total:
            out = self.ids[self.curr:]
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            remaining = self.batch - out.shape[0]
            out = torch.cat((out, self.ids[:remaining]), dim=0)
            self.curr = remaining
        else:
            out = self.ids[self.curr:self.curr + self.batch]
            self.curr += self.batch
        return out


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
                                        range(D - 1)])

        # Implementation according to the official code release
        # (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])

        # Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"

        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears + 1]))

        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear + 1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears + 1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear + 1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear + 1]))


# Ray helpers


def intersect_rays_aabb(rays_o, rays_d, compute_valid=False, eps=0.01):
    """
    A simple test of AABB (axis-alingned bounding box) intersection with a batch of rays.
    :param rays_o: Ray origins
    :param rays_d: Normalized ray directions
    :param compute_valid: If True, computes validity mask of rays, where valid rays intersect AABB
    :param eps: Defines AABB dilation factor (to allow few samples on the border and outside)
    :return: Near, far values for each ray and validity mask
    """
    invdirs = torch.reciprocal(rays_d)
    lb = torch.tensor([-(1. + eps)] * 3, device=rays_d.device)
    rt = torch.tensor([1. + eps] * 3, device=rays_d.device)
    t1 = (lb - rays_o) * invdirs
    t2 = (rt - rays_o) * invdirs
    near = torch.max(torch.min(t1, t2), dim=-1).values
    far = torch.min(torch.max(t1, t2), dim=-1).values
    if compute_valid:
        valid_mask = (far >= 0) & (near < far)
        return near, far, valid_mask
    return near, far


def get_rays(H, W, K, c2w, dir_center_pix=True, valid_only=True):
    """
    Creates rays forming an image of a given configuration
    :param H (int): Image height
    :param W (int): Image width
    :param K (Tensor): Camera intrinsics matrix
    :param c2w (Tensor): Camera-to-world transformation matrix
    :param dir_center_pix (bool): Cast rays through pixel centers instead of a corner
    :param valid_only (bool): Return only valid rays - those which pass through a unit volume
    :return: A tuple of:
            rays_o (Tensor N x 3): Ray origins
            rays_d (Tensor N x 3): Ray directions (unit)
            near (Tensor N x 1): Near intersection with the unit volume
            far (Tensor N x 1): Far intersection with the unit volume
            valid_mask (Tensor H x W): Validity mask or None
    """
    device = c2w.device
    i, j = torch.meshgrid(
        torch.linspace(0, W - 1, W, device=device),
        torch.linspace(0, H - 1, H, device=device),
        indexing='xy'
    )
    if dir_center_pix:
        # mipnerf and plenoxels offset rays to pass through the voxel center
        dirs = torch.stack([(i - K[0][2] + 0.5) / K[0][0], -(j - K[1][2] + 0.5) / K[1][1], -torch.ones_like(i)], -1)
    else:
        # original nerf implementation makes rays pass through the voxel corner
        dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)  # dot product eq: [c2w.dot(dir) for dir in dirs]
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    if valid_only:
        near, far, valid_mask = intersect_rays_aabb(rays_o, rays_d, compute_valid=True)
        rays_o = rays_o[valid_mask]
        rays_d = rays_d[valid_mask]
        near = near[valid_mask].unsqueeze(-1)
        far = far[valid_mask].unsqueeze(-1)
    else:
        near, far = intersect_rays_aabb(rays_o, rays_d, compute_valid=False)
        rays_o = rays_o.view(-1, 3)
        rays_d = rays_d.view(-1, 3)
        near = near.view(-1, 1)
        far = far.view(-1, 1)
        valid_mask = None
    return rays_o, rays_d, near, far, valid_mask


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)

    return rays_o, rays_d


def expand_envelope_pdf(weights):
    """
    2-tap max filter followed by a 2-tap blur filter (blurpool); c.f. MIP-NeRF
    :param weights: tensor of weights of shape [..., N], where last dimension is the unnormalized pdf
    :return: adjusted weights of the same shape.
    """
    weights = torch.nn.functional.pad(weights, (1, 1), 'replicate')  # [..., N+2]
    weights = torch.maximum(weights[..., :-1], weights[..., 1:])  # [..., N+1]
    weights = 0.5 * (weights[..., :-1] + weights[..., 1:])  # [..., N]
    return weights


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, C=1e-5, checks=True):
    """
    Get pdf
    :param bins:
    :param weights:
    :param N_samples:
    :param det:
    :param C: default prevents NaN, larger values increase exploration of low-probability areas
    :return:
    """
    if checks:
        assert torch.is_tensor(bins) and torch.is_tensor(weights)
        assert bins.shape[:-1] == weights.shape[:-1], f'{bins.shape=} {weights.shape=}'
        assert bins.shape[-1] == weights.shape[-1] + 1, f'{bins.shape=} {weights.shape=}'

    device = weights.device
    weights = weights + C
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def integrate_old(sigma, dists):
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  # NR x NS
    log_att = -sigma * dists  # NR x NS
    att = log_att.exp()  # NR x NS
    alpha = 1.0 - att  # NR x NS
    att = att + 1e-10  # NR x NS
    cum_att_exclusive = torch.cumprod(
        torch.cat([
            torch.ones((att.shape[0], 1)),  # NR x 1
            att  # NR x NS
        ], -1),  # NR x (NS + 1)
        -1)[:, :-1]  # NR x NS
    weights = alpha * cum_att_exclusive  # NR x NS
    return weights


def integrate_new(sigma, dists):
    dists = F.pad(dists, (0, 1), mode='constant', value=torch.finfo(dists.dtype).max)  # NR x NS
    log_att = -sigma * dists  # NR x NS
    log_att_pad_left = F.pad(log_att[:, :-1], (1, 0))  # NR x NS (add column zeros left, remove column right)
    att = log_att.exp()  # NR x NS
    alpha = 1.0 - att  # NR x NS
    cum_log_att_exclusive = torch.cumsum(log_att_pad_left, dim=1)  # NR x NS
    cum_att_exclusive = cum_log_att_exclusive.exp()
    weights = alpha * cum_att_exclusive  # NR x NS
    return weights


LPIPS = None


def rgb_lpips(img0, img1):
    import lpips
    global LPIPS
    assert torch.is_tensor(img0)
    assert torch.is_tensor(img1)
    assert len(img0.shape) == 3
    assert img0.shape[-1] == 3
    assert img0.shape == img1.shape
    assert img0.device == img1.device
    if LPIPS is None:
        LPIPS = lpips.LPIPS(net='vgg', verbose=False).to(img0.device)
    img0 = img0.permute([2, 0, 1]).contiguous()
    img1 = img1.permute([2, 0, 1]).contiguous()
    return LPIPS(img0, img1, normalize=True).item()


def rgb_ssim(img0, img1, max_val,
             filter_size=11,
             filter_sigma=1.5,
             k1=0.01,
             k2=0.03,
             return_map=False):
    # Modified from https://github.com/google/mipnerf/blob/16e73dfdb52044dcceb47cda5243a686391a6e0f/internal/math.py#L58
    assert type(img0) is np.ndarray
    assert type(img1) is np.ndarray
    assert len(img0.shape) == 3
    assert img0.shape[-1] == 3
    assert img0.shape == img1.shape

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma) ** 2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        import scipy.signal
        return scipy.signal.convolve2d(z, f, mode='valid')

    filt_fn = lambda z: np.stack([
        convolve2d(convolve2d(z[..., i], filt[:, None]), filt[None, :])
        for i in range(z.shape[-1])], -1)
    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0 ** 2) - mu00
    sigma11 = filt_fn(img1 ** 2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = np.maximum(0., sigma00)
    sigma11 = np.maximum(0., sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(
        np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = np.mean(ssim_map)
    return ssim_map if return_map else ssim


def positional_encoding(positions, freqs):
    freq_bands = 2 ** torch.arange(freqs, dtype=torch.float, device=positions.device)  # F
    pts = (positions[..., None] * freq_bands).reshape(
        positions.shape[:-1] + (freqs * positions.shape[-1],)
    )  # ..., D*F
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts
