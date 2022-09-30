import json
import math

import tntorch
from tqdm import tqdm

from ..core.ttnf import TTNF
from ..core.tt_core import *


def flops_svd(shape):
    """
    From "Matrix Computations" book, 4th edition, page 493, Fig(Tab). 8.6.1, last row:
    For reduced SVD with outputs (Sigma, U_1, V),
    - Golub-Reinsch SVD: 14mn^2 + 8n^3
    - R-SVD: 6mn^2 + 20n^3
    """
    assert len(shape) == 2
    m = max(shape)
    n = min(shape)
    n2 = n * n
    n3 = n2 * n
    mn2 = m * n2
    golub = 14 * mn2 + 8 * n3
    rsvd = 6 * mn2 + 20 * n3
    return min(golub, rsvd), max(golub, rsvd)


def flops_mm(k, m, n):
    return 2 * k * m * n


def flops_tt_svd(shape, ranks):
    if len(shape) + 1 != len(ranks) or ranks[0] != ranks[-1] != 1:
        raise ValueError('Invalid inputs')
    flops = []
    numel = torch.tensor(shape).prod().item()
    for i in range(len(shape) - 1):
        r_l, m, r_r = ranks[i], shape[i], ranks[i+1]
        sz_l, sz_r = r_l * m, numel // (r_l * m)
        numel = sz_r * r_r
        flops.append(flops_svd((sz_l, sz_r)))
        flops.append(flops_mm(1, r_r, sz_r))
    flops_min = int(sum(a[0] if type(a) is tuple else a for a in flops))
    flops_max = int(sum(a[1] if type(a) is tuple else a for a in flops))
    return flops_min, flops_max


def method_ttoi_as_matlab(
        Y, gt=None, ret_X_hat=False, ret_gt_metrics=True, tt_rank_max=None, max_iter=3, tol=None, **kwargs
):
    """
    Follows closely https://github.com/Lili-Zheng-stat/TTOI/blob/master/TTOI.m
    Performs Tensor-Train Orthogonal Iteration (TTOI) algorithm for noisy observed order-d tensor Y.
    :param Y: noisy tensor observation
    :param ranks: TT-ranks of Y
    :param max_iter: maximum number of iterations
    :param tol: tolerance
    :return: a list of the estimated tensors in all iterations
    """
    if tt_rank_max is None:
        raise ValueError(f'tt_rank_max not specified')
    if not torch.is_tensor(Y):
        raise ValueError(f'Invalid input tensor: {type(Y)}')
    if type(max_iter) != int or tol is not None and type(tol) not in (int, float):
        raise ValueError(f'Invalid method settings: {max_iter=} {tol=}')

    ranks = get_tt_ranks(list(Y.shape), max_rank=tt_rank_max, tt_rank_equal=False)
    ranks = ranks[1: -1]

    def prod(x):
        return torch.prod(torch.tensor(x)).item()

    def svds(A, k):
        U, S, Vt = torch.linalg.svd(A, full_matrices=False)
        return U[:, :k], torch.diag(S[:k]), Vt[:k, :].T, flops_svd(A.shape)

    def mm(A, B):
        assert A.dim() == 2
        assert B.dim() == 2
        return A @ B, flops_mm(A.shape[0], A.shape[1], B.shape[1])

    def eye(k):
        return torch.eye(k)

    def kron(A, B):
        # torch.kron does not like SVD outputs as inputs:
        # https://github.com/pytorch/pytorch/issues/74442#issuecomment-1111445351
        out = (A[:, None, :, None] * B[None, :, None, :]).reshape(A.shape[0] * B.shape[0], A.shape[1] * B.shape[1])
        flops = A.shape[0] * A.shape[1] * B.shape[0] * B.shape[1]
        return out, flops

    def reshape_fortran(A, *args):
        # Fortran order reshape
        # https://stackoverflow.com/a/63964246/411907
        if len(A.shape) > 0:
            A = A.permute(*reversed(range(len(A.shape))))
        A = A.reshape(*list(reversed(args)))
        A = A.permute(*reversed(range(A.dim())))
        return A

    dim_vec = Y.shape
    d = len(dim_vec)
    X_hat_arr = [None] * max_iter
    V_prod_arr = [None] * (max_iter // 2 + 1)
    U_prod_arr = [None] * ((max_iter + 1) // 2)
    Y_arr = [None] * (d-1)

    V_prod_arr[0] = [None] * (d-1)
    for i in range(d-1):
        Y_arr[i] = reshape_fortran(Y, prod(dim_vec[:i+1]), prod(dim_vec[i+1:]))

    n = 0
    chg = float('inf')
    Y_tilde_arr = [None] * (d-2)
    X_hat = None
    flops = []
    while n < max_iter and (not tol or chg > tol):
        if n & 1 == 0:
            idx_prod = n // 2
            # U_prod_arr{n/2+1}=cell(d-1,1);
            U_prod_arr[idx_prod] = [None] * (d-1)
            # Y_tilde_arr=cell(d-2,1);
            Y_tilde_arr = [None] * (d-2)
            # [U_temp,~,~]=svds(Y_arr{1} * V_prod_arr{n/2+1}{d-1},r_vec(1));
            V_prod_temp = V_prod_arr[idx_prod][d-2]
            if V_prod_temp is not None:
                YV_prod_temp, f = mm(Y_arr[0], V_prod_temp)
                flops.append(f)
            else:
                YV_prod_temp = Y_arr[0]
            U_temp, _, _, f = svds(YV_prod_temp, ranks[0])
            flops.append(f)
            # U_prod_arr{n/2+1}{1}=U_temp;
            U_prod_arr[idx_prod][0] = U_temp
            for k in range(1, d-1):
                # Y_temp=kron( eye(dim_vec(k)) , U_prod_arr{n/2+1}{k-1} ) ' * Y_arr{k};
                Y_temp, f = kron(eye(dim_vec[k]), U_prod_arr[idx_prod][k-1])
                flops.append(f)
                Y_temp, f = mm(Y_temp.T, Y_arr[k])
                flops.append(f)
                # Y_tilde_arr{k-1}=reshape(Y_temp,r_vec(k-1),prod(dim_vec(k:d)));
                Y_tilde_arr[k-1] = reshape_fortran(Y_temp, ranks[k-1], prod(dim_vec[k:]))
                # [U_temp,~,~]=svds(Y_temp * V_prod_arr{n/2+1}{d-k},r_vec(k));
                V_prod_temp = V_prod_arr[idx_prod][d-k-2]
                if V_prod_temp is not None:
                    YV_prod_temp, f = mm(Y_temp, V_prod_temp)
                    flops.append(f)
                else:
                    YV_prod_temp = Y_temp
                U_temp, _, _, f = svds(YV_prod_temp, ranks[k])
                flops.append(f)
                # U_prod_arr{n/2+1}{k}=kron(eye(dim_vec(k)),U_prod_arr{n/2+1}{k-1})*U_temp;
                U_prod_temp, f = kron(eye(dim_vec[k]), U_prod_arr[idx_prod][k-1])
                flops.append(f)
                U_prod_arr[idx_prod][k], f = mm(U_prod_temp, U_temp)
                flops.append(f)
            # X_hat_temp=U_prod_arr{n/2+1}{d-1}'*Y_arr{d-1};
            X_hat_temp, f = mm(U_prod_arr[idx_prod][d-2].T, Y_arr[d-2])
            flops.append(f)
            # X_hat_arr{n+1}=reshape(U_prod_arr{n/2+1}{d-1}*X_hat_temp,dim_vec);
            U_prod_temp, f = mm(U_prod_arr[idx_prod][d-2], X_hat_temp)
            flops.append(f)
            X_hat = reshape_fortran(U_prod_temp, dim_vec)
            X_hat_arr[n] = X_hat
        else:
            idx_prod = (n + 1) // 2
            # V_prod_arr{(n+1)/2+1}=cell(d-1,1);
            V_prod_arr[idx_prod] = [None] * (d-1)
            # [~,~,V_temp]=svds(U_prod_arr{(n+1)/2}{d-1}'*Y_arr{d-1},r_vec(d-1));
            U_prod_temp, f = mm(U_prod_arr[idx_prod-1][d-2].T, Y_arr[d-2])
            flops.append(f)
            _, _, V_temp, f = svds(U_prod_temp, ranks[d-2])
            flops.append(f)
            # V_prod_arr{(n+1)/2+1}{1}=V_temp;
            V_prod_arr[idx_prod][0] = V_temp
            for k in range(1, d-1):
                # [~,~,V_temp]=svds(Y_tilde_arr{d-k}*kron(V_prod_arr{(n+1)/2+1}{k-1},eye(dim_vec(d-k+1))),r_vec(d-k));
                V_temp, f = kron(V_prod_arr[idx_prod][k-1], eye(dim_vec[d-k-1]))
                flops.append(f)
                V_temp, f = mm(Y_tilde_arr[d-k-2], V_temp)
                flops.append(f)
                _, _, V_temp, f = svds(V_temp, ranks[d-k-2])
                flops.append(f)
                # V_prod_arr{(n+1)/2+1}{k}=kron(V_prod_arr{(n+1)/2+1}{k-1},eye(dim_vec(d-k+1)))*V_temp;
                V_prod_temp, f = kron(V_prod_arr[idx_prod][k-1], eye(dim_vec[d-k-1]))
                flops.append(f)
                V_prod_temp, f = mm(V_prod_temp, V_temp)
                flops.append(f)
                V_prod_arr[idx_prod][k] = V_prod_temp
            # X_hat_temp=Y_arr{1}*V_prod_arr{(n+1)/2+1}{d-1};
            X_hat_temp, f = mm(Y_arr[0], V_prod_arr[idx_prod][d-2])
            flops.append(f)
            # X_hat_arr{n+1}=reshape(X_hat_temp*V_prod_arr{(n+1)/2+1}{d-1}',dim_vec);
            X_hat_temp, f = mm(X_hat_temp, V_prod_arr[idx_prod][d - 2].T)
            flops.append(f)
            X_hat = reshape_fortran(X_hat_temp, dim_vec)
            X_hat_arr[n] = X_hat
        if n > 1:
            # chg=sum(X_hat_arr{n}(:).^2)-sum(X_hat_arr{n-1}(:).^2);
            chg = (X_hat_arr[n] ** 2).sum() - (X_hat_arr[n-1] ** 2).sum()
        n += 1

    flops_min = int(sum(a[0] if type(a) is tuple else a for a in flops))
    flops_max = int(sum(a[1] if type(a) is tuple else a for a in flops))

    out = {
        'max_iter': max_iter,
        'flops_total_min': flops_min,
        'flops_total_max': flops_max,
    }
    out.update(analyze_tensor(X_hat, gt, ret_X_hat, ret_gt_metrics))
    return out


def method_tt_svd(Y, gt=None, ret_X_hat=False, ret_gt_metrics=False, tt_rank_max=None):
    if tt_rank_max is None:
        raise ValueError(f'tt_rank_max not specified')
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action='ignore',
            category=UserWarning,
        )
        X_hat = tntorch.Tensor(Y, ranks_tt=tt_rank_max)
        ranks = [1] + X_hat.ranks_tt + [1]
        X_hat = X_hat.torch()
    flops_min, flops_max = flops_tt_svd(Y.shape, ranks)
    out = {
        'flops_total_min': flops_min,
        'flops_total_max': flops_max,
    }
    out.update(analyze_tensor(X_hat, gt, ret_X_hat, ret_gt_metrics))
    return out


def method_tt_cross(
        Y, gt=None, ret_X_hat=False, ret_gt_metrics=False, tt_rank_max=None,
        max_iter=100, val_size=1000, eps=1e-6
):
    if tt_rank_max is None:
        raise ValueError(f'tt_rank_max not specified')

    def fn_sample(coords):
        return sample_intcoord_tensor(Y, coords.long().chunk(Y.dim(), dim=1))

    with warnings.catch_warnings():
        warnings.filterwarnings(
            action='ignore',
            category=UserWarning,
        )
        X_hat_tt = tntorch.cross(
            function=fn_sample,
            domain=[torch.arange(0, d) for d in Y.shape],
            function_arg='matrix',
            ranks_tt=tt_rank_max,
            eps=eps,
            max_iter=max_iter,
            val_size=val_size,
            verbose=False,
            return_info=False,
            device='cpu',
            batch=False,
            suppress_warnings=True,
            detach_evaluations=True,
        )
        X_hat = X_hat_tt.torch()
    out = {}
    out.update(analyze_tensor(X_hat, gt, ret_X_hat, ret_gt_metrics))
    return out


def analyze_tensor(X_hat, gt, ret_X_hat, ret_gt_metrics):
    diff_gt = X_hat - gt
    out = {}
    if ret_gt_metrics:
        assert gt is not None
        out['L1'] = diff_gt.abs().mean().item()
        out['L2'] = (diff_gt ** 2).mean().sqrt().item()
        if math.isnan(out['L2']) or math.isnan(out['L1']):
            out.update({'L1': float('inf'), 'L2': float('inf')})
        out['X_hat_std'] = X_hat.std().item()
    if ret_X_hat:
        out['X_hat'] = X_hat.cpu()
    return out


def analyze_qttnf(model, batch_size, max_iter, ret_flops):
    out = {}
    if not ret_flops:
        return out
    out['num_uncompressed_params'] = model.num_uncompressed_params
    out['num_compressed_params'] = model.num_compressed_params
    out['compression_factor'] = model.compression_factor
    out['flops_contraction_one'] = model.fn_contract_grid_complexity['flops']
    out['flops_contraction_total'] = out['flops_contraction_one'] * max_iter
    out['flops_sample_one'] = model.fn_sample_complexity['flops']
    out['flops_sample_batch'] = out['flops_sample_one'] * batch_size
    out['flops_sample_total'] = out['flops_sample_batch'] * max_iter
    out['mem_sample_max_one'] = model.fn_sample_complexity['size_max_intermediate']
    out['mem_sample_max_batch'] = out['mem_sample_max_one'] * batch_size
    out['mem_sample_all_one'] = model.fn_sample_complexity['size_all_intermediate']
    out['mem_sample_all_batch'] = out['mem_sample_all_one'] * batch_size
    out['sigma_cores_model_init'] = model.sigma_cores
    out['sigma_cores_model_final'] = [
        c.std().item() for i, c in enumerate(model.get_cores()) if model.tt_core_isparam[i]
    ]
    return out


def select_full(model, Y, **kwargs):
    X_hat = model.contract()
    return X_hat, Y


def select_samples(model, Y, dim_vec=None, batch_size=None):
    coords = []
    for i in range(len(dim_vec)):
        coords.append(torch.randint(0, dim_vec[i], (batch_size,), device=Y.device))
    samples_X_hat = model(coords)
    samples_Y = sample_intcoord_tensor(Y, coords, last_core_is_payload=False, checks=False)
    return samples_X_hat, samples_Y


def method_qttnf_opt(
        Y, gt=None,
        ret_X_hat=False, ret_gt_metrics=False, ret_flops=False,
        log_invl=25, device='cuda', verbose=False,
        **qttnf_kwargs,
):
    if device != 'cpu' and not torch.cuda.is_available():
        if verbose:
            warnings.warn('CUDA unavailable, falling back to CPU')
        device = 'cpu'

    dim_vec = list(Y.shape)

    model = TTNF(
        dim_vec,
        qttnf_kwargs['tt_rank_max'],
        tt_rank_equal=False,
        tt_minimal_dof=(qttnf_kwargs['version_sample_qtt'] == 3),
        init_method='normal',
        expected_sample_batch_size=qttnf_kwargs['batch_size'],
        version_sample_qtt=qttnf_kwargs['version_sample_qtt'],
        dtype=Y.dtype,
        checks=False,
        verbose=False
    )

    sigma_cores_model_ttsvd = None
    if qttnf_kwargs['init_decomp']:
        model.init_with_decomposition(Y)
        sigma_cores_model_ttsvd = [
            c.std().item() for i, c in enumerate(model.get_cores()) if model.tt_core_isparam[i]
        ]
    model = model.to(device)
    Y = Y.to(device)
    if gt is not None:
        gt = gt.to(device)

    fn_loss = {
        'L1': torch.nn.L1Loss(),
        'L2': torch.nn.MSELoss(),
        'Huber': torch.nn.HuberLoss(),
    }[qttnf_kwargs['fn_loss']]

    cls_opt = {
        'sgd': torch.optim.SGD,
        'adam': torch.optim.Adam,
        'radam': torch.optim.RAdam,
    }[qttnf_kwargs['cls_opt']]

    optimizer = cls_opt(model.parameters(), lr=qttnf_kwargs['lr'], **qttnf_kwargs['opt_kwargs'])

    fn_loss_args = {
        'samples': select_samples,
        'full': select_full,
    }[qttnf_kwargs['opt_mode']]

    lr_warmup_steps = qttnf_kwargs['lr_warmup_steps']
    if lr_warmup_steps == 'auto':
        lr_warmup_steps = qttnf_kwargs['max_iter'] // 20

    loop_obj = tqdm(range(qttnf_kwargs['max_iter']), disable=not verbose)
    if verbose:
        print(qttnf_kwargs)
    for global_step in loop_obj:
        optimizer.zero_grad()
        arg_X, arg_Y = fn_loss_args(model, Y, dim_vec=dim_vec, batch_size=qttnf_kwargs['batch_size'])
        loss = fn_loss(arg_X, arg_Y)
        loss.backward()
        optimizer.step()

        decay_rate = 0.1
        new_lrate = qttnf_kwargs['lr'] * (decay_rate ** (
                qttnf_kwargs['lr_decay_pow10'] * global_step / qttnf_kwargs['max_iter']))
        if lr_warmup_steps > 0 and global_step < lr_warmup_steps:
            new_lrate = new_lrate * (global_step + 1) / lr_warmup_steps
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        if verbose and gt is not None and (global_step % log_invl == 0 or global_step == qttnf_kwargs['max_iter'] - 1):
            tmp = analyze_tensor(model.contract(), gt, ret_X_hat=False, ret_gt_metrics=True)
            loop_obj.set_postfix(tmp)

    out = {}
    out.update(qttnf_kwargs)
    out.update(analyze_tensor(model.contract(), gt, ret_X_hat, ret_gt_metrics))
    out.update(analyze_qttnf(model, qttnf_kwargs['batch_size'], qttnf_kwargs['max_iter'], ret_flops))
    out['sigma_cores_model_ttsvd'] = sigma_cores_model_ttsvd
    return out


def method_qttnf_opt_grid(
        grid_lr, grid_max_iter, cmp_metric,
        Y, gt=None, ret_X_hat=False, ret_gt_metrics=False, ret_flops=False, **kwargs
):
    best_out, best_kwargs = None, None
    for max_iter in grid_max_iter:
        for lr in grid_lr:
            kwargs_copy = dict(**kwargs)
            kwargs_copy['lr'] = lr
            kwargs_copy['max_iter'] = max_iter
            out = method_qttnf_opt(Y, gt=gt, ret_X_hat=False, ret_gt_metrics=True, ret_flops=True, **kwargs_copy)
            assert cmp_metric in out
            if best_out is None:
                best_out = out
                best_kwargs = kwargs_copy
            elif best_out[cmp_metric] > out[cmp_metric]:
                best_out = out
                best_kwargs = kwargs_copy
    if ret_X_hat:
        best_out = method_qttnf_opt(
            Y, gt=gt, ret_X_hat=ret_X_hat, ret_gt_metrics=ret_gt_metrics, ret_flops=ret_flops, **best_kwargs
        )
    return best_out


def make_ttlowrank(modes, tt_rank_max, dtype=torch.float32):
    ranks = get_tt_ranks(modes, max_rank=tt_rank_max, tt_rank_equal=False)
    sigma_cores = (-torch.tensor(ranks).double().log().sum() / (2. * len(modes))).exp().item()
    cores, shapes = [], []
    for i in range(len(modes)):
        shape = ranks[i], modes[i], ranks[i+1]
        cores.append(torch.randn(shape, dtype=dtype) * sigma_cores)
        shapes.append(shape)
    fn = compile_tt_contraction_fn(shapes)
    out = fn(*cores)
    return out, sigma_cores


def experiment(
        dim_mode, num_dims, dim_rank, seed, noise_type, noise_sigma,
        num_repeats, alg_type, dtype=torch.float, **alg_kwargs
):
    distribution = None
    if noise_sigma > 0:
        distribution = {
            'normal': torch.distributions.normal.Normal(0, noise_sigma),
            'laplace': torch.distributions.laplace.Laplace(0, noise_sigma),
        }[noise_type]

    outs = []
    failed_oom = False
    for i in range(num_repeats):
        torch.random.manual_seed(seed + i)

        try:
            modes = [dim_mode] * num_dims
            X, sigma_cores_gt = make_ttlowrank(modes, dim_rank, dtype=dtype)
            if noise_sigma > 0:
                Z = distribution.sample(modes).to(dtype)
                Y = X + Z
            else:
                Y = X.clone()

            if alg_type == 'tt_svd':
                out = method_tt_svd(Y, gt=X, ret_X_hat=False, ret_gt_metrics=True, tt_rank_max=dim_rank)
            elif alg_type == 'tt_cross':
                out = method_tt_cross(Y, gt=X, ret_X_hat=False, ret_gt_metrics=True, tt_rank_max=dim_rank, **alg_kwargs)
            elif alg_type == 'ttoi':
                out = method_ttoi_as_matlab(Y, gt=X, tt_rank_max=dim_rank, **alg_kwargs)
            elif alg_type == 'qttnf_opt':
                out = method_qttnf_opt(
                    Y, gt=X, ret_X_hat=False, ret_gt_metrics=True, ret_flops=True, tt_rank_max=dim_rank, **alg_kwargs
                )
            elif alg_type == 'qttnf_opt_grid':
                grid_kwargs = alg_kwargs.pop('grid_kwargs')
                out = method_qttnf_opt_grid(
                    grid_kwargs['grid_lr'], grid_kwargs['grid_max_iter'], grid_kwargs['cmp_metric'],
                    Y, gt=X, ret_X_hat=False, ret_gt_metrics=True, ret_flops=True, tt_rank_max=dim_rank, **alg_kwargs
                )
                # pick best found and proceed the remaining runs without grid search
                for k, v in out.items():
                    if k in alg_kwargs:
                        alg_kwargs[k] = v
                alg_type = 'qttnf_opt'
            else:
                raise ValueError(f'Invalid {alg_type=}')
            out['X_std'] = X.std().item()
            out['Y_std'] = Y.std().item()
            out['Z_std'] = Z.std().item() if noise_sigma > 0 else 0
            out['sigma_cores_gt'] = sigma_cores_gt
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                if alg_kwargs.get('verbose', False):
                    print(f'OOM in {alg_type} {alg_kwargs}')
                failed_oom = True
                break
            raise e

        outs.append(out)

    out = {}
    if failed_oom:
        out = {
            'status': 'FAILED_OOM'
        }
        out.update(**alg_kwargs)
    else:
        arr_L1 = torch.tensor([a['L1'] for a in outs])
        arr_L2 = torch.tensor([a['L2'] for a in outs])
        arr_X_std = [a['X_std'] for a in outs]
        arr_Y_std = [a['Y_std'] for a in outs]
        arr_Z_std = [a['Z_std'] for a in outs]
        arr_X_hat_std = [a['X_hat_std'] for a in outs]

        out.update(outs[0])
        del out['L1'], out['L2'], out['X_std'], out['Y_std'], out['Z_std']

        out.update({
            'status': 'OK',
            'L1_mean': arr_L1.mean().item(),
            'L1_std': arr_L1.std().item() if num_repeats > 1 else 0,
            'L2_mean': arr_L2.mean().item(),
            'L2_std': arr_L2.std().item() if num_repeats > 1 else 0,
            'X_std': arr_X_std,
            'Y_std': arr_Y_std,
            'Z_std': arr_Z_std,
            'X_hat_std': arr_X_hat_std,
        })
    out.update({
        'dim_mode': dim_mode,
        'num_dims': num_dims,
        'dim_rank': dim_rank,
        'seed': seed,
        'noise_type': noise_type,
        'noise_sigma': noise_sigma,
        'num_repeats': num_repeats,
        'alg_type': alg_type,
        'dtype': str(dtype),
    })

    if alg_kwargs.get('verbose', False):
        print(out)

    return out


def benchmark(
        dim_mode, num_dims, dim_rank, noise_type, noise_sigma, opt,
        grid_max_iter, grid_batch_size, grid_lr, grid_sample_versions,
        do_ttcross=False, do_baselines=True, out_file_path=None, num_repeats=30, seed=0
):
    out = []
    if do_baselines:
        out.append(experiment(
            dim_mode, num_dims, dim_rank, seed, noise_type=noise_type, noise_sigma=noise_sigma,
            num_repeats=num_repeats, alg_type='tt_svd', dtype=torch.float,
        ))
        out.append(experiment(
            dim_mode, num_dims, dim_rank, seed, noise_type=noise_type, noise_sigma=noise_sigma,
            num_repeats=num_repeats, alg_type='ttoi', dtype=torch.float,
            max_iter=5,
        ))
    if do_ttcross:
        out.append(experiment(
            dim_mode, num_dims, dim_rank, seed, noise_type=noise_type, noise_sigma=noise_sigma,
            num_repeats=num_repeats, alg_type='tt_cross', dtype=torch.float,
            max_iter=100,
        ))
    fn_loss = 'L1' if noise_type == 'laplace' else 'L2'
    for cur_max_iter in grid_max_iter:
        out.append(experiment(
            dim_mode, num_dims, dim_rank, seed, noise_type=noise_type, noise_sigma=noise_sigma,
            num_repeats=num_repeats, alg_type='qttnf_opt_grid', dtype=torch.float,
            max_iter=None, opt_mode='full', version_sample_qtt=3, **opt,
            lr=None, lr_decay_pow10=2, lr_warmup_steps='auto', init_decomp=True, fn_loss=fn_loss,
            device='cuda', batch_size=1, log_invl=50, verbose=True,
            grid_kwargs={
                'grid_lr': grid_lr,
                'grid_max_iter': [cur_max_iter],
                'cmp_metric': fn_loss,
            }
        ))
        for cur_batch_size in grid_batch_size:
            for version_sample_qtt in grid_sample_versions:
                out.append(experiment(
                    dim_mode, num_dims, dim_rank, seed, noise_type=noise_type, noise_sigma=noise_sigma,
                    num_repeats=num_repeats, alg_type='qttnf_opt_grid', dtype=torch.float,
                    max_iter=None, opt_mode='samples', version_sample_qtt=version_sample_qtt, **opt,
                    lr=None, lr_decay_pow10=2, lr_warmup_steps='auto', init_decomp=True, fn_loss=fn_loss,
                    device='cuda', batch_size=cur_batch_size, log_invl=50, verbose=True,
                    grid_kwargs={
                        'grid_lr': grid_lr,
                        'grid_max_iter': [cur_max_iter],
                        'cmp_metric': fn_loss,
                    }
                ))
    if out_file_path is not None:
        with open(out_file_path, 'w') as fp:
            json.dump(out, fp, indent=4)
    return out


def compare_characteristics(dim_mode, num_dims, batch_size, out_file_path=None, auto_ranks=True):
    dim_vec = [dim_mode] * num_dims
    rmax = max(get_tt_ranks(dim_vec))
    out = {
        'dim_mode': dim_mode,
        'num_dims': num_dims,
        'batch_size': batch_size,
        'num_el_full': torch.tensor(dim_vec).prod().item(),
        'rmax': rmax,
        'ranks': [],
        'dof_theoretical': [],
        'num_param_v1v2': [],
        'num_param_v3': [],
        'flops_contract': [],
        'flops_sample_one_v1v2': [],
        'flops_sample_one_v3': [],
        'flops_sample_batch_v1v2': [],
        'flops_sample_batch_v3': [],
        'mem_contraction_train': [],
        'mem_contraction_inference': [],
        'mem_sample_batch_train_v1': [],
        'mem_sample_batch_train_v2': [],
        'mem_sample_batch_train_v3': [],
        'mem_sample_batch_inference_v1': [],
        'mem_sample_batch_inference_v2': [],
        'mem_sample_batch_inference_v3': [],
    }
    if auto_ranks:
        ranks_left = [1] + torch.cumprod(torch.tensor(dim_vec), dim=0).tolist()
        ranks_set = []
        for i, r in enumerate(ranks_left):
            if r > rmax:
                break
            ranks_set.append(r)
            if i > 0:
                if r > 1:
                    ranks_set.append(r-1)
            if r < rmax:
                ranks_set.append(r+1)
        ranks_set = sorted(list(set(ranks_set)))
    else:
        ranks_set = range(1, rmax+1)
    for rank in tqdm(ranks_set):
        models = {
            1: TTNF(dim_vec, rank, version_sample_qtt=1, tt_minimal_dof=False,
                    expected_sample_batch_size=batch_size, checks=False, verbose=False),
            2: TTNF(dim_vec, rank, version_sample_qtt=2, tt_minimal_dof=False,
                    expected_sample_batch_size=batch_size, checks=False, verbose=False),
            3: TTNF(dim_vec, rank, version_sample_qtt=3, tt_minimal_dof=True,
                    expected_sample_batch_size=batch_size, checks=False, verbose=False),
        }
        out['ranks'].append(rank)
        out['num_param_v1v2'].append(models[2].num_compressed_params)
        out['num_param_v3'].append(models[3].num_compressed_params)
        out['flops_contract'].append(models[1].fn_contract_grid_complexity['flops'])
        out['flops_sample_one_v1v2'].append(models[2].fn_sample_complexity['flops'])
        out['flops_sample_one_v3'].append(models[3].fn_sample_complexity['flops'])
        out['flops_sample_batch_v1v2'].append(models[2].fn_sample_complexity['flops'] * batch_size)
        out['flops_sample_batch_v3'].append(models[3].fn_sample_complexity['flops'] * batch_size)
        out['mem_contraction_train'].append(models[1].fn_contract_grid_complexity['size_all_intermediate'])
        out['mem_contraction_inference'].append(models[1].fn_contract_grid_complexity['size_max_intermediate'])
        out['mem_sample_batch_train_v1'].append(models[1].fn_sample_complexity['size_all_intermediate'] * batch_size)
        out['mem_sample_batch_train_v2'].append(models[2].fn_sample_complexity['size_all_intermediate'] * batch_size)
        out['mem_sample_batch_train_v3'].append(models[3].fn_sample_complexity['size_all_intermediate'] * batch_size)
        out['mem_sample_batch_inference_v1'].append(models[1].fn_sample_complexity['size_max_intermediate'] * batch_size)
        out['mem_sample_batch_inference_v2'].append(models[2].fn_sample_complexity['size_max_intermediate'] * batch_size)
        out['mem_sample_batch_inference_v3'].append(models[3].fn_sample_complexity['size_max_intermediate'] * batch_size)
    for rank in range(1, rmax+1):
        ranks_tt = get_tt_ranks(dim_vec, max_rank=rank)
        dof = sum([ranks_tt[i] * dim_vec[i] * ranks_tt[i+1] - ranks_tt[i+1] ** 2 for i in range(num_dims)]) + 1
        out['dof_theoretical'].append(dof)
    if out_file_path is not None:
        with open(out_file_path, 'w') as fp:
            json.dump(out, fp, indent=4)
    return out


def dump_characteristics():
    for batch_size in (64, 256, 1024, 1024 * 4, 1024 * 16, 1024 * 64):
        compare_characteristics(2, 20, batch_size, f'../characteristics_m2_d20_bs{batch_size:05d}.json')
        compare_characteristics(2, 30, batch_size, f'../characteristics_m2_d30_bs{batch_size:05d}.json')


def dump_sigma_sweep():
    opt = {
        'cls_opt': 'adam',
        'opt_kwargs': {'betas': (0.9, 0.999)},
    }
    postfix = 'sigmasweep_final'

    grid_max_iter=(1000,)
    grid_batch_size=(1024*4,)
    grid_lr=(3e-2,)
    grid_sample_versions = (2, 3)
    grid_ranks = (16, 128,)
    grid_noise = (0, 0.1, 0.3, 1.0, 3.0, 10.0)
    num_repeats = 10

    for noise_type in ('normal', 'laplace'):
        for rank in grid_ranks:
            for noise_sigma in grid_noise:
                benchmark(
                    dim_mode=2,
                    num_dims=20,
                    dim_rank=rank,
                    noise_type=noise_type, noise_sigma=noise_sigma, opt=opt, grid_max_iter=grid_max_iter,
                    grid_batch_size=grid_batch_size, grid_lr=grid_lr, grid_sample_versions=grid_sample_versions,
                    do_ttcross=True,
                    out_file_path=f'../benchmark_m2_d20_r{rank}_{noise_type}_s{noise_sigma}_lr{grid_lr[0]}_nr{num_repeats}_{postfix}.json',
                    num_repeats=num_repeats,
                    seed=0
                )


if __name__ == '__main__':
    dump_characteristics()
    dump_sigma_sweep()
