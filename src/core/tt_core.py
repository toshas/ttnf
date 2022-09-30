import warnings
from functools import partial

import opt_einsum
import torch
import torch.nn.functional as F


def get_qtt3d_reshape_plan(dim_grid_log2, dim_payload, payload_last=True):
    dim_grid = 2 ** dim_grid_log2
    num_factors = dim_grid_log2 * 3

    shape_src = [dim_grid] * 3
    shape_dst = [8] * dim_grid_log2
    shape_factors = [2] * num_factors

    factor_ids = torch.arange(num_factors)
    permute_factors_src_to_dst = factor_ids.reshape(3, dim_grid_log2).T.reshape(-1).tolist()
    permute_factors_dst_to_src = factor_ids.reshape(dim_grid_log2, 3).T.reshape(-1).tolist()

    if payload_last:
        shape_src.append(dim_payload)
        shape_dst.append(dim_payload)
        shape_factors.append(dim_payload)
        permute_factors_src_to_dst.append(num_factors)
        permute_factors_dst_to_src.append(num_factors)
    else:
        shape_src.insert(0, dim_payload)
        shape_dst.insert(0, dim_payload)
        shape_factors.insert(0, dim_payload)
        permute_factors_src_to_dst = [c + 1 for c in permute_factors_src_to_dst]
        permute_factors_src_to_dst.insert(0, 0)
        permute_factors_dst_to_src = [c + 1 for c in permute_factors_dst_to_src]
        permute_factors_dst_to_src.insert(0, 0)

    return {
        'shape_factors': shape_factors,
        'shape_src': shape_src,
        'shape_dst': shape_dst,
        'permute_factors_src_to_dst': permute_factors_src_to_dst,
        'permute_factors_dst_to_src': permute_factors_dst_to_src,
    }


def tensor_order_to_qtt(x, plan):
    x = x.reshape(plan['shape_factors'])
    x = x.permute(plan['permute_factors_src_to_dst'])
    x = x.reshape(plan['shape_dst'])
    return x


def tensor_order_from_qtt(x, plan):
    x = x.reshape(plan['shape_factors'])
    x = x.permute(plan['permute_factors_dst_to_src'])
    x = x.reshape(plan['shape_src'])
    return x


def get_tt_ranks(shape, max_rank=None, tt_rank_equal=False):
    if type(shape) not in (tuple, list) or len(shape) == 0:
        raise ValueError(f'Invalid shape: {shape}')
    if len(shape) == 1:
        return [1, 1]
    if tt_rank_equal:
        if max_rank is None:
            raise ValueError('max_rank must be specified when tt_rank_equal is set')
        return [1] + [max_rank] * (len(shape) - 1) + [1]
    ranks_left = [1] + torch.cumprod(torch.tensor(shape), dim=0).tolist()
    ranks_right = list(reversed([1] + torch.cumprod(torch.tensor(list(reversed(shape))), dim=0).tolist()))
    ranks_tt = [min(a, b) for a, b in zip(ranks_left, ranks_right)]
    if max_rank is not None:
        ranks_tt = [min(r, max_rank) for r in ranks_tt]
    return ranks_tt


def gen_letter():
    next_letter_id = 0
    while True:
        yield opt_einsum.get_symbol(next_letter_id)
        next_letter_id += 1


def shapes(input):
    return [c.shape for c in input]


def is_tt_shapes(
        input_shapes,
        inputs_with_batch_dim=None,
        batch_size=None,
        allow_loose_rank_left=False,
        allow_loose_rank_right=False,
):
    if type(input_shapes) not in (tuple, list) or \
            any(len(s) != 3 for s in input_shapes) or \
            any(input_shapes[i-1][-1] != input_shapes[i][0] for i in range(1, len(input_shapes))):
        return False
    if not (allow_loose_rank_left or input_shapes[0][0] == 1):
        return False
    if not (allow_loose_rank_right or input_shapes[-1][-1] == 1):
        return False
    if inputs_with_batch_dim is not None and not (
            type(batch_size) is int and
            batch_size > 0 and
            type(inputs_with_batch_dim) in (tuple, list) and
            len(inputs_with_batch_dim) == len(input_shapes) and
            all([type(b) is bool for b in inputs_with_batch_dim])
    ):
        return False
    return True


def is_list_of_tensors(input):
    return type(input) in (list, tuple) and all(torch.is_tensor(c) for c in input)


def is_tt(input):
    return is_list_of_tensors(input) and is_tt_shapes(shapes(input))


def perf_report(equation, *shapes, einsum_opt_method='dp'):
    _, pathinfo = opt_einsum.contract_path(equation, *shapes, shapes=True, optimize=einsum_opt_method)
    out = {
        'flops': int(pathinfo.opt_cost),
        'size_max_intermediate': int(pathinfo.largest_intermediate),
        'size_all_intermediate': int(sum(pathinfo.size_list)),
        'equation': equation,
        'input_shapes': shapes,
    }
    return out


def tt_svd(A, ranks):
    warnings.warn('Use tntorch instead for a better approximation')
    if not torch.is_tensor(A):
        raise ValueError('Operand is not a tensor')
    if type(ranks) in (list, tuple):
        tt_ranks_max = get_tt_ranks(list(A.shape))
        if A.dim() + 1 != len(ranks) or ranks[0] != ranks[-1] != 1 or \
                any(a > b for a, b in zip(ranks, tt_ranks_max)):
            raise ValueError('Invalid ranks')
    elif type(ranks) is int:
        ranks = get_tt_ranks(A.shape, max_rank=ranks)
    else:
        raise ValueError('Invalid ranks type')
    shape = A.shape
    cores = []
    C = A
    for i in range(len(shape) - 1):
        r_l, m, r_r = ranks[i], shape[i], ranks[i+1]
        unfolding = C.reshape(r_l * m, -1)
        U, S, Vt = torch.linalg.svd(unfolding, full_matrices=False)
        U, S, Vt = U[:, :r_r], S[:r_r], Vt[:r_r, :]
        cores.append(U.reshape(r_l, m, r_r))
        C = S.view(-1, 1) * Vt
    cores.append(C.reshape(ranks[-2], shape[-1], ranks[-1]))
    return cores


def compile_tt_contraction_fn(
        input_shapes,
        inputs_with_batch_dim=None,
        batch_size=None,
        allow_loose_rank_left=False,
        allow_loose_rank_right=False,
        last_core_is_payload=False,
        output_modes_squeeze=False,
        output_last_rank_keep=False,
        einsum_opt_method='dp',
        report_flops=False
):
    if not is_tt_shapes(input_shapes, inputs_with_batch_dim, batch_size, allow_loose_rank_left, allow_loose_rank_right):
        raise ValueError(f'Operand shapes do not form a tensor train: {input_shapes=} {inputs_with_batch_dim=} '
                         f'{batch_size=} {allow_loose_rank_left=} {allow_loose_rank_right=}')

    have_batch_dim = inputs_with_batch_dim is not None and any(inputs_with_batch_dim)
    letter_batch = None
    letter = gen_letter()
    if have_batch_dim:
        letter_batch = next(letter)

    equation_left = ''
    equation_right = letter_batch if have_batch_dim else ''

    letter_core_last_rank_right = None
    input_shapes_with_batch_dim = []

    for i in range(len(input_shapes)):
        if inputs_with_batch_dim is not None and inputs_with_batch_dim[i]:
            input_shapes_with_batch_dim.append([batch_size] + list(input_shapes[i]))
        else:
            input_shapes_with_batch_dim.append(list(input_shapes[i]))

        letter_rank_left = next(letter) if i == 0 else letter_core_last_rank_right
        letters_modes = [next(letter) for _ in range(len(input_shapes[i]) - 2)]
        letter_rank_right = next(letter)
        letter_core_last_rank_right = letter_rank_right
        if i > 0:
            equation_left += ','
        if inputs_with_batch_dim is not None and inputs_with_batch_dim[i]:
            equation_left += letter_batch
        equation_left += letter_rank_left
        equation_left += ''.join(letters_modes)
        equation_left += letter_rank_right
        if i == 0 and input_shapes[i][0] > 1:
            equation_right += letter_rank_left
        if output_modes_squeeze:
            for c, si in zip(letters_modes, input_shapes[i][1:-1]):
                if si > 1 or (last_core_is_payload and i == len(input_shapes) - 1):
                    equation_right += c
        else:
            equation_right += ''.join(letters_modes)
        if i == len(input_shapes) - 1 and (output_last_rank_keep or input_shapes[i][-1] > 1):
            equation_right += letter_rank_right

    equation = equation_left + '->' + equation_right
    contraction_fn = opt_einsum.contract_expression(equation, *input_shapes_with_batch_dim, optimize=einsum_opt_method)

    if report_flops:
        report = perf_report(equation, *input_shapes_with_batch_dim, einsum_opt_method=einsum_opt_method)
        return contraction_fn, report

    return contraction_fn


def compile_tt_simplification_fn(input_shapes, report_flops=False):
    """
    Merges cores with mode=1 into neighboring cores
    :param input_shapes: Shapes of TT cores
    :return: Simplification contraction function
    """
    if not is_tt_shapes(input_shapes):
        raise ValueError(f'Operand shapes do not form a tensor train: {input_shapes}')
    is_matrix = [s[1] == 1 and i != len(input_shapes) - 1 for i, s in enumerate(input_shapes)]
    isolated_cores_left = is_matrix.index(True)
    isolated_cores_right = list(reversed(is_matrix)).index(True)
    if not all(is_matrix[isolated_cores_left: (-isolated_cores_right or None)]):
        raise ValueError(
            f'The chunk with mode=1 cores needs to be continuous in this implementation, got {input_shapes}'
        )
    if isolated_cores_left == isolated_cores_right == 0:
        raise ValueError(f'Shapes not subject to simplification, got {input_shapes}')

    out_fn, out_rep, out_is_left, out_shapes, out_cost = None, None, None, None, float('inf')
    if isolated_cores_left > 0:
        # merging into left
        fn_left, rep_left = compile_tt_contraction_fn(
            input_shapes[isolated_cores_left-1: (-isolated_cores_right or None)],
            allow_loose_rank_left=(isolated_cores_left > 1),
            allow_loose_rank_right=(isolated_cores_right > 0),
            report_flops=True
        )
        simplified_core_left_r_l = input_shapes[isolated_cores_left-1][0]
        simplified_core_left_modes = list(input_shapes[isolated_cores_left-1][1:-1])
        simplified_core_left_r_r = input_shapes[-isolated_cores_right-1][-1]
        simplified_core_left_shape = \
            [simplified_core_left_r_l] + simplified_core_left_modes + [simplified_core_left_r_r]
        out_shapes_left = input_shapes[:isolated_cores_left-1] + [simplified_core_left_shape]
        if isolated_cores_right > 0:
            out_shapes_left += input_shapes[-isolated_cores_right:]
        cost = compile_tt_contraction_fn(out_shapes_left, report_flops=True)[1]['flops']

        def _fn_left(*cores):
            core_simplified = fn_left(*cores[isolated_cores_left-1: (-isolated_cores_right or None)])
            core_simplified = core_simplified.reshape(simplified_core_left_shape)
            out = list(cores[:isolated_cores_left-1]) + [core_simplified]
            if isolated_cores_right > 0:
                out += cores[-isolated_cores_right:]
            return out

        if cost < out_cost:
            out_fn, out_rep, out_is_left, out_shapes, out_cost = _fn_left, rep_left, True, out_shapes_left, cost

    if isolated_cores_right > 0:
        # merging into right
        fn_right, rep_right = compile_tt_contraction_fn(
            input_shapes[isolated_cores_left: (-isolated_cores_right+1 or None)],
            allow_loose_rank_left=(isolated_cores_left > 0),
            allow_loose_rank_right=(isolated_cores_right > 1),
            report_flops=True
        )
        simplified_core_right_r_l = input_shapes[isolated_cores_left][0]
        simplified_core_right_modes = list(input_shapes[-isolated_cores_right][1:-1])
        simplified_core_right_r_r = input_shapes[-isolated_cores_right][-1]
        simplified_core_right_shape = \
            [simplified_core_right_r_l] + simplified_core_right_modes + [simplified_core_right_r_r]
        out_shapes_right = input_shapes[:isolated_cores_left] + [simplified_core_right_shape]
        if isolated_cores_right > 1:
            out_shapes_right += input_shapes[-isolated_cores_right+1:]
        cost = compile_tt_contraction_fn(out_shapes_right, report_flops=True)[1]['flops']

        def _fn_right(*cores):
            core_simplified = fn_right(*cores[isolated_cores_left: (-isolated_cores_right+1 or None)])
            core_simplified = core_simplified.reshape(simplified_core_right_shape)
            out = list(cores[:isolated_cores_left]) + [core_simplified]
            if isolated_cores_right > 1:
                out += cores[-isolated_cores_right+1:]
            return out

        if cost < out_cost:
            out_fn, out_rep, out_is_left, out_shapes, out_cost = _fn_right, rep_right, False, out_shapes_right, cost

    if report_flops:
        out_rep['simplification_direction'] = 'left' if out_is_left else 'right'
        out_rep['post_simplification_contraction_cost'] = out_cost
        out_rep['out_shapes'] = out_shapes
        return out_fn, out_rep
    return out_fn


def convert_qtt_to_tensor(input, qtt_reshape_plan=None, fn_contract=None, checks=False):
    if checks and not is_tt(input):
        raise ValueError('Operand is not a tensor train')
    if fn_contract is None:
        input_shapes = shapes(input)
        fn_contract = compile_tt_contraction_fn(input_shapes)
    out = fn_contract(*input)
    if qtt_reshape_plan is not None:
        out = tensor_order_from_qtt(out, qtt_reshape_plan)
    return out


def convert_tensor_to_qtt(input, qtt_reshape_plan, rmax):
    if not torch.is_tensor(input):
        raise ValueError(f'First operand is not a tensor')
    if type(rmax) is not int or rmax < 1:
        raise ValueError(f'Invalid rank constraint')
    try:
        import tntorch
    except ImportError:
        raise ImportError('tntorch is required only in this one function')
    out = tensor_order_to_qtt(input, qtt_reshape_plan)
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', category=UserWarning)
        out = tntorch.Tensor(out, ranks_tt=rmax)
    return out.cores


def coord_tensor_to_coord_qtt3d(coords_xyz, dim_grid_log2, chunk=False, checks=False):
    if checks:
        if not torch.is_tensor(coords_xyz) or coords_xyz.dim() != 2 or coords_xyz.shape[1] != 3:
            raise ValueError('Coordinates is not an Nx3 tensor')
        if not coords_xyz.dtype in (torch.int, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError('Coordinates are not integer')
        if torch.any(coords_xyz < 0) or torch.any(coords_xyz > 2 ** dim_grid_log2 - 1):
            raise ValueError('Coordinates out of bounds')
    bit_factors = 2 ** torch.arange(
        start=dim_grid_log2-1,
        end=-1,
        step=-1,
        device=coords_xyz.device,
        dtype=coords_xyz.dtype
    )
    bits_xyz = coords_xyz.unsqueeze(-1).bitwise_and(bit_factors).ne(0).byte()  # N x 3 x dim_grid_log2
    bits_xyz = bits_xyz * torch.tensor([[[4], [2], [1]]], device=coords_xyz.device, dtype=torch.uint8)  # qtt octets
    core_indices = bits_xyz.sum(dim=1, dtype=torch.uint8)  # N x dim_grid_log2
    if chunk:
        core_indices = core_indices.chunk(dim_grid_log2, dim=1)  # [core_0_ind, ..., core_last_ind]
        core_indices = [c.view(-1) for c in core_indices]
    return core_indices


def coord_qtt3d_to_coord_tensor(coords_qtt, checks=False):
    if checks:
        if coords_qtt.dim() != 2:
            raise ValueError('Coordinates are not a matrix')
        if not coords_qtt.dtype in (torch.int, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError('Coordinates are not integer')
        if torch.any(coords_qtt < 0) or torch.any(coords_qtt > 7):
            raise ValueError('Coordinates out of bounds')

    dim_grid_log2 = coords_qtt.shape[1]
    bit_factors = 2 ** torch.arange(
        start=max(dim_grid_log2, 3) - 1,
        end=-1,
        step=-1,
        device=coords_qtt.device,
        dtype=coords_qtt.dtype
    )

    bits_qtt = coords_qtt.unsqueeze(-1).bitwise_and(bit_factors[-3:]).ne(0).byte()  # N x dim_grid_log2 x 3
    coords_xyz_bits = bits_qtt.permute([0, 2, 1]) * bit_factors[-dim_grid_log2:]  # N x 3 x dim_grid_log2
    return coords_xyz_bits.sum(dim=-1)


def sample_intcoord_tt_v1__compile_tt_contraction_fn(
        input_shapes, batch_size, last_core_is_payload=False, report_flops=False
):
    if last_core_is_payload:
        out = compile_tt_contraction_fn(
            [[s[0], 1, s[2]] for s in input_shapes[:-1]] + [list(input_shapes[-1])],
            inputs_with_batch_dim=[True] * (len(input_shapes) - 1) + [False],  # all cores except payload are indexed
            batch_size=batch_size, last_core_is_payload=last_core_is_payload,
            output_modes_squeeze=True, output_last_rank_keep=False, einsum_opt_method='dp',
            report_flops=report_flops
        )
    else:
        out = compile_tt_contraction_fn(
            [[s[0], 1, s[2]] for s in input_shapes],
            inputs_with_batch_dim=[True] * len(input_shapes),  # all cores are indexed
            batch_size=batch_size, last_core_is_payload=last_core_is_payload,
            output_modes_squeeze=True, output_last_rank_keep=True, einsum_opt_method='dp',
            report_flops=report_flops
        )
    if report_flops:
        out[1]['flops'] //= batch_size
        out[1]['size_max_intermediate'] //= batch_size
        out[1]['size_all_intermediate'] //= batch_size
    return out


def sample_intcoord_tt_v1(input, coords, last_core_is_payload=False, fn_contract_samples=None, checks=False):
    """
    Performs sampling of TT using integer coordinates (version 1). This version performs indexing of core
    tensors using mode coordinates to form batches of matrices (same number of batches as there are cores). The batch
    dimension corresponds to the batch dimension of input coordinates, and the matrices form a TT representation. The
    function further performs batch-matrix-matrix multiplication using the contraction function. Due to indexing of
    cores, the model is effectively copied before every contraction, which leads to OOM with large rank values.
    :param input (List[torch.Tensor]): TT cores
    :param coords (List[torch.Tensor]): Coordinates indexing TT cores
    :param fn_contract_samples: Sample batch contraction function obtained via
        sample_intcoord_tt_v1__compile_tt_contraction_fn
    :param checks: Extra checks
    :return: torch.Tensor of a shape (N, dim_payload) containing payloads at the sampled points.
    """
    if checks:
        if not is_tt(input):
            raise ValueError('Operand is not a tensor train')
        if last_core_is_payload and len(input) < 2:
            raise ValueError('Operand does not carry a payload (need at least two cores)')
        if len(coords) + int(last_core_is_payload) != len(input):
            raise ValueError('Coordinates do not cover all non-payload modes')
        if fn_contract_samples is None:
            raise ValueError('Contraction function not specified')

    if last_core_is_payload:
        cores = [c.index_select(1, i.int()).permute(1, 0, 2).unsqueeze(2) for c, i in zip(input[:-1], coords)] + [input[-1]]
    else:
        cores = [c.index_select(1, i.int()).permute(1, 0, 2).unsqueeze(2) for c, i in zip(input, coords)]
    out = fn_contract_samples(*cores)
    return out


def sample_intcoord_qtt3d_v1(input, coords_xyz, fn_contract_samples=None, checks=False):
    """
    Performs sampling of QTT voxel grid using integer coordinates (version 1). See `sample_intcoord_tt_v1`.
    :param input (List[torch.Tensor]): TT cores
    :param coords_xyz (torch.Tensor): Batch of coordinates (N, 3) in the X,Y,Z format
    :param fn_contract_samples: Sample batch contraction function obtained via
        sample_intcoord_qtt_v1__compile_tt_contraction_fn
    :param checks: Extra checks
    :return: torch.Tensor of a shape (N, dim_payload) containing payloads at the sampled points.
    """
    coords = coord_tensor_to_coord_qtt3d(coords_xyz, len(input) - 1, chunk=True, checks=checks)
    out = sample_intcoord_tt_v1(input, coords, last_core_is_payload=True, fn_contract_samples=fn_contract_samples, checks=checks)
    return out


def batched_indexed_gemv(bv, mm, m_indices, return_unordered=False, checks=False):
    if checks:
        if not (torch.is_tensor(bv) and torch.is_tensor(mm) and torch.is_tensor(m_indices)):
            raise ValueError('Operand is not a tensor')
        if not (bv.dtype == mm.dtype and m_indices.dtype is torch.uint8):
            raise ValueError(f'Incompatible dtypes: {bv.dtype=} {mm.dtype=} {m_indices.dtype=}')
        if bv.dim() != 2 or mm.dim() != 3 or m_indices.dim() != 1 or bv.shape[0] != m_indices.shape[0] \
                or bv.shape[1] != mm.shape[1]:
            raise ValueError(f'Invalid operand shapes: {bv.shape=} {mm.shape=} {m_indices.shape=}')
    m_indices_uniq_vals, m_indices_uniq_cnts = m_indices.unique(sorted=True, return_counts=True)
    if checks:
        if m_indices_uniq_vals.max() >= mm.shape[0]:
            raise ValueError('Incompatible index and matrices')
    m_indices_order_fwd = m_indices.argsort()
    m_indices_order_bwd = m_indices_order_fwd.argsort()
    bv = bv[m_indices_order_fwd]
    bv_split = bv.split(m_indices_uniq_cnts.cpu().tolist())
    bv_out = []
    for i, v_in in zip(m_indices_uniq_vals.cpu().tolist(), bv_split):
        v_out = F.linear(v_in, mm[i].T, None)
        bv_out.append(v_out)
    bv_out = torch.cat(bv_out, dim=0)
    if return_unordered:
        return bv_out, m_indices_order_fwd, m_indices_order_bwd
    bv_out = bv_out[m_indices_order_bwd]
    return bv_out


def sample_intcoord_tt_v2(input, coords, last_core_is_payload=False, checks=False):
    """
    Performs sampling of TT using integer coordinates (version 2). This version avoids model replication by
    applying applying an algorithm that (1) permutes all samples according to mode slice that will be used to propagate
    using the TT contraction formula, then (2) applies torch.linear mode times to each of the groups, and (3) keeps
    track of the permutation to either recover the initial order or return the inverse permutation.
    :param input (List[torch.Tensor]): TT cores
    :param coords (List[torch.Tensor]): Coordinates indexing TT cores
    :param last_core_is_payload (bool): When True, last core is a payload and thus needs no indexing
    :param checks: Extra checks
    :return: torch.Tensor of a shape (N, dim_payload) containing payloads at the sampled points.
    """
    if checks:
        if not is_tt(input):
            raise ValueError('Operand is not a tensor train')
        if last_core_is_payload and len(input) < 2:
            raise ValueError('Operand does not carry a payload (need at least two cores)')
        if len(coords) + int(last_core_is_payload) != len(input):
            raise ValueError('Coordinates do not cover all non-payload modes')

    core_indices_0 = coords[0].int()
    bv = input[0].squeeze(0).index_select(0, core_indices_0)
    permutation_fwd, permutation_bwd = None, None

    for ci in range(1, len(input) - int(last_core_is_payload)):
        mm = input[ci].permute(1, 0, 2)  # m x r_l x r_r
        core_ind = coords[ci]
        if permutation_fwd is not None:
            core_ind = core_ind[permutation_fwd]
        bv, p_fwd, p_bwd = batched_indexed_gemv(bv, mm, core_ind, return_unordered=True, checks=checks)
        permutation_fwd = p_fwd if permutation_fwd is None else permutation_fwd[p_fwd]
        permutation_bwd = p_bwd if permutation_bwd is None else p_bwd[permutation_bwd]

    if permutation_bwd is not None:
        bv = bv[permutation_bwd]

    if last_core_is_payload:
        mm_payload = input[-1].squeeze(-1).T
        bv = F.linear(bv, mm_payload, None)

    return bv


def sample_intcoord_qtt3d_v2(input, coords_xyz, checks=False):
    """
    Performs sampling of QTT voxel grid using integer coordinates (version 2). See `sample_intcoord_tt_v2`.
    :param input (List[torch.Tensor]): TT cores
    :param coords_xyz (torch.Tensor): Batch of coordinates (N, 3) in the X,Y,Z format
    :param checks: Extra checks
    :return: torch.Tensor of a shape (N, dim_payload) containing payloads at the sampled points.
    """
    coords = coord_tensor_to_coord_qtt3d(coords_xyz, len(input) - 1, chunk=True, checks=checks)
    bv = sample_intcoord_tt_v2(input, coords, last_core_is_payload=True, checks=checks)
    return bv


def sample_intcoord_tt_v3(input, coords, tt_core_isparam=None, last_core_is_payload=False, checks=False):
    """
    Performs sampling of TT using integer coordinates (version 3). This version borrows the same algorithm
    from version 2, but also checks whether the trailing TT cores are matricized identities (represented with buffers).
    For a group of such cores, the function saves computation by replacing matrix multiplication with indexing.
    :param input (List[torch.Tensor]): TT cores
    :param coords (List[torch.Tensor]): Coordinates indexing TT cores
    :param tt_core_isparam (List[bool]): Indicates which cores are parameters and which are identity matricizations.
    :param checks: Extra checks
    :return: torch.Tensor of a shape (N, dim_payload) containing payloads at the sampled points.
    """
    if tt_core_isparam is None or all(tt_core_isparam):
        return sample_intcoord_tt_v2(input, coords, last_core_is_payload=last_core_is_payload, checks=checks)
    if checks:
        if not is_tt(input):
            raise ValueError('Operand is not a tensor train')
        if last_core_is_payload and len(input) < 2:
            raise ValueError('Operand does not carry a payload (need at least two cores)')
        if len(input) != len(tt_core_isparam):
            raise ValueError('Incompatible operands')
        if len(coords) + int(last_core_is_payload) != len(input):
            raise ValueError('Coordinates do not cover all non-payload modes')

    bv = None
    permutation_fwd, permutation_bwd = None, None
    indices_left, indices_right = None, None

    for ci in range(0, len(input) - int(last_core_is_payload)):
        core_ind = coords[ci]
        if not tt_core_isparam[ci] and bv is None:
            # left
            if indices_left is None:
                indices_left = core_ind.long().clone()
            else:
                indices_left *= input[ci].shape[1]
                indices_left += core_ind.long()
        elif tt_core_isparam[ci]:
            # middle
            if bv is None:
                bv = input[ci][indices_left, core_ind.long(), :]
            else:
                mm = input[ci].permute(1, 0, 2)  # m x r_l x r_r
                if permutation_fwd is not None:
                    core_ind = core_ind[permutation_fwd]
                bv, p_fwd, p_bwd = batched_indexed_gemv(bv, mm, core_ind, return_unordered=True, checks=checks)
                permutation_fwd = p_fwd if permutation_fwd is None else permutation_fwd[p_fwd]
                permutation_bwd = p_bwd if permutation_bwd is None else p_bwd[permutation_bwd]
        else:
            # right
            cur_ind = core_ind.long().unsqueeze(-1) * input[ci].shape[2]
            if indices_right is None:
                indices_right = cur_ind
            else:
                indices_right += cur_ind

    if permutation_bwd is not None:
        bv = bv[permutation_bwd]

    if indices_right is not None:
        if last_core_is_payload:
            indices_right = indices_right + torch.arange(
                input[-1].shape[1], dtype=indices_right.dtype, device=indices_right.device
            ).view(1, -1)
        bv = bv.take_along_dim(indices_right, 1)

    return bv


def perf_report_sample_tt_v2(input_shapes):
    return perf_report_sample_tt_v3(
        input_shapes,
        tt_core_isparam=[True] * len(input_shapes),
    )


def perf_report_sample_tt_v3(input_shapes, tt_core_isparam):
    bv_dim = None
    intermediate_sizes = []
    flops = 0
    for si in range(0, len(input_shapes)):
        if not tt_core_isparam[si] and bv_dim is None:
            continue
        elif tt_core_isparam[si]:
            if bv_dim is not None:
                flops += 2 * bv_dim * input_shapes[si][2]
            bv_dim = input_shapes[si][2]
            intermediate_sizes.append(bv_dim)
        else:
            continue
    out = {
        'flops': flops,
        'size_max_intermediate': max(intermediate_sizes),
        'size_all_intermediate': sum(intermediate_sizes),
    }
    return out


def sample_intcoord_qtt3d_v3(input, coords_xyz, tt_core_isparam=None, checks=False):
    """
    Performs sampling of TT voxel grid using integer coordinates (version 3). See `sample_intcoord_tt_v3`.
    :param input (List[torch.Tensor]): TT cores
    :param coords_xyz (torch.Tensor): Batch of coordinates (N, 3) in the X,Y,Z format
    :param tt_core_isparam (List[bool]): Indicates which cores are parameters and which are identity matricizations.
    :param checks: Extra checks
    :return: torch.Tensor of a shape (N, dim_payload) containing payloads at the sampled points.
    """
    if tt_core_isparam is None or all(tt_core_isparam):
        return sample_intcoord_qtt3d_v2(input, coords_xyz, checks=checks)
    coords = coord_tensor_to_coord_qtt3d(coords_xyz, len(input) - 1, chunk=True, checks=checks)
    bv = sample_intcoord_tt_v3(input, coords, tt_core_isparam=tt_core_isparam, last_core_is_payload=True, checks=checks)
    return bv


def sample_intcoord_tensor(input, coords, last_core_is_payload=False, checks=False):
    """
    Performs sampling of a full tensor using integer coordinates.
    :param input (List[torch.Tensor]): TT cores
    :param coords (List[torch.Tensor]): Coordinates indexing TT cores
    :param last_core_is_payload (bool): When True, last core is a payload and thus needs no indexing
    :param checks: Extra checks
    :return: torch.Tensor of a shape (N, dim_payload) containing payloads at the sampled points.
    """
    if checks:
        if not torch.is_tensor(input):
            raise ValueError('Input is not a tensor')
        if type(coords) not in (tuple, list):
            raise ValueError('Coords is not a list')
        if last_core_is_payload and len(input) < 2:
            raise ValueError('Operand does not carry a payload (need at least two cores)')
        if len(coords) + int(last_core_is_payload) != input.dim():
            raise ValueError('Coordinates do not cover all non-payload modes')
        if not all(
                torch.is_tensor(coo) and coo.dim() == 1 and
                coo.dtype in (torch.int, torch.int8, torch.int16, torch.int32, torch.int64) and
                coo.numel() == coords[0].numel() and coo.dtype == coords[0].dtype and
                torch.all(coo >= 0) and torch.all(coo < input.shape[i])
                for i, coo in enumerate(coords)
        ):
            raise ValueError('Bad coordinates')
    if last_core_is_payload:
        input_flat = input.reshape(-1, input.shape[-1])
    else:
        input_flat = input.reshape(-1, 1)
    strides = torch.tensor(input.shape[1:len(coords)]).flip(0).cumprod(0).flip(0).tolist() + [1]
    coords_flat = torch.cat([coo.view(-1, 1) for coo in coords], dim=1)
    coords_flat = (coords_flat.long() * torch.tensor(strides).to(coords_flat.device).view(1, -1)).sum(dim=1)
    out = input_flat.index_select(0, coords_flat)
    return out


def sample_intcoord_tensor3d(input, coords_xyz, checks=False):
    coords = coords_xyz.chunk(3, dim=1)
    coords = [c.view(-1) for c in coords]
    return sample_intcoord_tensor(input, coords, last_core_is_payload=True, checks=checks)


def sample_generic_3d(
        input,
        coords_xyz,
        fn_sample_intcoord,
        sample_redundancy_handling=True,
        outliers_handling=None,
        checks=False,
):
    if type(input) in (list, tuple):
        dim_grid = 2 ** (len(input) - 1)
        dim_payload = input[-1].shape[-2]
    elif torch.is_tensor(input):
        dim_grid = input.shape[0]
        dim_payload = input.shape[-1]
    else:
        raise ValueError('Invalid input')

    if checks:
        if not torch.is_tensor(coords_xyz) or coords_xyz.dim() != 2 or coords_xyz.shape[1] != 3:
            raise ValueError('Coordinates is not an Nx3 tensor')
        if not coords_xyz.dtype in (torch.float, torch.float16, torch.float32, torch.float64):
            raise ValueError('Coordinates are not floats')

    batch_size = coords_xyz.shape[0]
    mask_valid, mask_need_remap = None, None

    if outliers_handling:
        if outliers_handling == 'raise':
            mask_bad_left = coords_xyz < 0
            if torch.any(mask_bad_left):
                mask_bad_left = mask_bad_left.any(dim=-1)
                bad_coords = coords_xyz[mask_bad_left]
                raise ValueError(f'Coordinates out of bounds < 0: {bad_coords}')
            mask_bad_right = coords_xyz > dim_grid - 1
            if torch.any(mask_bad_right):
                mask_bad_right = mask_bad_right.any(dim=-1)
                bad_coords = coords_xyz[mask_bad_right]
                raise ValueError(f'Coordinates out of bounds > {dim_grid-1}: {bad_coords}')
        elif outliers_handling == 'clamp':
            coords_xyz.clamp_(min=0, max=dim_grid-1)
        elif outliers_handling == 'zeros':
            mask_valid = torch.all(coords_xyz >= 0, dim=1) & torch.all(coords_xyz <= dim_grid - 1, dim=1)
            coords_xyz = coords_xyz[mask_valid]
            if coords_xyz.shape[0] == 0:
                return torch.zeros(batch_size, dim_payload, dtype=coords_xyz.dtype, device=coords_xyz.device)
            mask_need_remap = coords_xyz.shape[0] < batch_size
        else:
            raise ValueError(f'Unknown outliers handling: {outliers_handling}')

    offs = torch.tensor([
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ], device=coords_xyz.device)

    coo_left_bottom_near = torch.floor(coords_xyz)
    wb = coords_xyz - coo_left_bottom_near
    wa = 1.0 - wb

    coo_left_bottom_near = coo_left_bottom_near.int()

    coo_cube = [coo_left_bottom_near] + [coo_left_bottom_near + offs[i] for i in range(7)]
    coo_cube = torch.cat(coo_cube, dim=0)
    coo_cube.clamp_max_(dim_grid - 1)

    # sample without redundancy
    if sample_redundancy_handling:
        coo_unique, coo_map_back = coo_cube.unique(dim=0, sorted=False, return_inverse=True)
        val_unique = fn_sample_intcoord(input, coo_unique, checks=checks)
        val = val_unique[coo_map_back]
    else:
        val = fn_sample_intcoord(input, coo_cube, checks=checks)

    # trilinear interpolation
    num_samples = coords_xyz.shape[0]
    val = val.view(4, 2, num_samples, dim_payload)
    val = val[:, 0, :, :] * wa[None, :, 2, None] + val[:, 1, :, :] * wb[None, :, 2, None]  # 4 x B x P
    val = val.view(2, 2, num_samples, dim_payload)
    val = val[:, 0, :, :] * wa[None, :, 1, None] + val[:, 1, :, :] * wb[None, :, 1, None]  # 2 x B x P
    val = val[0] * wa[:, 0, None] + val[1] * wb[:, 0, None]  # B x P

    if outliers_handling == 'zeros' and mask_need_remap:
        out_sparse = torch.zeros(batch_size, dim_payload, dtype=coords_xyz.dtype, device=coords_xyz.device)
        out_sparse[mask_valid] = val
        return out_sparse

    return val


def sample_qtt3d(
        input,
        coords_xyz,
        fn_contract_samples=None,
        fn_contract_grid=None,
        reshape_plan=None,
        sample_by_contraction=False,
        outliers_handling=None,
        tt_core_isparam=None,
        version=3,
        checks=False,
):
    sample_redundancy_handling = True
    if sample_by_contraction:
        if reshape_plan is None:
            raise ValueError('reshape_plan is required for qtt3d')
        input = convert_qtt_to_tensor(
            input,
            qtt_reshape_plan=reshape_plan,
            fn_contract=fn_contract_grid,
            checks=checks,
        )
        fn_sample_intcoord = sample_intcoord_tensor3d
        sample_redundancy_handling = False
    elif version == 1:
        fn_sample_intcoord = partial(sample_intcoord_qtt3d_v1, fn_contract_samples=fn_contract_samples)
    elif version == 2:
        fn_sample_intcoord = sample_intcoord_qtt3d_v2
    elif version == 3:
        fn_sample_intcoord = partial(sample_intcoord_qtt3d_v3, tt_core_isparam=tt_core_isparam)
    else:
        raise ValueError(f'Invalid sampling version {version}')
    return sample_generic_3d(
        input,
        coords_xyz,
        fn_sample_intcoord,
        sample_redundancy_handling=sample_redundancy_handling,
        outliers_handling=outliers_handling,
        checks=checks
    )


def sample_tensor3d(input, coords_xyz, outliers_handling=None, checks=False):
    return sample_generic_3d(
        input,
        coords_xyz,
        sample_intcoord_tensor3d,
        sample_redundancy_handling=False,
        outliers_handling=outliers_handling,
        checks=checks,
    )
