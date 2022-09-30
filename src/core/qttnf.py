from .tt_core import *


class QTTNF(torch.nn.Module):
    def __init__(
            self,
            dim_grid,
            dim_payload,
            tt_rank_max,
            tt_rank_equal=False,
            tt_minimal_dof=False,
            init_method='normal',
            outliers_handling='raise',
            sample_by_contraction=True,
            expected_sample_batch_size=1024,
            version_sample_qtt=3,
            dtype=torch.float32,
            checks=True,
            verbose=True,
    ):
        """
        Creates an optimizable compressed voxel grid of arbitrary dimensionality.
        :param dim_grid (int): Side of the 3D-cube (number of voxels in each dimension).
        :param dim_payload (int): Number of floats to store in each voxel grid point.
        :param tt_rank_max (int): Maximum tensor train rank value; defines the compression-quality trade-off. Should not
            be less than dim_payload, and should not be larger than some very large value determined at run-time.
        :param tt_rank_equal (bool): When True, all intermediate tt ranks are set to tt_rank_max, otherwise they are
            computed from tt_rank_max and tensorized modes. Incompatible with tt_minimal_dof.
        :param tt_minimal_dof (bool): When True, skips parameterization of cores with square unfolding, explained here:
            https://arxiv.org/pdf/2103.04217.pdf . When False, parameterizes all cores, which may turn out to be better
            for optimization. Incompatible with tt_rank_equal.
        :param init_method (str):
            - normal: chooses sigma of Gaussian distribution initialization of parameterized cores such that elements of
                      the grid contraction have unit variance.
            - zeros: zero initialization of cores.
            - eye: initializes all cores[:, i, :] slices as (truncated) identity matrices for all i.
        :param outliers_handling (optional str): Defines behavior when sampling coordinates outside of grid validity:
            - None: assumes no outliers; undefined behavior when they are encountered.
            - raise: raises a ValueError when outliers are encountered.
            - zeros: returns zero vectors in positions corresponding to outlier coordinates.
            - clamp: clamps coordinates to the valid range before sampling.
        :param sample_by_contraction (bool): When True, performs contraction and samples via direct lookup of values.
        :param expected_sample_batch_size (int): Expected number of points passed to `sample` method at once. The exact
            value is not important, but the overall order of magnitude affects the efficiency of the tensor network
            contraction algorithm, and in the end, defines FLOPs of the sampling procedure.
        :param version_sample_qtt (int): Version of the qtt sampling function. Currently available versions:
            - 1: Each sample results in instantiation and contraction of a TT slice
                 [r_0 x r_1][r_1 x r_2]...[r_{N-1} x r_N]. This is O(ttrank^2) and O(number of samples).
            - 2: Memory saved by not instantiating TT slices for each sample. Instead, samples are grouped based on
                 which mode slice will be used (there are 8 options in each core), and then a torch Linear layer is
                 re-used to perform the operation. Grouping and ungrouping are implemented as permutations.
            - 3: Memory saved by additionally replacing matmul operations with indexing for non-learned cores,
                 represented with identity matricizations.
        :param dtype (torch.dtype): Floating point type to use for storage.
        :param checks (bool): Enables all sorts of checks; should be off at training time.
        :param verbose (bool): Prints verbose messages.
        """
        super().__init__()

        if type(dim_grid) is not int or dim_grid & (dim_grid - 1) != 0:
            raise ValueError('dim_grid must be a power of two integer')
        if type(dim_payload) is not int or dim_payload < 1:
            raise ValueError('dim_payload must be a positive integer')
        if type(tt_rank_max) is not int or tt_rank_max < dim_payload:
            raise ValueError('tt_rank_max must be an integer larger than dim_payload')
        if tt_rank_equal and tt_minimal_dof:
            raise ValueError('tt_rank_equal and tt_minimal_dof are incompatible')
        if init_method not in ('zeros', 'eye', 'normal'):
            raise ValueError('init_method can be either zeros, eye, or normal')

        self.dim_grid = dim_grid
        self.dim_payload = dim_payload
        self.tt_rank_max = tt_rank_max
        self.tt_rank_equal = tt_rank_equal
        self.tt_minimal_dof = tt_minimal_dof
        self.init_method = init_method
        self.outliers_handling = outliers_handling
        self.sample_by_contraction = sample_by_contraction
        self.expected_sample_batch_size = expected_sample_batch_size
        self.version_sample_qtt = version_sample_qtt
        self.dtype = dtype
        self.checks = checks
        self.verbose = verbose

        self.dim_grid_log2 = int(torch.log2(torch.tensor(dim_grid)).item())
        self.reshape_plan = get_qtt3d_reshape_plan(self.dim_grid_log2, dim_payload, payload_last=True)
        self.grid_shape = self.reshape_plan['shape_src']
        self.tt_modes = self.reshape_plan['shape_dst']
        self.num_cores = len(self.tt_modes)
        self.tt_ranks = get_tt_ranks(self.tt_modes, max_rank=tt_rank_max, tt_rank_equal=tt_rank_equal)

        tt_rank_max_possible = max(get_tt_ranks(self.tt_modes, max_rank=None))
        if tt_rank_max > tt_rank_max_possible:
            if tt_rank_equal:
                if verbose:
                    print(f'Warning: Current grid becomes redundant with tt_rank_max>{tt_rank_max_possible}, '
                          f'given {tt_rank_max}')
            else:
                raise ValueError(f'tt_rank_max must not exceed {tt_rank_max_possible} for {dim_grid=} and {dim_payload=}')
        self.tt_core_shapes = [
            (self.tt_ranks[i], self.tt_modes[i], self.tt_ranks[i+1])
            for i in range(self.num_cores)
        ]

        if tt_minimal_dof:
            self.tt_core_isparam = [s[0] * s[1] != s[2] and s[0] != s[1] * s[2] for s in self.tt_core_shapes]
            if not any(self.tt_core_isparam):
                # full rank case, need to appoint one (largest) core as a parameter
                core_sizes = [s[0] * s[1] * s[2] for s in self.tt_core_shapes]
                largest_core_idx = core_sizes.index(max(core_sizes))
                self.tt_core_isparam[largest_core_idx] = True
        else:
            self.tt_core_isparam = [True] * self.num_cores

        for i in range(self.num_cores):
            if self.tt_core_isparam[i]:
                self.register_parameter(
                    self._get_core_name_by_id(i),
                    torch.nn.Parameter(torch.zeros(*self.tt_core_shapes[i], dtype=dtype))
                )
            else:
                core_shape = self.tt_core_shapes[i]
                if core_shape[0] == core_shape[1] * core_shape[2]:
                    eye_size = core_shape[0]
                else:
                    eye_size = core_shape[2]
                buf_init = torch.eye(eye_size, dtype=dtype).reshape(core_shape)
                self.register_buffer(self._get_core_name_by_id(i), buf_init)

        if init_method == 'normal':
            num_buffers_on_the_left = self.tt_core_isparam.index(True)
            num_buffers_on_the_right = list(reversed(self.tt_core_isparam)).index(True)
            ranks_between_two_param_cores = self.tt_ranks[1+num_buffers_on_the_left : -1-num_buffers_on_the_right]
            d = sum([int(a) for a in self.tt_core_isparam])
            sigma_cores = (-torch.tensor(ranks_between_two_param_cores).double().log().sum() / (2. * d)).exp().item()
            for i, c in enumerate(self.get_cores()):
                if not self.tt_core_isparam[i]:
                    continue
                with torch.no_grad():
                    c.copy_((torch.randn_like(c) * sigma_cores).to(dtype))
        elif init_method == 'eye':
            for i, c in enumerate(self.get_cores()):
                if not self.tt_core_isparam[i]:
                    continue
                with torch.no_grad():
                    c.copy_(torch.eye(
                        self.tt_ranks[i], self.tt_ranks[i+1], dtype=dtype
                    ).unsqueeze(1).repeat(1, self.tt_modes[i], 1))

        self.fn_contract_samples = None
        self.fn_sample_complexity = None
        if self.version_sample_qtt == 1:
            self.fn_contract_samples, self.fn_sample_complexity = \
                sample_intcoord_tt_v1__compile_tt_contraction_fn(
                    self.tt_core_shapes,
                    self.expected_sample_batch_size,
                    last_core_is_payload=True,
                    report_flops=True
                )
        elif self.version_sample_qtt == 2:
            self.fn_sample_complexity = perf_report_sample_tt_v2(self.tt_core_shapes)
        elif self.version_sample_qtt == 3:
            self.fn_sample_complexity = perf_report_sample_tt_v3(
                self.tt_core_shapes, self.tt_core_isparam
            )
        self.fn_contract_grid, self.fn_contract_grid_complexity = compile_tt_contraction_fn(
            self.tt_core_shapes,
            report_flops=True
        )

        self.dtype_sz_bytes = {
            torch.float16: 2,
            torch.float32: 4,
            torch.float64: 8,
        }[self.dtype]
        self.num_uncompressed_params = torch.prod(torch.tensor(self.grid_shape)).item()
        self.num_compressed_params = sum([torch.prod(torch.tensor(p.shape)) for p in self.parameters()]).item()
        self.sz_uncompressed_gb = self.num_uncompressed_params * self.dtype_sz_bytes / (1024 ** 3)
        self.sz_compressed_gb = self.num_compressed_params * self.dtype_sz_bytes / (1024 ** 3)
        self.compression_factor = self.num_uncompressed_params / self.num_compressed_params

        if verbose:
            print(f'Sampling complexity:\n{self.fn_sample_complexity}')
            print(f'Grid contraction complexity:\n{self.fn_contract_grid_complexity}')

    def forward(self, coords_xyz):
        """
        (Distributed-)Data-Parallel-friendly wrapper around all sampling functions flavors.
        :param coords_xyz (torch.Tensor): Coordinates of N points in three dimensions within the voxel grid specified
            by the resolution parameter, or the native resolution when it is not specified. The tensor must have a shape
            (N, 3), be of compatible floating point dtype. Note that the allowed range of coordinates is in
            [0, resolution-1] when it is specified, or [0, dim_grid-1] when it is not specified. This parameter can be
            omitted when return_cores is True.
        :return: torch.Tensor of a shape (N, dim_payload) containing payloads at the sampled points, or TT cores
            corresponding to the selected resolution, if return_cores is True.
        """
        return sample_qtt3d(
            self.get_cores(),
            coords_xyz,
            fn_contract_samples=self.fn_contract_samples,
            fn_contract_grid=self.fn_contract_grid,
            reshape_plan=self.reshape_plan,
            sample_by_contraction=self.sample_by_contraction,
            outliers_handling=self.outliers_handling,
            tt_core_isparam=self.tt_core_isparam,
            version=self.version_sample_qtt,
            checks=self.checks
        )

    @staticmethod
    def _get_core_name_by_id(i):
        return f'core{i:02d}'

    def _get_core(self, i):
        return getattr(self, self._get_core_name_by_id(i))

    def get_cores(self):
        return [self._get_core(i) for i in range(self.num_cores)]

    def init_with_decomposition(self, voxel_grid):
        """
        Sets the initial weights to represent the supplied voxel grid, to the extent the selected tt_rank_max permits.
        :param voxel_grid (torch.Tensor): Uncompressed voxel grid of compatible dimensions.
        :return: None
        """
        if not (torch.is_tensor(voxel_grid) and voxel_grid.dtype == self.dtype):
            raise ValueError('Incompatible voxel grid format')
        if list(voxel_grid.shape) != self.grid_shape:
            raise ValueError(f'Incompatible voxel dimensions, expecting {self.grid_shape}, got {voxel_grid.shape}')
        qtt = convert_tensor_to_qtt(voxel_grid, self.reshape_plan, self.tt_rank_max)
        if shapes(qtt) != self.tt_core_shapes:
            raise ValueError(
                f'Unexpected TT shapes of the decomposition: {shapes(qtt)}; expected {self.tt_core_shapes}'
            )
        with torch.no_grad():
            for i in range(self.num_cores):
                self._get_core(i).copy_(qtt[i])
            if self.tt_minimal_dof:
                self.reduce_parameterization(self.get_cores())

    @staticmethod
    @torch.no_grad()
    def reduce_parameterization(cores):
        if not is_tt(cores):
            raise ValueError('Input is not a Tensor Train')
        for i, c in enumerate(cores):
            if c.shape[0] * c.shape[1] != c.shape[2] or i == len(cores) - 1:
                break
            r = c.shape[2]
            i_neigh = i + 1
            c_neigh = cores[i_neigh]
            s_neigh = c_neigh.shape
            c_neigh_new = c.view(-1, r).mm(c_neigh.view(r, -1)).view(s_neigh)
            c_neigh.copy_(c_neigh_new)
            c.copy_(torch.eye(r, device=c.device, dtype=c.dtype).view(c.shape))
        for i, c in enumerate(reversed(cores)):
            if c.shape[0] != c.shape[1] * c.shape[2] or i == len(cores) - 1:
                break
            r = c.shape[0]
            i_neigh = len(cores) - 2 - i
            c_neigh = cores[i_neigh]
            s_neigh = c_neigh.shape
            c_neigh_new = c_neigh.view(-1, r).mm(c.view(r, -1)).view(s_neigh)
            c_neigh.copy_(c_neigh_new)
            c.copy_(torch.eye(r, device=c.device, dtype=c.dtype).view(c.shape))

    def contract(self):
        """
        Computes the entire uncompressed voxel grid.
        Caution: may cause out-of-memory.
        :return: torch.Tensor of shape (dim_grid, dim_grid, dim_grid, dim_payload).
        """
        out = convert_qtt_to_tensor(
            self.get_cores(),
            qtt_reshape_plan=self.reshape_plan,
            fn_contract=self.fn_contract_grid,
            checks=self.checks,
        )
        return out

    def load_from_checkpoint(self, path, strict=False):
        cores = torch.load(path)
        prefix = os.path.commonprefix(cores.keys())
        if not prefix.endswith('.core'):
            raise ValueError(f'Invalid checkpoint at path {path}: prefix="{prefix}"')
        cores = {k[len(prefix):]: v for k, v in cores.items()}
        cores = [cores[f'{i:02d}'] for i in range(len(cores))]
        num_cores = len(cores)
        tt_core_shapes = shapes(cores)
        tt_ranks = [s[0] for s in tt_core_shapes] + tt_core_shapes[-1][-1]
        if not is_tt_shapes(tt_core_shapes):
            raise ValueError(f'Invalid checkpoint at path {path}: core shapes are not TT: {tt_core_shapes}')
        if strict:
            if tt_core_shapes != self.tt_core_shapes:
                raise ValueError(
                    f'Invalid checkpoint at path {path}: mismatching cores {tt_core_shapes} != {self.tt_core_shapes}'
                )
        else:
            if num_cores > self.num_cores or \
                    any(tt_ranks[i] > self.tt_ranks[i] for i in range(num_cores)) or \
                    tt_ranks[-2] > self.tt_ranks[-2]:
                raise ValueError(
                    f'Invalid checkpoint at path {path}: mismatching number of cores {num_cores} > {self.num_cores}'
                )
        if self.verbose:
            if strict:
                print(f'Loading TT {tt_core_shapes}')
            else:
                print(f'Loading TT {tt_core_shapes} into {self.tt_core_shapes}')
        num_updated_cores = min(self.num_cores, num_cores)
        with torch.no_grad():
            for i in range(num_updated_cores-1):
                self._get_core(i)[:tt_ranks[i], :, :tt_ranks[i+1]].copy_(cores[i])
            self._get_core(self.num_cores-1)[:tt_ranks[-2], :, :tt_ranks[-1]].copy_(cores[-1])

    def extra_repr(self) -> str:
        core_shapes_status = ', '.join([
            f"{c} ({'param' if p else 'buffer'})"
            for c, p in zip(self.tt_core_shapes, self.tt_core_isparam)
        ])
        return \
            f'parameterized voxel grid: {self.grid_shape}\n' + \
            f'number of uncompressed parameters: {self.num_uncompressed_params}\n' + \
            f'number of compressed parameters: {self.num_compressed_params}\n' + \
            f'size uncompressed: {self.sz_uncompressed_gb:.3f} Gb\n' \
            f'size compressed: {self.sz_compressed_gb:.3f} Gb\n' \
            f'compression factor: {self.compression_factor:.3f}x\n' + \
            f'sample flops: {self.fn_sample_complexity["flops"]}\n' + \
            f'sample mem max: {self.fn_sample_complexity["size_max_intermediate"]}\n' + \
            f'sample mem sum: {self.fn_sample_complexity["size_all_intermediate"]}\n' + \
            f'core shapes: {core_shapes_status}\n' + \
            f'dtype: {self.dtype}'
