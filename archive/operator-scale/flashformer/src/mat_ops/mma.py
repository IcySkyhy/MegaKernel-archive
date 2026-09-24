from flashtransformer.utils.safe_data_index_ptr_utils import (
    SafeDataIndexPtr,
    SDIPConfig,
)
from flashtransformer.utils import LocalBarrier
from flashtransformer.components.memory.sync_copy import ThreadVectorCopyFn
from cheetah.index_tools import DimName, Indices, Dims
import cheetah.api as ch
from cheetah.api import ty
from flashtransformer.utils.scope_fn_utils import SmartScopeFn
from contextlib import contextmanager


class MMAMatmulFn(SmartScopeFn):
    def __init__(
        self,
        dims: Dims,
        mat1_cfg: SDIPConfig,
        mat2_cfg: SDIPConfig,
        out_cfg: SDIPConfig,
        warp_idx: DimName,
        lane_idx: DimName,
        m_idx: DimName,
        k_idx: DimName,
        n_idx: DimName,
        accumulate: bool = False,
        mat1_k_major: bool = True,
        mat2_k_major: bool = True,
    ):
        self.sdip_configs = [mat1_cfg, mat2_cfg, out_cfg]
        super().__init__(
            dims,
            mat1_cfg,
            mat2_cfg,
            out_cfg,
            warp_idx,
            lane_idx,
            m_idx,
            k_idx,
            n_idx,
            accumulate,
            mat1_k_major,
            mat2_k_major,
        )

    def setup(
        self,
        mat1_cfg: SDIPConfig,
        mat2_cfg: SDIPConfig,
        out_cfg: SDIPConfig,
        warp_idx,
        lane_idx,
        m_idx,
        k_idx,
        n_idx,
        accumulate=False,
        mat1_k_major=True,
        mat2_k_major=True,
    ):
        self.mat1_cfg = mat1_cfg
        self.mat2_cfg = mat2_cfg
        self.out_cfg = out_cfg

        if self.dims.size(m_idx) < 8:
            self.m_idx = self.dims.new_dim("m_idx", 8)
            self.limit_m_idx = self.dims.new_dim("limit_m_idx", self.dims.size(m_idx))
            self.limit_mat1 = True
            self.limit_out = True
        else:
            self.m_idx = m_idx
            self.limit_m_idx = m_idx
            self.limit_mat1 = False
            self.limit_out = False

        assert mat1_k_major
        assert self.dims.size(m_idx) <= 8, "HAHAHAHA"

        in_ptr_type = mat1_cfg.ptr_type.try_to_ptr()[0]
        out_ptr_type = out_cfg.ptr_type.try_to_ptr()[0]
        assert in_ptr_type == ty.bf16  # will add more types later
        assert out_ptr_type == ty.f32

        self.accumulate = accumulate
        self.warp_idx = warp_idx
        self.lane_idx = lane_idx
        self.k_idx = k_idx
        self.n_idx = n_idx
        self.in_type = in_ptr_type
        self.out_type = out_ptr_type
        self.mat1_k_major = mat1_k_major
        self.mat2_k_major = mat2_k_major

        if mat1_k_major:
            self.mat1_ptr_idx = self.dims.new_dim_with_eq(
                "mat1_ptr_idx", (self.limit_m_idx, self.k_idx)
            )
        else:
            assert not self.limit_mat1, "mat1 is not k major"
            self.mat1_ptr_idx = self.dims.new_dim_with_eq(
                "mat1_ptr_idx", (self.k_idx, self.m_idx)
            )
        if mat2_k_major:
            self.mat2_ptr_idx = self.dims.new_dim_with_eq(
                "mat2_ptr_idx", (self.n_idx, self.k_idx)
            )
        else:
            self.mat2_ptr_idx = self.dims.new_dim_with_eq(
                "mat2_ptr_idx", (self.k_idx, self.n_idx)
            )
        self.out_ptr_idx = self.dims.new_dim_with_eq(
            "out_ptr_idx", (self.limit_m_idx, self.n_idx)
        )

        # Create configs with the proper indices using the overloaded check_convert
        self.mat1_ptr_cfg = self.mat1_cfg.check_convert(
            self.mat1_ptr_idx, None, "mat1_ptr"
        )
        self.mat2_ptr_cfg = self.mat2_cfg.check_convert(
            self.mat2_ptr_idx, None, "mat2_ptr"
        )
        self.out_ptr_cfg = self.out_cfg.check_convert(self.out_ptr_idx, None, "out_ptr")

        self.k_iter_idx = self.dims.new_dim("k_iter_idx")
        self.n_iter_idx = self.dims.new_dim("n_iter_idx")

        # this for now
        self.m_tile_subidx = self.dims.new_dim("m_tile_subidx", 8)
        self.k_tile_subidx = self.dims.new_dim("k_tile_subidx", 8)
        self.n_tile_subidx = self.dims.new_dim("n_tile_subidx", 8)

        self.bot_half_m_idx = self.dims.new_dim("bot_half_m_idx", 2)
        self.out_cache_subidx = self.dims.new_dim_with_eq("out_cache_subidx", 2)
        if self.mat2_k_major:
            self.out_cache_idx = self.dims.new_dim_with_eq(
                "out_cache_idx", (self.n_iter_idx, self.out_cache_subidx)
            )
        else:
            self.out_cache_idx = self.dims.new_dim_with_eq(
                "out_cache_idx",
                (self.out_cache_subidx, self.n_iter_idx),
            )

        self.load_mat1_scope = self.dims.new_scope("load_mat1_scope")
        self.load_mat2_scope = self.dims.new_scope("load_mat2_scope")
        self.write_out_scope = self.dims.new_scope("write_out_scope")

        self.mat1_subcache_idx = self.dims.new_dim(
            "mat1_subcache_idx", 2  # 128 / 32, drop > 8
        )
        self.mat1_cache_idx = self.dims.new_dim_with_eq(
            "mat1_cache_idx", (self.k_iter_idx, self.mat1_subcache_idx)
        )
        self.mat2_subcache_idx = self.dims.new_dim("mat2_subcache_idx", 2)  # 64 / 32
        self.mat2_cache_idx = self.dims.new_dim_with_eq(
            "mat2_cache_idx", (self.n_iter_idx, self.mat2_subcache_idx)
        )
        self.dims.eq(self.m_idx, self.m_tile_subidx)
        self.dims.eq(self.k_idx, (self.k_iter_idx, self.k_tile_subidx))
        if self.mat2_k_major:
            self.dims.eq(
                self.n_idx, (self.warp_idx, self.n_iter_idx, self.n_tile_subidx)
            )
        else:
            self.dims.eq(
                self.n_idx, (self.warp_idx, self.n_tile_subidx, self.n_iter_idx)
            )

        # lies dormant outside scope
        self.lane_group_superidx = self.dims.new_dim("lane_group_superidx", 8)
        self.lane_group_subidx = self.dims.new_dim("lane_group_subidx", 4)
        self.dims.eq(self.lane_idx, (self.lane_group_superidx, self.lane_group_subidx))

        with self.dims.scope(self.load_mat1_scope):
            # breakdown is as follows
            self.dims.eq(self.m_tile_subidx, self.lane_group_superidx)
            self.dims.eq(
                self.k_tile_subidx, (self.lane_group_subidx, self.mat1_subcache_idx)
            )
            # Create a register SDIPConfig for mat1 cache
            mat1_cache_ptr_type = ty.ptr_mut(ty.bf16)
            self.mat1_cache_cfg = SDIPConfig(
                self.dims,
                self.mat1_cache_idx,
                mat1_cache_ptr_type,
                "register",
                "mat1_cache",
            )
            self.thread_mat1_cpy_fn = ThreadVectorCopyFn(
                dims=self.dims,
                src_cfg=self.mat1_ptr_cfg,
                dst_cfg=self.mat1_cache_cfg,
                cpy_idx=self.mat1_subcache_idx,
                volatile=False,
            )

            @contextmanager
            def mat1_limit(ix):
                if self.limit_mat1:
                    with ix.scope():
                        with ch.if_(
                            ix[self.m_tile_subidx] < self.dims.size(self.limit_m_idx)
                        ):
                            ix.set_index(self.limit_m_idx, ix[self.m_tile_subidx])
                            yield
                        ch.raw_stmt(f"__syncwarp();")
                else:
                    yield

            self.mat1_limit = mat1_limit

        with self.dims.scope(self.load_mat2_scope):
            self.dims.eq(self.n_tile_subidx, self.lane_group_superidx)
            self.dims.eq(
                self.k_tile_subidx, (self.lane_group_subidx, self.mat2_subcache_idx)
            )
            mat2_cache_ptr_type = ty.ptr_mut(ty.bf16)
            if self.mat2_k_major:
                # Create a register SDIPConfig for mat2 cache
                self.mat2_cache_cfg = SDIPConfig(
                    self.dims,
                    self.mat2_cache_idx,
                    mat2_cache_ptr_type,
                    "register",
                    "mat2_cache",
                )
                self.thread_mat2_cpy_fn = ThreadVectorCopyFn(
                    dims=self.dims,
                    src_cfg=self.mat2_ptr_cfg,
                    dst_cfg=self.mat2_cache_cfg,
                    cpy_idx=self.mat2_subcache_idx,
                    volatile=False,
                )
            else:
                if self.dims.size(self.n_iter_idx) >= 2:  # hardcoded baby
                    print(
                        f"mat2_k_major is false and n_iter_idx is {self.dims.size(self.n_iter_idx)}"
                    )
                    print(
                        f"m k n {self.dims.size(self.m_idx)} {self.dims.size(self.k_idx)} {self.dims.size(self.n_idx)}"
                    )
                # Create a register SDIPConfig for mat2 cache with n_iter_idx
                self.mat2_cache_cfg = SDIPConfig(
                    self.dims,
                    self.n_iter_idx,
                    mat2_cache_ptr_type,
                    "register",
                    "mat2_cache_n",
                )
                self.thread_mat2_cpy_fn = ThreadVectorCopyFn(
                    dims=self.dims,
                    src_cfg=self.mat2_ptr_cfg,
                    dst_cfg=self.mat2_cache_cfg,
                    cpy_idx=self.n_iter_idx,
                    volatile=False,
                )

        with self.dims.scope(self.write_out_scope):
            # breakdown is as follows
            self.dims.eq(self.m_tile_subidx, self.lane_group_superidx)
            self.dims.eq(
                self.n_tile_subidx, (self.lane_group_subidx, self.out_cache_subidx)
            )
            if self.mat2_k_major:
                self.out_loop_idx = self.n_iter_idx
                self.out_cpy_idx = self.out_cache_subidx
            else:
                self.out_cpy_idx = self.out_cache_idx
                self.out_loop_idx = self.dims.new_dim("dummy_loop_idx", 1)
            # Create a register SDIPConfig for out cache
            out_cache_ptr_type = ty.ptr_mut(ty.f32)
            self.out_cache_cfg = SDIPConfig(
                self.dims,
                self.out_cache_idx,
                out_cache_ptr_type,
                "register",
                "out_cache",
            )
            self.thread_load_out_fn = ThreadVectorCopyFn(
                dims=self.dims,
                src_cfg=self.out_ptr_cfg,
                dst_cfg=self.out_cache_cfg,
                cpy_idx=self.out_cpy_idx,
                volatile=False,
            )
            self.thread_store_out_fn = ThreadVectorCopyFn(
                dims=self.dims,
                src_cfg=self.out_cache_cfg,
                dst_cfg=self.out_ptr_cfg,
                cpy_idx=self.out_cpy_idx,
                volatile=False,
            )

            @contextmanager
            def out_limit(ix):
                if self.limit_out:
                    with ix.scope():
                        with ch.if_(
                            ix[self.m_tile_subidx] < self.dims.size(self.limit_m_idx)
                        ):
                            ix.set_index(self.limit_m_idx, ix[self.m_tile_subidx])
                            yield
                        ch.raw_stmt(f"__syncwarp();")
                else:
                    yield

            self.out_limit = out_limit

    def __call__(
        self,
        ix: Indices,
        mat1_ptr: SafeDataIndexPtr,
        mat2_ptr: SafeDataIndexPtr,
        out_ptr: SafeDataIndexPtr,
        tile,
        local_barrier,
    ):
        mat1_ptr, mat2_ptr, out_ptr = self.check_configs(mat1_ptr, mat2_ptr, out_ptr)
        super().__call__(ix, mat1_ptr, mat2_ptr, out_ptr, tile, local_barrier)

    def mma(self, ix, mat1_cache, mat2_cache, out_cache):
        d_reg_str = "$d0, $d1, $d2, $d3"
        a_reg_str = "$a0, 0.0"
        b_reg_str = "$b0"
        c_reg_str = "$c0, $c1, 0.0, 0.0"
        # a_reg_str = ", ".join([f"a{i}" for i in range(ix.size(self.mat1_))])
        command = f"""mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {{ {d_reg_str}}}
        , {{ {a_reg_str}}}
        , {{ {b_reg_str}}}
        , {{ {c_reg_str}}};"""
        asm_kwargs = {}
        for i in range(2):
            with ix.scope():
                ix.set_index(self.out_cache_subidx, i)
                asm_kwargs[f"c{i}"] = out_cache.idx_get()
        for i in range(4):
            asm_kwargs[f"d{i}"] = ty.f32
        with ix.scope():
            ix.set_index(self.mat1_subcache_idx, 0)
            ix.set_index(self.mat2_subcache_idx, 0)
            asm_kwargs["a0"] = mat1_cache.idx_offset(self.mat1_cache_idx).cast(
                ty.ptr_mut(ty.i32)
            )[0]
            asm_kwargs["b0"] = mat2_cache.idx_offset(self.mat2_cache_idx).cast(
                ty.ptr_mut(ty.i32)
            )[0]
        outputs = ch.asm_volatile(command, **asm_kwargs)
        for i in range(ix.size(self.out_cache_subidx)):
            with ix.scope():
                ix.set_index(self.out_cache_subidx, i)
                out_cache.idx_set(outputs[f"d{i}"])

    def generate(
        self,
        ix: Indices,
        mat1_ptr: SafeDataIndexPtr,
        mat2_ptr: SafeDataIndexPtr,
        out_ptr: SafeDataIndexPtr,
        tile,
        local_barrier,
    ):

        mat1_cache = SafeDataIndexPtr.from_alloc(
            ix, self.mat1_cache_idx, self.in_type, "mat1_cache"
        )
        mat2_cache = SafeDataIndexPtr.from_alloc(
            ix, self.mat2_cache_idx, self.in_type, "mat2_cache"
        )
        out_cache = SafeDataIndexPtr.from_alloc(
            ix, self.out_cache_idx, self.out_type, "out_cache"
        )
        debug = True

        with ix.scope(self.write_out_scope):
            with self.out_limit(ix):
                if self.accumulate:
                    with ix.loop(self.out_loop_idx):
                        self.thread_load_out_fn(ix, out_ptr, out_cache)
                else:
                    with ix.loop(self.out_loop_idx):
                        self.thread_store_out_fn.zero_out(ix, out_cache)
        with ix.scope(self.load_mat1_scope):
            with self.mat1_limit(ix):
                with ix.loop(self.k_iter_idx):
                    self.thread_mat1_cpy_fn(ix, mat1_ptr, mat1_cache)
        with ix.loop(self.k_iter_idx):
            with ix.scope(self.load_mat2_scope):
                if self.mat2_k_major:
                    with ix.loop(self.n_iter_idx):
                        self.thread_mat2_cpy_fn(ix, mat2_ptr, mat2_cache)
                else:
                    t_cache = SafeDataIndexPtr.from_alloc(
                        ix, self.n_iter_idx, ty.bf16, "transpose_cache"
                    )
                    with ix.loop(self.mat2_subcache_idx):
                        self.thread_mat2_cpy_fn(ix, mat2_ptr, t_cache)
                        with ix.loop(self.n_iter_idx):
                            mat2_cache.idx_set(t_cache[self.n_iter_idx])
            # run the mma!
            with ix.loop(self.n_iter_idx):
                self.mma(ix, mat1_cache, mat2_cache, out_cache)
        with ix.scope(self.write_out_scope):
            with self.out_limit(ix):
                with ix.loop(self.out_loop_idx):
                    self.thread_store_out_fn(ix, out_cache, out_ptr)

        local_barrier.wait()

    def zero_out_accumulator(
        self, ix: Indices, out_ptr: SafeDataIndexPtr, local_barrier: LocalBarrier
    ):
        with ix.scope(self.scope):
            out_ptr = self.out_ptr_cfg.wrap_sdip(out_ptr)
            with ix.scope(self.write_out_scope):
                with self.out_limit(ix):
                    self.thread_store_out_fn.zero_out(ix, out_ptr)
        local_barrier.wait()
