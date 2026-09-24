from flashtransformer.utils.safe_data_index_ptr_utils import (
    SafeDataIndexPtr,
    SDIPConfig,
)
from cheetah.index_tools import DimName, Indices, Dims
import cheetah.api as ch
from cheetah.api import ty
from flashtransformer.utils.scope_fn_utils import SmartScopeFn
from flashtransformer.utils.cuda_utils import (
    get_cast_dtype,
    get_default_elems_per_transaction,
)
from flashtransformer.utils.kernel_utils import make_cond_print
from flashtransformer.utils.index_utils import make_overflow_idx, get_nondivisible_dim
from flashtransformer.components.memory.sync_copy import (
    ThreadVectorCopyFn,
    WGSyncCopyFn,
)
from typing import Optional
from math import gcd


class RegisterTypeCastFn(SmartScopeFn):
    """
    Thread-level function for type casting between registers.
    Takes a source register pointer of one type and casts to a destination register pointer of another type.
    Works on a single vector at a time for optimal register usage.
    """

    def __init__(
        self,
        dims: Dims,
        src_cfg: SDIPConfig,
        dst_cfg: SDIPConfig,
        cpy_idx: DimName,
        debug: bool = False,
    ):
        self.debug = debug
        self.sdip_configs = [src_cfg, dst_cfg]
        super().__init__(dims, src_cfg, dst_cfg, cpy_idx)

    def setup(
        self,
        src_cfg: SDIPConfig,
        dst_cfg: SDIPConfig,
        cpy_idx: DimName,
    ):
        self.src_cfg = src_cfg
        self.dst_cfg = dst_cfg
        self.cpy_idx = cpy_idx

        # Verify that both configs are for register memory
        assert (
            src_cfg.memtype == "register"
        ), "Source must be in registers for RegisterTypeCastFn"
        assert (
            dst_cfg.memtype == "register"
        ), "Destination must be in registers for RegisterTypeCastFn"

        # Get element types for source and destination
        self.src_elem_type = src_cfg.ptr_type.try_to_ptr()[0]
        self.dst_elem_type = dst_cfg.ptr_type.try_to_ptr()[0]

        # Check if type conversion is actually needed
        self.type_conversion_needed = self.src_elem_type != self.dst_elem_type

    def __call__(
        self, ix: Indices, src_ptr: SafeDataIndexPtr, dst_ptr: SafeDataIndexPtr
    ):
        src_ptr, dst_ptr = self.check_configs(src_ptr, dst_ptr)
        super().__call__(ix, src_ptr, dst_ptr)

    def generate(
        self, ix: Indices, src_ptr: SafeDataIndexPtr, dst_ptr: SafeDataIndexPtr
    ):
        # Loop through the copy index (vector) and perform element-wise type casting
        with ix.loop(self.cpy_idx):
            if self.type_conversion_needed:
                # Typecast from source to destination type
                dst_ptr.idx_set(src_ptr.idx_get().cast(self.dst_elem_type))
            else:
                # Simple copy if types are the same
                dst_ptr.idx_set(src_ptr.idx_get())


class WGTypeCastCopyFn(SmartScopeFn):
    """
    Work-group copy function with type casting support.
    Uses a vector-first approach: first copies data into registers, then typecasts,
    then copies to destination. This ensures better performance and simpler code.
    """

    def __init__(
        self,
        dims: Dims,
        src_cfg: SDIPConfig,
        dst_cfg: SDIPConfig,
        thread_idx: DimName,
        cpy_idx: DimName,
        elems_per_load: int = 4,
        volatile: bool = False,
        debug: bool = False,
    ):
        self.debug = debug
        self.sdip_configs = [src_cfg, dst_cfg]
        super().__init__(
            dims,
            src_cfg,
            dst_cfg,
            thread_idx,
            cpy_idx,
            elems_per_load,
            volatile,
        )

    def setup(
        self,
        src_cfg: SDIPConfig,
        dst_cfg: SDIPConfig,
        thread_idx: DimName,
        cpy_idx: DimName,
        elems_per_load: Optional[int] = None,
        volatile: bool = False,
    ):
        self.src_cfg = src_cfg
        self.dst_cfg = dst_cfg
        self.thread_idx = thread_idx
        self.cpy_idx = cpy_idx
        self.volatile = volatile
        self.src_memtype = src_cfg.memtype
        self.dst_memtype = dst_cfg.memtype
        self.src_elem_type = src_cfg.ptr_type.try_to_ptr()[0]
        self.dst_elem_type = dst_cfg.ptr_type.try_to_ptr()[0]
        self.type_conversion_needed = self.src_elem_type != self.dst_elem_type

        # Calculate elements per load if not specified
        if elems_per_load is None:
            init_elems_per_load = self.dims.size(self.cpy_idx) // self.dims.size(
                self.thread_idx
            )
            if src_cfg.memtype == "global" or dst_cfg.memtype == "global":
                memtype = "global"
            elif src_cfg.memtype == "shared" or dst_cfg.memtype == "shared":
                memtype = "shared"
            else:
                memtype = "register"
            elems_per_load = gcd(
                init_elems_per_load,
                get_default_elems_per_transaction(self.src_cfg.ptr_type, memtype),
            )

        # Calculate total threads and elements that can be copied per iteration
        self.vector_idx = self.dims.new_dim("vector_idx", elems_per_load)
        check_dim = get_nondivisible_dim(
            self.dims, self.cpy_idx, (self.thread_idx, self.vector_idx)
        )
        self.iter_idx = self.dims.new_dim("iter_idx")

        if check_dim is self.thread_idx:
            self.overflow_idx, self.overflow_check_and_set = make_overflow_idx(
                self.dims,
                self.cpy_idx,
                (self.iter_idx, self.thread_idx, self.vector_idx),
                self.thread_idx,
                self.iter_idx,
            )
            self.needs_to_check_size = True
        elif check_dim is None:
            self.dims.eq(
                self.cpy_idx, (self.iter_idx, self.thread_idx, self.vector_idx)
            )
            self.needs_to_check_size = False
        else:
            raise ValueError(f"Invalid check dimension: {check_dim}")

        # Create configurations and functions for different memory and type combinations
        if not self.type_conversion_needed:
            # No type conversion needed, use direct copy
            self.direct_copy = ThreadVectorCopyFn(
                self.dims,
                src_cfg=src_cfg,
                dst_cfg=dst_cfg,
                cpy_idx=self.vector_idx,
                volatile=volatile,
            )
            return

        # Type conversion is needed: prepare for register-based typecast operation

        # Define register configs only if needed
        # Source register config - only needed if source is not already in registers
        if self.src_memtype != "register":
            self.src_reg_idx = self.dims.new_dim_with_eq("src_reg_idx", self.vector_idx)
            self.src_reg_cfg = SDIPConfig(
                self.dims,
                self.src_reg_idx,
                ty.ptr_mut(self.src_elem_type),
                "register",
                "src_reg",
            )

            # Create copy function from source memory to source registers
            self.src_to_reg_copy = ThreadVectorCopyFn(
                self.dims,
                src_cfg=src_cfg,
                dst_cfg=self.src_reg_cfg,
                cpy_idx=self.vector_idx,
                override_types=True,
            )

        # Destination register config - only needed if destination is not already in registers
        if self.dst_memtype != "register":
            self.dst_reg_idx = self.dims.new_dim_with_eq("dst_reg_idx", self.vector_idx)
            self.dst_reg_cfg = SDIPConfig(
                self.dims,
                self.dst_reg_idx,
                ty.ptr_mut(self.dst_elem_type),
                "register",
                "dst_reg",
            )

            # Create copy function from destination registers to destination memory
            self.reg_to_dst_copy = ThreadVectorCopyFn(
                self.dims,
                src_cfg=self.dst_reg_cfg,
                dst_cfg=dst_cfg,
                cpy_idx=self.vector_idx,
                override_types=True,
            )

        # Create register-to-register typecast functions based on memory types
        if self.src_memtype == "register" and self.dst_memtype == "register":
            # Both source and destination are already in registers
            self.reg_typecast = RegisterTypeCastFn(
                self.dims,
                self.src_cfg,
                self.dst_cfg,
                self.vector_idx,
                debug=self.debug,
            )
        elif self.src_memtype == "register":
            # Source is in register, destination is in memory
            self.reg_typecast = RegisterTypeCastFn(
                self.dims,
                self.src_cfg,
                self.dst_reg_cfg,
                self.vector_idx,
                debug=self.debug,
            )
        elif self.dst_memtype == "register":
            # Source is in memory, destination is in register
            self.reg_typecast = RegisterTypeCastFn(
                self.dims,
                self.src_reg_cfg,
                self.dst_cfg,
                self.vector_idx,
                debug=self.debug,
            )
        else:
            # Neither source nor destination is in registers
            self.reg_typecast = RegisterTypeCastFn(
                self.dims,
                self.src_reg_cfg,
                self.dst_reg_cfg,
                self.vector_idx,
                debug=self.debug,
            )

    def __call__(
        self,
        ix: Indices,
        src_ptr: SafeDataIndexPtr,
        dst_ptr: SafeDataIndexPtr,
        local_barrier,
    ):
        src_ptr, dst_ptr = self.check_configs(src_ptr, dst_ptr)
        super().__call__(ix, src_ptr, dst_ptr, local_barrier)

    def generate(
        self,
        ix: Indices,
        src_ptr: SafeDataIndexPtr,
        dst_ptr: SafeDataIndexPtr,
        local_barrier,
    ):
        # Create debug print function if debugging is enabled
        debug_print = make_cond_print(
            ch.and_(ix[self.thread_idx] == 0), self.debug, local_barrier
        )

        debug_print(
            f"TypeCastCopy: src_memtype={self.src_memtype}, dst_memtype={self.dst_memtype}"
        )

        # Wait for all threads to be ready before starting operations
        local_barrier.wait()

        # Determine copy strategy based on memory types and conversion needs
        if not self.type_conversion_needed:
            # No type conversion needed, use direct copy
            debug_print("Using direct copy (no type conversion)")
            self._direct_copy(ix, src_ptr, dst_ptr, local_barrier)
        else:
            # Type conversion needed
            if self.src_memtype == "register" and self.dst_memtype == "register":
                # Both source and destination are in registers
                debug_print(
                    "Both source and destination are in registers, using direct register-to-register typecast"
                )
                self._iter_wrapper(ix, lambda: self.reg_typecast(ix, src_ptr, dst_ptr))
            elif self.src_memtype == "register":
                # Source is in registers, destination is not
                debug_print("Source is in registers, destination is not")
                # Allocate destination registers
                dst_reg_ptr = self.dst_reg_cfg.alloc_reg(ix)

                def process_reg_to_mem():
                    # 1. Typecast from source registers to destination registers
                    debug_print(
                        "Typecasting from source registers to destination registers"
                    )
                    self.reg_typecast(ix, src_ptr, dst_reg_ptr)

                    # 2. Copy from destination registers to destination memory
                    debug_print(
                        "Copying from destination registers to destination memory"
                    )
                    self.reg_to_dst_copy(ix, dst_reg_ptr, dst_ptr)

                self._iter_wrapper(ix, process_reg_to_mem)
            elif self.dst_memtype == "register":
                # Destination is in registers, source is not
                debug_print("Destination is in registers, source is not")
                # Allocate source registers
                src_reg_ptr = self.src_reg_cfg.alloc_reg(ix)

                def process_mem_to_reg():
                    # 1. Copy from source memory to source registers
                    debug_print("Copying from source memory to source registers")
                    self.src_to_reg_copy(ix, src_ptr, src_reg_ptr)

                    # 2. Typecast from source registers to destination registers
                    debug_print(
                        "Typecasting from source registers to destination registers"
                    )
                    self.reg_typecast(ix, src_reg_ptr, dst_ptr)

                self._iter_wrapper(ix, process_mem_to_reg)
            else:
                # Neither source nor destination is in registers
                debug_print(
                    "Vector-first approach: Copy to registers, typecast, copy to destination"
                )
                self._vector_first_copy_with_cast(
                    ix, src_ptr, dst_ptr, local_barrier, debug_print
                )

        # Wait for all threads to finish before returning
        local_barrier.wait()

    def _direct_copy(self, ix, src_ptr, dst_ptr, local_barrier):
        """Perform direct copy when no type conversion is needed."""
        if self.dims.size(self.iter_idx) > 1:
            with ix.loop(self.iter_idx):
                if self.needs_to_check_size:
                    with self.overflow_check_and_set(ix):
                        self.direct_copy(ix, src_ptr, dst_ptr)
                else:
                    self.direct_copy(ix, src_ptr, dst_ptr)
        else:
            if self.needs_to_check_size:
                with self.overflow_check_and_set(ix):
                    self.direct_copy(ix, src_ptr, dst_ptr)
            else:
                self.direct_copy(ix, src_ptr, dst_ptr)

    def _vector_first_copy_with_cast(
        self, ix, src_ptr, dst_ptr, local_barrier, debug_print
    ):
        """
        Perform vector-first copy with casting when neither source nor destination is in registers.
        1. Copy from source to source-type registers
        2. Typecast from source-type to destination-type registers
        3. Copy from destination-type registers to destination
        """

        # Common debug function to safely print float values, regardless of type
        def safe_debug_print_val(message, val):
            debug_print(message, val.cast(ty.f32))

        # Allocate register arrays
        debug_print("Allocating register arrays")
        src_reg_ptr = self.src_reg_cfg.alloc_reg(ix)
        dst_reg_ptr = self.dst_reg_cfg.alloc_reg(ix)

        def process_vector_first():
            # 1. Copy from source to source-type registers
            debug_print("1. Copying from source to source-type registers")
            self.src_to_reg_copy(ix, src_ptr, src_reg_ptr)
            safe_debug_print_val(
                "   First register value after source copy: %f", src_reg_ptr.raw_idx(0)
            )

            # 2. Typecast data from source registers to destination registers
            debug_print("2. Typecasting from source registers to destination registers")
            self.reg_typecast(ix, src_reg_ptr, dst_reg_ptr)
            safe_debug_print_val(
                "   First register value after typecast: %f", dst_reg_ptr.raw_idx(0)
            )

            # 3. Copy from destination-type registers to destination
            debug_print("3. Copying from destination registers to destination")
            self.reg_to_dst_copy(ix, dst_reg_ptr, dst_ptr)
            safe_debug_print_val(
                "   First destination value after copy: %f", dst_ptr.raw_idx(0)
            )

        # Execute the process with iteration handling
        self._iter_wrapper(ix, process_vector_first)

    def _iter_wrapper(self, ix, process_fn):
        """Helper method to handle iterations and overflow checking."""
        if self.dims.size(self.iter_idx) > 1:
            with ix.loop(self.iter_idx):
                if self.needs_to_check_size:
                    with self.overflow_check_and_set(ix):
                        process_fn()
                else:
                    process_fn()
        else:
            if self.needs_to_check_size:
                with self.overflow_check_and_set(ix):
                    process_fn()
            else:
                process_fn()
