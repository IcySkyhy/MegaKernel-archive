import cheetah.api as ch
from cheetah.api import ty
from cheetah.index_tools import ScopeFn, Indices, Dims, DimName
from typing import Optional

from flashtransformer.utils.scope_fn_utils import SmartScopeFn
from flashtransformer.utils.index_utils import make_overflow_idx, get_nondivisible_dim
from flashtransformer.utils.safe_data_index_ptr_utils import (
    SDIPConfig,
    SafeDataIndexPtr,
)
from flashtransformer.utils.barrier_utils import LocalBarrier
from flashtransformer.utils.kernel_utils import make_cond_print
from flashtransformer.components.memory import ThreadVectorCopyFn
from math import gcd
from flashtransformer.utils.cuda_utils import init_groups, group_reduce, Tile
from flashtransformer.utils.cuda_utils import get_default_elems_per_transaction
from math import gcd


class RMSNormFn(SmartScopeFn):
    """
    RMSNorm: Root Mean Square Layer Normalization with parallel reduction

    This implementation uses parallel reduction across threads to efficiently compute
    the RMS value and normalizes the activations by the RMS of the input.
    """

    def __init__(
        self,
        dims: Dims,
        in_cfg: SDIPConfig,
        out_cfg: SDIPConfig,
        weight_cfg: SDIPConfig,
        thread_idx: DimName,
        hidden_idx: DimName,
        elems_per_thread: Optional[int] = None,
        eps: float | ch.Expr = 1e-5,
    ):
        """
        Initialize the RMSNorm function.

        Args:
            dims: The dimensions object
            in_cfg: Configuration for input data
            out_cfg: Configuration for output data
            weight_cfg: Configuration for weight (scale) data
            thread_idx: Thread index dimension
            hidden_idx: Hidden dimension index
            elems_per_thread: Elements processed per thread (optional)
            eps: Small value added for numerical stability (default: 1e-5)
        """
        self.sdip_configs = [in_cfg, out_cfg, weight_cfg]
        # Pass all args to ScopeFn's init which will call our setup method
        super().__init__(
            dims,
            in_cfg,
            out_cfg,
            weight_cfg,
            thread_idx,
            hidden_idx,
            elems_per_thread,
            eps,
        )

    def setup(
        self,
        in_cfg: SDIPConfig,
        out_cfg: SDIPConfig,
        weight_cfg: SDIPConfig,
        thread_idx: DimName,
        hidden_idx: DimName,
        elems_per_thread: Optional[int] = None,
        eps: float | ch.Expr = 1e-5,
    ):
        """Setup the RMSNorm function with the given configurations."""
        self.in_cfg = in_cfg
        self.out_cfg = out_cfg
        self.weight_cfg = weight_cfg
        self.thread_idx = thread_idx
        self.hidden_idx = hidden_idx

        # If eps is already a Cheetah expression, use it directly
        if isinstance(eps, ch.Expr):
            self.eps = eps
        else:
            # Otherwise, create a constant with the float value
            self.eps = ch.const(eps, ty.f32)

        # Get the types for input, output, and weights
        self.in_type = in_cfg.get_elem_type()
        self.out_type = out_cfg.get_elem_type()
        self.weight_type = weight_cfg.get_elem_type()

        # Determine elements per thread if not specified
        if elems_per_thread is None:
            # Calculate the naive distribution based on hidden size and thread count
            naive_elems = self.dims.size(self.hidden_idx) // self.dims.size(
                self.thread_idx
            )

            # Get the ideal elements per transaction for this type and memory space
            default_elems = get_default_elems_per_transaction(self.in_type, "shared")

            # Take the GCD to find a size that divides both evenly
            elems_per_thread = gcd(naive_elems, default_elems)

            # Ensure we have at least 1 element per thread
            elems_per_thread = max(1, elems_per_thread)

        # Create vector_idx for elements processed by each thread
        self.vector_idx = self.dims.new_dim("vector_idx", elems_per_thread)

        # Check if we need to handle non-divisible dimensions
        check_dim = get_nondivisible_dim(
            self.dims, self.hidden_idx, (self.thread_idx, self.vector_idx)
        )

        self.iter_idx = self.dims.new_dim("iter_idx")

        # Handle non-divisible dimensions
        if check_dim is self.thread_idx:
            self.overflow_idx, self.overflow_check_and_set = make_overflow_idx(
                self.dims,
                self.hidden_idx,
                (self.iter_idx, self.thread_idx, self.vector_idx),
                self.thread_idx,
                self.iter_idx,
            )
            self.needs_to_check_size = True
        elif check_dim is None:
            self.dims.eq(
                self.hidden_idx, (self.iter_idx, self.thread_idx, self.vector_idx)
            )
            self.needs_to_check_size = False
        else:
            raise ValueError(f"Invalid check dimension: {check_dim}")

        # Create register SDIPs for thread-local data
        vector_in_cfg = SDIPConfig(
            self.dims, self.vector_idx, in_cfg.ptr_type, "register", "vector_in"
        )

        vector_weight_cfg = SDIPConfig(
            self.dims, self.vector_idx, weight_cfg.ptr_type, "register", "vector_weight"
        )

        # Create ThreadVectorCopyFn instances for input and weight copying
        # These copy from shared memory to registers for each thread
        self.in_copy_fn = ThreadVectorCopyFn(
            self.dims,
            src_cfg=self.in_cfg,
            dst_cfg=vector_in_cfg,
            cpy_idx=self.vector_idx,
        )

        self.weight_copy_fn = ThreadVectorCopyFn(
            self.dims,
            src_cfg=self.weight_cfg,
            dst_cfg=vector_weight_cfg,
            cpy_idx=self.vector_idx,
        )

        # Store the configurations for register caches
        self.vector_in_cfg = vector_in_cfg
        self.vector_weight_cfg = vector_weight_cfg

        # Store the hidden size for calculations
        self.hidden_size = self.dims.size(self.hidden_idx)

        # Set up a tile size for reduction - use the entire thread block for reduction
        # This should be the number of threads, not just 32 (warp size)
        self.tile_size = self.dims.size(self.thread_idx)

    def __call__(
        self,
        ix: Indices,
        in_ptr: SafeDataIndexPtr,
        out_ptr: SafeDataIndexPtr,
        weight_ptr: SafeDataIndexPtr,
        rms_shared: SafeDataIndexPtr,
        tile: Tile,
        local_barrier: LocalBarrier,
        debug: bool = False,
    ):
        """
        Perform RMSNorm operation with parallel reduction.

        Args:
            ix: Indices object
            in_ptr: Input data pointer
            out_ptr: Output data pointer
            weight_ptr: Weight (scale) data pointer
            local_barrier: Local barrier for synchronization
            rms_shared: Optional shared memory for RMS computation
            debug: Whether to enable debug prints
        """
        in_ptr, out_ptr, weight_ptr = self.check_configs(in_ptr, out_ptr, weight_ptr)

        # Call generate via super().__call__
        super().__call__(
            ix,
            in_ptr,
            out_ptr,
            weight_ptr,
            rms_shared,
            tile,
            local_barrier,
            debug,
        )

    def generate(
        self,
        ix: Indices,
        in_ptr: SafeDataIndexPtr,
        out_ptr: SafeDataIndexPtr,
        weight_ptr: SafeDataIndexPtr,
        rms_shared: SafeDataIndexPtr = None,
        tile: Tile = None,
        local_barrier: LocalBarrier = None,
        debug: bool = False,
    ):
        """
        Implementation of the RMSNorm operation.

        This function:
        1. Creates SDIP caches for input and weight data
        2. Uses ThreadVectorCopyFn to copy data to register caches
        3. Calculates sum of squares for RMS computation
        4. Uses tile-based reduction to compute total sum
        5. Applies normalization and scaling
        """
        # Configure debug printing
        self.debug_print = make_cond_print(
            ch.and_(ix[self.thread_idx] == 0), debug, local_barrier
        )
        # Initialize cooperative groups including a tile for reduction

        # Create or check the shared memory for reduction if needed

        # Create register caches for input and weight data
        input_cache = SafeDataIndexPtr.from_alloc(
            ix, self.vector_idx, self.in_type, "input_cache"
        )

        weight_cache = SafeDataIndexPtr.from_alloc(
            ix, self.vector_idx, self.weight_type, "weight_cache"
        )

        # Initialize thread's sum of squares
        thread_sum_squares = ch.alloc(ty.f32, 0.0)

        # Process the input in chunks based on iter_idx and vector_idx
        if self.dims.size(self.iter_idx) > 1:
            # Multi-iteration processing
            with ix.loop(self.iter_idx):
                # Copy data to register caches and accumulate sum of squares
                self.process_chunk(
                    ix,
                    in_ptr,
                    weight_ptr,
                    input_cache,
                    weight_cache,
                    thread_sum_squares,
                )
        else:
            # Single iteration processing
            self.process_chunk(
                ix, in_ptr, weight_ptr, input_cache, weight_cache, thread_sum_squares
            )

        # Use tile-based reduction to compute the total sum of squares
        # This gives all threads in the tile the reduced result
        self.debug_print("sum of squares: Before reduction: %f", thread_sum_squares.val)
        thread_sum_squares = group_reduce(thread_sum_squares, tile, "sum")
        self.debug_print("sum of squares: After reduction: %f", thread_sum_squares.val)

        # Calculate values to pass to debug printing
        if debug:
            debug_total_sum_squares = thread_sum_squares.val
            debug_mean_squares = debug_total_sum_squares / ch.const(
                self.hidden_size, ty.f32
            )
            debug_rms = ch.raw_expr(
                "sqrtf($ms) + $eps", ty.f32, ms=debug_mean_squares, eps=self.eps
            )
            debug_inv_rms = ch.const(1.0, ty.f32) / debug_rms
            self.debug_print(
                "RMSNorm - Sum squares: %f, Mean squares: %f, RMS: %f, Inv RMS: %f",
                debug_total_sum_squares,
                debug_mean_squares,
                debug_rms,
                debug_inv_rms,
            )

        # Thread 0 calculates RMS and stores in shared memory
        with ch.if_(ix[self.thread_idx] == 0):
            # Get the total sum of squares
            total_sum_squares = thread_sum_squares.val

            # Calculate RMS (root mean square)
            mean_squares = total_sum_squares / ch.const(self.hidden_size, ty.f32)
            # First take square root of mean_squares, then add epsilon
            rms = ch.raw_expr(
                "sqrtf($ms) + $eps", ty.f32, ms=mean_squares, eps=self.eps
            )

            # Store inverse RMS in shared memory for all threads to access
            # Using raw_idx_set to store at index 0 of our shared buffer
            rms_shared.raw_idx_set(0, ch.const(1.0, ty.f32) / rms)

        # Move debug printing outside the if statement

        # Wait for thread 0 to complete the calculation
        local_barrier.wait()

        # Get the inverse RMS value from shared memory
        # All threads need to read from index 0, not their own thread index
        inv_rms = rms_shared.raw_idx(0)

        # Apply normalization and scaling to produce the output
        if self.dims.size(self.iter_idx) > 1:
            # Multi-iteration normalization
            with ix.loop(self.iter_idx):
                # Reload the input and weight caches for each iteration
                self.in_copy_fn(ix, in_ptr, input_cache)
                self.weight_copy_fn(ix, weight_ptr, weight_cache)

                # Apply normalization with freshly loaded data
                self.apply_normalization(
                    ix, out_ptr, input_cache, weight_cache, inv_rms
                )
        else:
            # Single iteration normalization
            self.apply_normalization(ix, out_ptr, input_cache, weight_cache, inv_rms)

        # Final synchronization
        local_barrier.wait()

    def process_chunk(
        self,
        ix: Indices,
        in_ptr: SafeDataIndexPtr,
        weight_ptr: SafeDataIndexPtr,
        input_cache: SafeDataIndexPtr,
        weight_cache: SafeDataIndexPtr,
        thread_sum_squares: ch.Expr,
    ):
        """
        Process a chunk of data by copying from shared to registers and computing sum of squares.

        This function:
        1. Uses ThreadVectorCopyFn to copy input and weight data to register caches
        2. Calculates sum of squares for RMS computation
        """
        if self.needs_to_check_size:
            with self.overflow_check_and_set(ix):
                # Copy input data to register cache
                self.in_copy_fn(ix, in_ptr, input_cache)

                # Copy weight data to register cache
                self.weight_copy_fn(ix, weight_ptr, weight_cache)

                # Calculate sum of squares from register cache
                with ix.loop(self.vector_idx):
                    # Get value from register cache
                    input_val = input_cache.idx_get()

                    # Convert to f32 for computation if needed
                    if self.in_type != ty.f32:
                        input_val = input_val.cast(ty.f32)

                    # Accumulate sum of squares
                    thread_sum_squares.val += input_val * input_val
        else:
            # Copy input data to register cache
            self.in_copy_fn(ix, in_ptr, input_cache)

            # Copy weight data to register cache
            self.weight_copy_fn(ix, weight_ptr, weight_cache)

            # Calculate sum of squares from register cache
            with ix.loop(self.vector_idx):
                # Get value from register cache
                input_val = input_cache.idx_get()

                # Convert to f32 for computation if needed
                if self.in_type != ty.f32:
                    input_val = input_val.cast(ty.f32)

                # Accumulate sum of squares
                thread_sum_squares.val += input_val * input_val

    def apply_normalization(
        self,
        ix: Indices,
        out_ptr: SafeDataIndexPtr,
        input_cache: SafeDataIndexPtr,
        weight_cache: SafeDataIndexPtr,
        inv_rms: ch.Expr,
    ):
        """
        Apply normalization and scaling to produce the output.

        This function:
        1. Gets input and weight values from register caches
        2. Applies normalization (input * inv_rms)
        3. Applies scaling (normalized * weight)
        4. Stores result to output
        """
        if self.needs_to_check_size:
            with self.overflow_check_and_set(ix):
                # Normalize and scale each element
                with ix.loop(self.vector_idx):
                    # Get cached input and weight values
                    input_val = input_cache.idx_get()
                    weight_val = weight_cache.idx_get()

                    # Convert to f32 for computation if needed
                    if self.in_type != ty.f32:
                        input_val = input_val.cast(ty.f32)
                    if self.weight_type != ty.f32:
                        weight_val = weight_val.cast(ty.f32)

                    # Normalize and scale
                    output_val = input_val * inv_rms * weight_val

                    # Cast to output type if needed
                    if self.out_type != ty.f32:
                        output_val = output_val.cast(self.out_type)

                    # Store result
                    out_ptr.idx_set(output_val)
        else:
            # Normalize and scale each element
            with ix.loop(self.vector_idx):
                # Get cached input and weight values
                input_val = input_cache.idx_get()
                weight_val = weight_cache.idx_get()

                # Convert to f32 for computation if needed
                if self.in_type != ty.f32:
                    input_val = input_val.cast(ty.f32)
                if self.weight_type != ty.f32:
                    weight_val = weight_val.cast(ty.f32)

                # Normalize and scale
                output_val = input_val * inv_rms * weight_val

                # Cast to output type if needed
                if self.out_type != ty.f32:
                    output_val = output_val.cast(self.out_type)

                # Store result
                out_ptr.idx_set(output_val)
