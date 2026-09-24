import cheetah.api as ch
from cheetah.api import ty
from cheetah.index_tools import DimName, Dims
import math
from contextlib import contextmanager
from typing import Union, Callable, Optional
from flashtransformer.utils.cuda_utils import get_default_elems_per_transaction


def check_divisible(
    dims: Dims, lhs: DimName, rhs: tuple[Union[DimName, int], ...]
) -> bool:
    size = dims.size(lhs)
    for d in rhs:
        if isinstance(d, DimName):
            size /= dims.size(d)
        else:
            size /= d
    return size % 1 == 0


def get_nondivisible_dim(
    dims: Dims, lhs: DimName, rhs: tuple[DimName, ...]
) -> Optional[DimName]:

    # return the first dimension of rhs such that
    # lhs is not divisible by the product of it and the dimensions after it
    # return none if lhs is evenly divisible
    size = dims.size(lhs)
    for d in reversed(rhs):
        size /= dims.size(d)
        if size % 1 != 0:
            return d
    return None


def make_overflow_idx(
    dims: Dims,
    lhs: DimName,
    rhs: tuple[DimName, ...],
    check_idx: Optional[DimName],
    iter_idx: DimName,
) -> tuple[DimName, Callable]:
    # for equation eq, which contains check_idx
    # and describes a subproblem which should be conditionally computed
    # make a new dim called overflow_idx
    # which is larger than the lhs
    # and evenly divides the subproblem size
    # then, create an eqn overflow_idx = rhs
    # with iter_idx's size set as needed
    # returns:
    #     overflow_idx
    #     and a function which checks if overflow_idx is larger than lhs
    #         and returns a bool

    # first, make a new dim called overflow_idx
    assert check_idx != iter_idx
    if check_idx is None:
        dims.eq(lhs, rhs)

        @contextmanager
        def overflow_check_and_set(ix):
            with ix.scope():
                yield

        return lhs, overflow_check_and_set
    assert check_idx is not None
    assert iter_idx is not None
    overflow_idx = dims.new_dim("overflow_idx")
    problem_idx = dims.new_dim("problem_idx")
    problem_overflow_idx = dims.new_dim("problem_overflow_idx")
    # figure out the size of maybe_frac_idx
    size = dims.size(lhs)
    after_check_idx = False
    subproblem_size = 1
    after_indices = []
    before_indices = []
    cur_idx_set = before_indices
    for d in rhs:
        cur_idx_set.append(d)
        if after_check_idx:
            subproblem_size *= dims.size(d)
        else:
            assert size % 1 == 0, f"lhs {lhs} is not divisible by {before_indices}"
        if d != iter_idx:
            size /= dims.size(d)
        if d == check_idx:
            assert cur_idx_set == before_indices
            after_check_idx = True
            cur_idx_set = after_indices
    dims.eq(problem_overflow_idx, tuple(before_indices))
    # check that the lhs is evenly divisible by the iter_problem_size
    # round size up
    size = math.ceil(size)
    dims.eq(iter_idx, size)
    # then, create an eqn overflow_idx = rhs
    dims.eq(overflow_idx, (problem_overflow_idx, *after_indices))
    dims.eq(lhs, (problem_idx, *after_indices))

    assert (
        dims.size(lhs) % subproblem_size == 0
    ), f"lhs {lhs} does not divide {after_indices}"

    # and a function which checks if overflow_idx is larger than lhs
    # and only runs the code within its body if the check passes
    @contextmanager
    def overflow_check_and_set(ix):
        with ix.scope():
            for d in after_indices:
                if d not in ix.state.indices_partial:
                    ix.set_index(d, 0)
            # ix.debug_check(before_indices + after_indices)
            cond = ix[overflow_idx] < ix.size(lhs)
        with ch.if_(cond):
            ix.set_index(problem_idx, ix[problem_overflow_idx])
            yield

    return overflow_idx, overflow_check_and_set


def smart_div_size(
    dims: Dims,
    dividend_dims: DimName | tuple[DimName, ...],
    divisor_dims: DimName | tuple[DimName, ...],
    missing_ok: bool = True,
) -> int:
    if type(dividend_dims) == tuple:
        dividend_size = 1
        for d in dividend_dims:
            size = dims.size(d, error_on_missing=False)
            if size is None:
                if not missing_ok:
                    raise ValueError(f"Unsolved dividend dimensions: {d}")
            dividend_size *= size
    elif type(dividend_dims) == DimName:
        dividend_size = dims.size(dividend_dims, error_on_missing=False)
        if dividend_size is None:
            if missing_ok:
                dividend_size = 1
            else:
                raise ValueError(f"Unsolved dividend dimensions: {dividend_dims}")
    else:
        raise ValueError(f"Invalid dividend dimensions: {dividend_dims}")

    if type(divisor_dims) == tuple:
        divisor_size = 1
        for d in divisor_dims:
            size = dims.size(d, error_on_missing=False)
            if size is None:
                if not missing_ok:
                    raise ValueError(f"Unsolved divisor dimensions: {d}")
            else:
                divisor_size *= size
    elif type(divisor_dims) == DimName:
        divisor_size = dims.size(divisor_dims, error_on_missing=False)
        if divisor_size is None:
            if missing_ok:
                divisor_size = 1
            else:
                raise ValueError(f"Unsolved divisor dimensions: {divisor_dims}")
    else:
        raise ValueError(f"Invalid divisor dimensions: {divisor_dims}")

    if dividend_size % divisor_size != 0:
        raise ValueError(
            f"""Dividend dimensions {dividend_dims} must be divisible by divisor dimensions {divisor_dims}
            dividend_size: {dividend_size}
            divisor_size: {divisor_size}
            """
        )
    size = dividend_size // divisor_size
    return size


def smart_vector_idx_size(
    dims, dividend_dims, divisor_dims, ptr_type, memtype, missing_ok=True
):
    default_vector_idx_size = get_default_elems_per_transaction(ptr_type, memtype)
    max_size = smart_div_size(dims, dividend_dims, divisor_dims, missing_ok)
    if max_size > default_vector_idx_size:
        assert max_size % default_vector_idx_size == 0
        return default_vector_idx_size
    return max_size
