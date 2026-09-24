import cheetah.api as ch

from cheetah.api import ty
from cheetah.index_tools import DimName, Indices, Dims

from .cuda_utils import get_elem_size
from typing import Optional, overload
from dataclasses import dataclass
from numbers import Number


class SafeDataIndexPtr:
    def __init__(
        self,
        ix: Indices,
        ptr: ch.Expr,
        ptr_idx: DimName,
        memtype: str,
        name: Optional[str] = None,
    ):
        self.ix = ix
        self._ptr = ptr
        self._ptr_idx = ptr_idx
        self.memtype = memtype
        assert memtype in ["shared", "register", "global"]
        self.name = name
        if name is not None:
            ch.set_name(name, self._ptr)

    def __repr__(self) -> str:
        return f"SafeDataIndexPtr({self.name})"

    @staticmethod
    def from_alloc(
        ix: Indices,
        ptr_idx: DimName,
        ptr_ty,
        name: Optional[str] = None,
    ) -> "SafeDataIndexPtr":
        ptr = ch.alloc_array(ptr_ty, ix.size(ptr_idx))
        return SafeDataIndexPtr(ix, ptr, ptr_idx, "register", name)

    @staticmethod
    def from_shared_dynamic(
        ix: Indices,
        shared_ptr: ch.Expr,
        ptr_idx: DimName,
        ptr_ty,
        name: Optional[str] = None,
        align_to: int = 16,
        round_up: bool = True,
    ) -> tuple["SafeDataIndexPtr", ch.Expr]:
        if not round:
            assert (
                get_elem_size(ptr_ty) * ix.size(ptr_idx) % align_to == 0
            ), f"size {get_elem_size(ptr_ty) * ix.size(ptr_idx)} % {align_to} != 0"
            offset = ix.size(ptr_idx)
        else:
            align = align_to / get_elem_size(ptr_ty)
            offset = int(align * ((ix.size(ptr_idx) + align - 1) // align))
        orig_ptr = shared_ptr.cast(ty.ptr_mut(ptr_ty))
        new_shared_ptr = orig_ptr.offset(offset)
        return SafeDataIndexPtr(ix, orig_ptr, ptr_idx, "shared", name), new_shared_ptr

    @staticmethod
    def from_shared_static(
        ix: Indices,
        ptr_idx: DimName,
        ptr_ty,
        name: Optional[str] = None,
    ) -> "SafeDataIndexPtr":
        ptr = ch.alloc_shared_array(ptr_ty, ix.size(ptr_idx))
        return SafeDataIndexPtr(ix, ptr, ptr_idx, "shared", name)

    def type(self) -> ch.Type:
        return self._ptr.type()

    def elem_type(self) -> ch.Type:
        return self._ptr.type().try_to_ptr()[0]

    def mutability(self) -> ch.core.Mutability:
        return self._ptr.type().try_to_ptr()[1]

    def cast(
        self,
        new_ptr_ty: ch.Type,
        new_ptr_idx: DimName | None = None,
        new_name: Optional[str] = None,
    ) -> "SafeDataIndexPtr":

        assert (
            new_ptr_ty.try_to_ptr() is not None
        ), f"tried to cast a ptr to {new_ptr_ty}"
        if new_ptr_idx is None:
            new_ptr_idx = self._ptr_idx
        size_ratio = self.get_elem_size() / get_elem_size(new_ptr_ty)
        assert (
            self.ix.size(new_ptr_idx) == self.ix.size(self._ptr_idx) * size_ratio
        ), f"with cast, size_ratio {size_ratio} != {self.ix.size(new_ptr_idx)} / {self.ix.size(self._ptr_idx)}"

        if new_name is None and self.name is not None:
            new_name = f"{self.name}_cast{new_ptr_ty.try_to_ptr()[0]}"
        return SafeDataIndexPtr(
            self.ix, self._ptr.cast(new_ptr_ty), new_ptr_idx, self.memtype, new_name
        )

    def ptx_memaddr_offset(self, idx: Optional[DimName] = None) -> ch.Expr:
        if self.memtype == "shared":
            return ch.raw_expr(
                "__cvta_generic_to_shared($ptr)", ty.u32, ptr=self.idx_offset(idx)
            )
        elif self.memtype == "global":
            return ch.raw_expr(
                "__cvta_generic_to_global($ptr)", ty.u64, ptr=self.idx_offset(idx)
            )

    def __getitem__(self, idx: DimName) -> ch.Expr:
        return self.idx_get(idx)

    def __setitem__(self, idx: DimName, val: ch.Expr | Number) -> None:
        return self.idx_set(idx, val)

    def start(self) -> ch.Expr:
        with self.ix.scope():
            self.ix.set_index(self._ptr_idx, 0)
            return self.offset(self._ptr_idx)

    def subidx(
        self,
        inter_idx: DimName,
        intra_idx: DimName,
        name: Optional[str] = None,
    ) -> "SafeDataIndexPtr":
        assert self.ix.size(inter_idx) * self.ix.size(intra_idx) == self.ix.size(
            self._ptr_idx
        ), f"""subidx {inter_idx} * {intra_idx} size {self.ix.size(inter_idx)} * {self.ix.size(intra_idx)} 
                  != {self._ptr_idx} size {self.ix.size(self._ptr_idx)}, for name {self.name}"""
        with self.ix.scope():
            # self.ix.set_index(inter_idx, inter_val)
            self.ix.set_index(intra_idx, 0)
            return SafeDataIndexPtr(
                self.ix, self.idx_offset(), intra_idx, self.memtype, name
            )

    def split(
        self, start_idx: DimName, end_idx: Optional[DimName] = None
    ) -> tuple["SafeDataIndexPtr", "SafeDataIndexPtr"]:
        if end_idx is None:
            assert self.ix.size(start_idx) < self.ix.size(self._ptr_idx)
            with self.ix.scope():
                return SafeDataIndexPtr(self.ix, self.start(), start_idx, self.memtype)
        else:
            assert self.ix.size(self._ptr_idx) == self.ix.size(
                start_idx
            ) + self.ix.size(end_idx)
            with self.ix.scope():
                return SafeDataIndexPtr(
                    self.ix, self.start(), start_idx, self.memtype
                ), SafeDataIndexPtr(
                    self.ix,
                    self._ptr.offset(self.ix.size(start_idx)),
                    end_idx,
                    self.memtype,
                )

    def idx_get(self, idx: Optional[DimName] = None) -> ch.Expr:
        if idx is None:
            idx = self._ptr_idx
        assert idx == self._ptr_idx, f"access idx {idx} != origin idx{self._ptr_idx}"
        return self._ptr[self.ix[self._ptr_idx]]

    def raw_idx(self, raw_idx: int) -> ch.Expr:
        return self._ptr[raw_idx]

    def idx_set_check(self, idx: DimName, val: ch.Expr | Number) -> None:
        if idx is None:
            idx = self._ptr_idx
        if self.mutability() == "const":
            raise ValueError(f"Pointer {self.name} is const")
        assert idx == self._ptr_idx, f"set idx {idx} != origin idx{self._ptr_idx}"
        self._ptr[self.ix[self._ptr_idx]] = val

    def idx_set_nocheck(self, val: ch.Expr | Number) -> None:
        if self.mutability() == "const":
            raise ValueError(f"Pointer {self.name} is const")
        self._ptr[self.ix[self._ptr_idx]] = val

    def idx_set(
        self,
        idx_or_val: DimName | ch.Expr | Number,
        val: Optional[ch.Expr | Number] = None,  # type: ignore
    ) -> None:
        if val is None:
            val = idx_or_val
            assert not isinstance(val, DimName)
            self.idx_set_nocheck(val)
        elif isinstance(idx_or_val, DimName):
            self.idx_set_check(idx_or_val, val)
        else:
            raise ValueError(f"Invalid type: {type(idx_or_val)}")

    def raw_idx_set(self, raw_idx: int, val: ch.Expr | Number) -> None:
        if self.mutability() == "const":
            raise ValueError(f"Pointer {self.name} is const")
        self._ptr[raw_idx] = val

    def idx_offset(self, idx: Optional[DimName] = None) -> ch.Expr:
        if idx is None:
            idx = self._ptr_idx
        assert idx == self._ptr_idx, f"offset idx {idx} != origin idx{self._ptr_idx}"
        return self._ptr.offset(self.ix[self._ptr_idx])

    def get_ptr_size(self) -> int:
        return self.ix.size(self._ptr_idx) * get_elem_size(self._ptr)

    def get_elem_size(self) -> int:
        assert get_elem_size(self._ptr) is not None
        return get_elem_size(self._ptr)

    def check_type(self, ptr_type: ch.Type) -> None:
        assert (
            self._ptr.type() == ptr_type
        ), f"for {self.name}, ptr type {self._ptr.type()} != {ptr_type}"

    def is_const(self) -> bool:
        return self._ptr.type().try_to_ptr()[1] == ch.core.Mutability.CONST

    def check_memtype(self, memtype: str) -> None:
        assert self.memtype == memtype

    def check_convert(
        self,
        new_ptr_idx: DimName,
        memtype: str,
        ptr_type: Optional[ch.Type] = None,
    ) -> "SafeDataIndexPtr":
        assert self.ix.size(new_ptr_idx) == self.ix.size(
            self._ptr_idx
        ), f"check_convert {new_ptr_idx} to size {self.ix.size(new_ptr_idx)} != orig size {self._ptr_idx} size{self.ix.size(self._ptr_idx)}, for {self.name}"
        assert (
            self.memtype == memtype
        ), f"check_convert {new_ptr_idx} memtype {self.memtype} != {memtype}"
        if ptr_type is not None:
            self.check_type(ptr_type)
        return SafeDataIndexPtr(
            self.ix, self._ptr, new_ptr_idx, self.memtype, self.name
        )

    def switch_ptr(self, new_ptr_sdip: "SafeDataIndexPtr") -> None:
        assert self.ix == new_ptr_sdip.ix
        assert self.memtype == new_ptr_sdip.memtype
        assert self.ptr_type == new_ptr_sdip.ptr_type
        assert self.ix.size(self._ptr_idx) == self.ix.size(new_ptr_sdip._ptr_idx)
        self._ptr.val = new_ptr_sdip._ptr.val


@dataclass
class SDIPConfig:
    dims: Dims
    idx: DimName | tuple[DimName, ...]
    ptr_type: ch.Type
    memtype: Optional[str] = None
    name: Optional[str] = None

    def __post_init__(self):
        assert self.memtype in [
            "shared",
            "register",
            "global",
        ], f"memtype must be shared, register, or global, got {self.memtype}"

        if isinstance(self.idx, tuple):
            self.idx = self.dims.new_dim_with_eq(f"{self.name}_idx", self.idx)

        assert (
            self.ptr_type.try_to_ptr() is not None
        ), f"ptr_type must be a pointer type, got {self.ptr_type}"

    def get_elem_size(self) -> int:
        return get_elem_size(self.ptr_type)

    def get_elem_type(self) -> ch.Type:
        return self.ptr_type.try_to_ptr()[0]

    def get_size(self) -> int:
        return self.dims.size(self.idx) * get_elem_size(self.ptr_type)

    def wrap_ptr(self, ix: Indices, ptr: ch.Expr) -> SafeDataIndexPtr:
        assert self.ptr_type == ptr.type()
        assert self.memtype is not None
        assert self.name is not None
        return SafeDataIndexPtr(ix, ptr, self.idx, self.memtype, self.name)

    def wrap_sdip(self, sdip: SafeDataIndexPtr) -> SafeDataIndexPtr:
        # check sizes
        return sdip.check_convert(self.idx, self.memtype, self.ptr_type)

    def alloc_reg(self, ix: Indices) -> SafeDataIndexPtr:
        assert self.memtype == "register"
        elem_type = self.ptr_type.try_to_ptr()[0]
        return SafeDataIndexPtr.from_alloc(ix, self.idx, elem_type, self.name)

    def alloc_static_shared(self, ix: Indices) -> SafeDataIndexPtr:
        """Create a SafeDataIndexPtr from static shared memory using the configuration in this SDIPConfig."""
        assert self.memtype == "shared", f"memtype must be shared, got {self.memtype}"
        elem_type = self.ptr_type.try_to_ptr()[0]
        return SafeDataIndexPtr.from_shared_static(ix, self.idx, elem_type, self.name)

    def alloc_dynamic_shared(
        self, ix: Indices, raw_shared_ptr: ch.Expr
    ) -> tuple[SafeDataIndexPtr, ch.Expr]:
        assert self.memtype == "shared", f"memtype must be shared, got {self.memtype}"
        elem_type = self.ptr_type.try_to_ptr()[0]
        return SafeDataIndexPtr.from_shared_dynamic(
            ix, raw_shared_ptr, self.idx, elem_type, self.name
        )

    @staticmethod
    def fill_buffer(
        buffer_cfg: "SDIPConfig", row_cfg: "SDIPConfig"
    ) -> tuple[DimName, "SDIPConfig"]:
        dims = buffer_cfg.dims
        assert dims == row_cfg.dims
        row_idx = dims.new_dim("row_idx")
        dtype_size_ratio = get_elem_size(buffer_cfg.ptr_type) // get_elem_size(
            row_cfg.ptr_type
        )
        dsr_idx = dims.new_dim("dsr_idx", dtype_size_ratio)
        dims.eq(buffer_cfg.idx, (row_idx, row_cfg.idx, dsr_idx))
        row_buffer_idx = dims.new_dim_with_eq("row_buffer_idx", (row_idx, row_cfg.idx))
        new_buffer_cfg = SDIPConfig(
            row_buffer_idx, buffer_cfg.ptr_type, buffer_cfg.memtype, buffer_cfg.name
        )

        return (row_idx, new_buffer_cfg)

    def check_convert_sdip(self, new_cfg: "SDIPConfig") -> "SDIPConfig":
        if self.dims != new_cfg.dims:
            raise ValueError(f"dims {self.dims} != {new_cfg.dims}")
        if self.memtype != new_cfg.memtype:
            raise ValueError(f"memtype {self.memtype} != {new_cfg.memtype}")
        # check if the sizes match
        new_size = new_cfg.dims.size(new_cfg.idx) * get_elem_size(new_cfg.ptr_type)
        old_size = self.dims.size(self.idx) * get_elem_size(self.ptr_type)
        if new_size != old_size:
            raise ValueError(f"size {new_size} != {old_size}")
        return new_cfg

    @overload
    def check_convert(self, new_cfg: "SDIPConfig") -> "SDIPConfig":
        pass

    def check_convert_manual(
        self,
        new_idx: DimName,
        new_ptr_type: Optional[ch.Type] = None,
        new_name: Optional[str] = None,
    ) -> "SDIPConfig":
        if new_ptr_type is None:
            new_ptr_type = self.ptr_type
            if new_name is None:
                new_name = self.name
        elif new_name is None and self.name is not None:
            new_name = self.name + f"_cast_{new_ptr_type}"
        # check if the sizes match
        new_size = self.dims.size(new_idx) * get_elem_size(new_ptr_type)
        old_size = self.dims.size(self.idx) * get_elem_size(self.ptr_type)
        if new_size != old_size:
            raise ValueError(f"size {new_size} != {old_size}")
        return SDIPConfig(self.dims, new_idx, new_ptr_type, self.memtype, new_name)

    @overload
    def check_convert(self, new_idx, new_ptr_type=None, new_name=None) -> "SDIPConfig":
        pass

    def check_convert(self, idx_or_cfg, ptr_type=None, name=None) -> "SDIPConfig":
        if isinstance(idx_or_cfg, SDIPConfig):
            assert ptr_type is None and name is None
            return self.check_convert_sdip(idx_or_cfg)
        elif isinstance(idx_or_cfg, DimName):
            return self.check_convert_manual(idx_or_cfg, ptr_type, name)
        else:
            raise ValueError(f"Invalid type: {type(idx_or_cfg)}")
