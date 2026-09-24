from .safe_data_index_ptr_utils import SDIPConfig, SafeDataIndexPtr
from dataclasses import dataclass
import cheetah.api as ch
from cheetah.index_tools import DimName, Dims, Indices
from cheetah.api import ty
from .cuda_utils import get_elem_size


@dataclass
class QuantConfig:
    dims: Dims
    packed_dtype: ch.Type = ty.i32
    bits: int = 4
    buffer_dtype: ch.Type = ty.bf16

    def __post_init__(self):
        assert self.packed_dtype == ty.i32
        assert self.buffer_dtype == ty.bf16
        assert self.bits == 4

        self.pack_to_buffer_idx = self.dims.new_dim_with_eq(
            "pack_to_buffer_idx",
            get_elem_size(self.packed_dtype) // get_elem_size(self.buffer_dtype),
        )
        self.buffer_to_elem_idx = self.dims.new_dim_with_eq(
            "buffer_to_elem_idx",
            get_elem_size(self.buffer_dtype) * 8 // self.bits,
        )
        self.pack_to_elem_idx = self.dims.new_dim_with_eq(
            "pack_to_elem_idx",
            (self.pack_to_buffer_idx, self.buffer_to_elem_idx),
        )

    def get_quant_buffer_dim(self, orig_idx: DimName):
        quant_idx = self.dims.new_dim(f"{orig_idx}_buffer_quant")
        self.dims.eq(orig_idx, (quant_idx, self.buffer_to_elem_idx))
        return quant_idx

    def get_quant_pack_dim(self, orig_idx: DimName):
        quant_idx = self.dims.new_dim(f"{orig_idx}_pack_quant")
        self.dims.eq(quant_idx, (orig_idx, self.pack_to_buffer_idx))
        return quant_idx


class I4QuantizedSafeDataIndexPtr(SafeDataIndexPtr):

    def __init__(
        self,
        ix,
        ptr,
        quant_ptr_idx,
        memtype,
        name,
        quant_config: QuantConfig,
    ):
        super().__init__(ix, ptr, quant_ptr_idx, memtype, name)
        self.quant_config = quant_config

    def load_and_dequantize(self, ix, out_arr: SafeDataIndexPtr):
        assert out_arr.type() == ty.ptr_mut(ty.bf16)
        inp_as_i32 = ch.alloc(ty.i32)
        inp_as_i32.val = self[self._ptr_idx]
        out_as_i32 = out_arr.cast(
            new_ptr_idx=self.quant_config.buffer_to_elem_idx,
            new_ptr_ty=ty.ptr_mut(ty.i32),
            new_name=f"{out_arr.name}_i32",
        )

        with ch.for_(ix.size(self.quant_config.buffer_to_elem_idx)) as i:
            with ix.scope():
                out_as_i32._ptr[i] = ch.asm(
                    "lop3.b32 $out, $inp, $mask, $magic_num, (0xf0 & 0xcc) | 0xaa;",
                    out=ty.i32,
                    inp=inp_as_i32.val,
                    mask=ch.raw_expr("0x000f000f", ty.i32),
                    magic_num=ch.raw_expr("0x43004300", ty.i32),
                )["out"]
                with ch.if_(i != 3):
                    inp_as_i32.val = inp_as_i32.val >> 4
        with ix.scope():
            with ch.for_(ix.size(self.quant_config.buffer_to_elem_idx)) as i:
                out_as_i32._ptr[i] = ch.asm(
                    "fma.rn.bf16x2 $out, $inp, $scale, $bias;",
                    out=ty.i32,
                    inp=out_as_i32._ptr[i],
                    scale=ch.raw_expr("0x3f803f80", ty.i32),
                    bias=ch.raw_expr("0xC300C300", ty.i32),
                )["out"]

    def check_convert(self, new_ptr_idx, memtype, ptr_type):
        assert self.ix.size(new_ptr_idx) == self.ix.size(
            self._ptr_idx
        ), f"check_convert {new_ptr_idx} size {self.ix.size(new_ptr_idx)} != {self._ptr_idx} size{self.ix.size(self._ptr_idx)}"
        assert self.memtype == memtype
        if ptr_type is not None:
            self.check_type(ty.ptr_mut(ptr_type))
        return I4QuantizedSafeDataIndexPtr(
            self.ix,
            self._ptr,
            new_ptr_idx,
            memtype,
            self.name,
            self.quant_config,
        )


@dataclass
class I4QuantSDIPConfig(SDIPConfig):
    dims: Dims
    idx: DimName
    memtype: str
    name: str
    type: ch.Type
    quant_config: QuantConfig

    def wrap_ptr(self, ix: Indices, ptr: ch.Expr) -> I4QuantizedSafeDataIndexPtr:
        return I4QuantizedSafeDataIndexPtr(
            ix, ptr, self.idx, self.memtype, self.name, self.quant_config
        )

    def wrap_sdip(
        self, ix: Indices, sdip: SafeDataIndexPtr
    ) -> I4QuantizedSafeDataIndexPtr:
        return self.check_convert(self.idx, self.memtype, self.type)

    @staticmethod
    def fill_buffer(
        buffer_cfg: SDIPConfig, row_cfg: DimName, quant_config: QuantConfig
    ) -> tuple[DimName, SDIPConfig]:
        raise NotImplementedError("Not implemented")
