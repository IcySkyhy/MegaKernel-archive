import cheetah.api as ch
from cheetah.index_tools import DimName, Indices, ScopeFn
from .safe_data_index_ptr_utils import SDIPConfig, SafeDataIndexPtr
from .kernel_utils import make_cond_print
from .memory_pipeline import MemoryPipeline
from typing import Sequence, Optional, Tuple


class SmartScopeFn(ScopeFn):

    sdip_configs: list[SDIPConfig]

    def __init__(self, *args, **kwargs):
        """keeps track of the SDIP configs for each SDIP
        ```
        self.sdip_configs = [a for a in args if isinstance(a, SDIPConfig)] + [
            a for a in kwargs.values() if isinstance(a, SDIPConfig)
        ]
        super().__init__(*args, **kwargs)
        ```
        this is be used for easier automated testing.
        """

        super().__init__(*args, **kwargs)
        assert hasattr(self, "sdip_configs")

    def _check_configs(
        self, configs: Sequence[SDIPConfig], sdips: Sequence[SafeDataIndexPtr]
    ) -> SafeDataIndexPtr | tuple[SafeDataIndexPtr, ...]:
        assert len(configs) == len(sdips)
        out = tuple(sc.wrap_sdip(sp) for sc, sp in zip(configs, sdips))
        if len(out) == 1:
            return out[0]
        return out

    def check_configs(self, *sdips: SafeDataIndexPtr):
        return self._check_configs(self.sdip_configs, sdips)

    def __call__(self, *args, **kwargs):
        """checks configs, example below
        self.check_configs(
            [a for a in args if isinstance(a, SafeDataIndexPtr)]
            + [a for a in kwargs.values() if isinstance(a, SafeDataIndexPtr)]
        )
        return super().__call__(*args, **kwargs)
        """
        return super().__call__(*args, **kwargs)


# TODO: implement this
# might benefit to improvements to index tools
# class CleanScopeFn(ScopeFn):


class ProducerConsumerKernel(SmartScopeFn):
    def setup(
        self,
        num_consumer_warps: int,
        num_producer_warps: int,
        warp_idx: DimName,
    ):
        self.warp_idx = warp_idx
        assert self.dims.size(warp_idx) == num_consumer_warps + num_producer_warps
        self.consumer_warp_idx = self.dims.new_dim(
            "consumer_warp_idx", num_consumer_warps
        )
        self.producer_warp_idx = self.dims.new_dim(
            "producer_warp_idx", num_producer_warps
        )

    def generate(self, ix: Indices, *args, **kwargs):
        is_producer = ix[self.warp_idx] < ix.size(self.consumer_warp_idx)
        self.common_generate(ix, *args, **kwargs)
        with ch.if_else(is_producer) as branch:
            with branch.then():
                with ix.scope():
                    ix.set_index(self.consumer_warp_idx, ix[self.warp_idx])
                    self.consume(ix, *args, **kwargs)
            with branch.else_():
                with ix.scope():
                    ix.set_index(
                        self.producer_warp_idx,
                        ix[self.warp_idx] - ix.size(self.consumer_warp_idx),
                    )
                    self.produce(ix, *args, **kwargs)


class ProducerConsumerSmartFn(SmartScopeFn):

    def __init__(self, *args, **kwargs):
        ScopeFn.__init__(self, *args, **kwargs)
        assert hasattr(self, "consumer_sdip_configs")
        assert hasattr(self, "producer_sdip_configs")

    def setup(
        self,
        mem_pipeline: MemoryPipeline,
    ):
        self.consumer_warp_idx = mem_pipeline.consumer_warp_idx
        self.producer_warp_idx = mem_pipeline.producer_warp_idx

    def check_consumer_configs(
        self, *sdips: SafeDataIndexPtr
    ) -> SafeDataIndexPtr | tuple[SafeDataIndexPtr, ...]:
        return self._check_configs(self.consumer_sdip_configs, sdips)

    def check_producer_configs(
        self, *sdips: SafeDataIndexPtr
    ) -> SafeDataIndexPtr | tuple[SafeDataIndexPtr, ...]:
        return self._check_configs(self.producer_sdip_configs, sdips)

    def consume(self, ix: Indices, *args, **kwargs):
        with ix.scope(self.scope):
            self.child_consume(ix, *args, **kwargs)

    def produce(self, ix: Indices, *args, **kwargs):
        with ix.scope(self.scope):
            self.child_produce(ix, *args, **kwargs)

    def generate(self, ix: Indices, *args, **kwargs):
        raise NotImplementedError("PCChildFn should not be used directly")

    def pc_generate(
        self,
        ix: Indices,
        warp_idx: DimName,
        *args,
        reg_split: Optional[Tuple[int, int]] = None,
        **kwargs,
    ):
        is_consumer = ix[warp_idx] < ix.size(self.consumer_warp_idx)
        with ix.scope(self.scope):
            with ch.if_else(is_consumer) as branch:
                with branch.then():
                    if reg_split is not None:
                        ch.asm(f"setmaxnreg.inc.sync.aligned.u32 {reg_split[0]};")
                    ix.set_index(self.consumer_warp_idx, ix[warp_idx])
                    self.child_consume(ix, *args, **kwargs)
                with branch.else_():
                    if reg_split is not None:
                        ch.asm(f"setmaxnreg.dec.sync.aligned.u32 {reg_split[1]};")
                    ix.set_index(
                        self.producer_warp_idx,
                        ix[warp_idx] - ix.size(self.consumer_warp_idx),
                    )
                    self.child_produce(ix, *args, **kwargs)
