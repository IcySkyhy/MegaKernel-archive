from __future__ import annotations

from typing import Any

import torch


def capture_cuda_graph(
    *,
    device: torch.device,
    engine_stream: torch.cuda.Stream,
    comm_stream: torch.cuda.Stream,
    extra_streams: tuple[torch.cuda.Stream, ...] = (),
    overlap_comm: bool,
    pool,
    fn,
) -> tuple[torch.cuda.CUDAGraph, Any]:
    streams = [comm_stream]
    for stream in extra_streams:
        if all(stream is not existing for existing in streams):
            streams.append(stream)

    torch.cuda.synchronize(device)
    if overlap_comm:
        for stream in streams:
            stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(engine_stream):
        with torch.cuda.graph(graph, pool=pool, stream=engine_stream):
            if overlap_comm:
                for stream in streams:
                    stream.wait_stream(engine_stream)
            fn()
            if overlap_comm:
                for stream in streams:
                    engine_stream.wait_stream(stream)

    engine_stream.synchronize()
    if overlap_comm:
        for stream in streams:
            stream.synchronize()
    return graph, graph.pool() if pool is None else pool
