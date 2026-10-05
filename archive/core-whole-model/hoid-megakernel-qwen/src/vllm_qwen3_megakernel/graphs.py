"""Explicit ownership of captured graphs on a single workspace stream."""


class OwnedGraph:
    def __init__(self, workspace, record, capture_stream):
        import torch

        workspace._check(workspace.stream)
        self.workspace = workspace
        self.closed = False
        self._graph = torch.cuda.CUDAGraph()
        workspace._graphs[id(self)] = self
        try:
            capture_stream.wait_stream(workspace.stream)
            with torch.cuda.graph(self._graph, stream=capture_stream):
                record(capture_stream)
            workspace.stream.wait_stream(capture_stream)
        except BaseException as error:
            try:
                capture_stream.synchronize()
                self._graph.reset()
            except BaseException as cleanup_error:
                workspace.poisoned = True
                error.add_note(f'failed capture remains retained: {cleanup_error}')
            else:
                self.closed = True
                del workspace._graphs[id(self)]
            raise

    def replay(self):
        if self.closed:
            raise RuntimeError('graph is retired')
        self.workspace._check(self.workspace.stream)
        self._graph.replay()

    def close(self):
        if self.closed:
            raise RuntimeError('graph is already retired')
        self.workspace._check(self.workspace.stream)
        self.workspace.stream.synchronize()
        self._graph.reset()
        self.closed = True
        del self.workspace._graphs[id(self)]
