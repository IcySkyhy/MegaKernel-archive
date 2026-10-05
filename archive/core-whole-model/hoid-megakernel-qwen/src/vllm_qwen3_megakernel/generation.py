"""Serving generation ownership for decoder, captured graphs and queued outputs."""


class Generation:
    def __init__(self, decoder, number, *, profiling):
        if type(number) is not int or number < 1:
            raise ValueError('positive generation required')
        self.decoder = decoder
        self.number = number
        self.profiling = profiling
        self.manager = None
        self.closed = False

    def own_graph_manager(self, manager):
        """Retain the actual pinned manager before capture, including failure paths."""
        self.decoder._check(self.decoder.stream)
        verify = getattr(self.decoder, 'verify_bindings', None)
        if verify is not None:
            # The full identity scan runs where graphs start to depend on bindings;
            # each replay then costs only the O(1) generation check.
            verify()
        if self.closed or self.manager is not None or manager.graphs:
            raise RuntimeError('generation must own an empty manager before capture')
        self.manager = manager
        # This root prevents even the low-level public close from bypassing the
        # serving owner. The owner removes it only after actual graph reset.
        self.decoder._graphs[id(self)] = self
        if hasattr(manager, 'run_fullgraph'):
            replay = manager.run_fullgraph
            def checked_replay(desc):
                if self.closed or self.manager is not manager:
                    raise RuntimeError('captured serving generation was retired')
                self.decoder._check(self.decoder.stream)
                return replay(desc)
            manager.run_fullgraph = checked_replay

    def retire(self):
        if self.closed:
            raise RuntimeError('generation already retired')
        decoder = self.decoder
        decoder._check(decoder.stream)
        try:
            # Completion/validation is required even for cancelled engine rows.
            for lease in tuple(decoder._deliveries.values()):
                lease.cancel()
            decoder.stream.synchronize()
            self.destroy_graphs()
            decoder.close(all_graphs_destroyed=True)
        except BaseException:
            decoder.poisoned = True
            raise
        self.closed = True

    def destroy_graphs(self):
        decoder = self.decoder
        decoder._check(decoder.stream)
        decoder.stream.synchronize()
        if self.manager is not None:
            try:
                for graph in tuple(self.manager.graphs.values()):
                    graph.reset()
                self.manager.graphs.clear()
            except BaseException:
                decoder.poisoned = True
                raise
            del decoder._graphs[id(self)]
            self.manager = None

    def replace_profiling(self, factory, *, profiling=False):
        """Only startup profiling storage can be replaced; serving epochs persist.

        factory runs after completion and destruction and must construct a fresh
        independently admitted Decoder borrowing the next physical allocation.
        No real-serving generation can silently reset sticky device status.
        """
        if not self.profiling or self.closed:
            raise RuntimeError('only a live profiling generation can be rebound')
        self.retire()
        return Generation(factory(), self.number + 1, profiling=profiling)
