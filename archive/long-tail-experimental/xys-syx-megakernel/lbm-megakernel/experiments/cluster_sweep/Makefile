# Invoke from the repository root. Leaves the original targets and files intact.
include Makefile
.DEFAULT_GOAL := sweep
SWEEP := experiments/cluster_sweep
SWEEP_COMMON := $(wildcard include/* kernels/*) src/grid.cuh src/driver.cuh $(SWEEP)/variants.cuh $(SWEEP)/launch.cuh
.PHONY: sweep sweep-report
sweep: build/sweep/lbm build/sweep/model
build/sweep/driver.cuh: src/driver.cuh $(SWEEP)/prepare_driver.py
	$(PYTHON) $(SWEEP)/prepare_driver.py
build/sweep/lbm: $(SWEEP_COMMON) $(SWEEP)/main.cu build/sweep/driver.cuh build/config.txt $(SWEEP)/Makefile
	$(CUDA_CXX) $(FLAGS) -I$(SWEEP) $(SWEEP)/main.cu -o $@ $(LIBS)
build/sweep/model: $(SWEEP)/model.cpp include/layout_config.h
	@mkdir -p build/sweep
	$(CXX) -O2 -std=c++11 $< -o $@
sweep-report:
	$(PYTHON) $(SWEEP)/report.py
