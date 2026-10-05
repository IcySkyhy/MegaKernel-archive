# Standalone direct-Clang build. No Enzyme, MLIR, parent tree, or profiler needed.
CUDA_CXX ?= clang++
CUDA_PATH ?= /usr/local/cuda-12.9
SM ?= 120
PYTHON ?= python3
EXTRA_FLAGS ?=
FLAGS := -O3 -ffast-math -std=c++11 -fno-exceptions -DABI -DSIZE=20 \
         -Iinclude -I$(CUDA_PATH)/include --cuda-path=$(CUDA_PATH) \
         --cuda-gpu-arch=sm_$(SM) $(EXTRA_FLAGS)
LIBS := -L$(CUDA_PATH)/lib64 -lcudart_static -ldl -lrt -lpthread -lm
SOURCES := $(wildcard src/* kernels/* include/*)

.PHONY: all check bench report clean FORCE
all: build/lbm
build:
	mkdir -p $@
# Rebuild if the compiler/flags change, without rebuilding on every invocation.
build/config.txt: FORCE | build
	@{ $(CUDA_CXX) --version; printf '%s\n' '$(CUDA_CXX) $(FLAGS) $(LIBS)'; } > $@.tmp
	@cmp -s $@.tmp $@ && rm $@.tmp || mv $@.tmp $@
build/lbm: $(SOURCES) Makefile build/config.txt | build
	$(CUDA_CXX) $(FLAGS) src/main.cu -o $@ $(LIBS)
check: all
	$(PYTHON) scripts/validate.py
bench: all
	$(PYTHON) scripts/benchmark.py
report:
	$(PYTHON) scripts/report.py
clean:
	rm -rf build

# One kernel for the whole time loop (kernels/onekernel.cuh), B0's launch shape and mapping kept.
.PHONY: onekernel
onekernel: build/lbm-onekernel
build/lbm-onekernel: $(SOURCES) Makefile build/config.txt | build
	$(CUDA_CXX) $(FLAGS) src/onekernel_main.cu -o $@ $(LIBS)

# Isolated Z2/DSMEM experiment: preserves the original executable and results.
.PHONY: cluster cluster-check cluster-bench cluster-profile cluster-report cluster-all
cluster: build/lbm-z2 build/z2-ownership build/z2-model
build/lbm-z2: $(SOURCES) build/config.txt Makefile | build
	$(CUDA_CXX) $(FLAGS) src/cluster_main.cu -o $@ $(LIBS)
build/z2-ownership: $(SOURCES) build/config.txt Makefile | build
	$(CUDA_CXX) $(FLAGS) src/cluster_ownership.cu -o $@ $(LIBS)
cluster-check: cluster
	$(PYTHON) experiments/z2/run.py gates
cluster-bench: cluster
	$(PYTHON) experiments/z2/run.py bench
cluster-profile: cluster
	$(PYTHON) experiments/z2/run.py profile
cluster-report:
	$(PYTHON) experiments/z2/report.py
cluster-all: cluster-check
	$(PYTHON) experiments/z2/run.py bench
	$(PYTHON) experiments/z2/run.py profile
	$(PYTHON) experiments/z2/report.py

build/z2-model: experiments/z2/model.cpp include/layout_config.h Makefile | build
	$(CXX) -O2 -std=c++11 $< -o $@
