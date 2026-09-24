# Compiler and Python toolchain.  PYTHON_CONFIG is derived from PYTHON so a
# virtual environment cannot silently mix one interpreter's pybind11 headers
# with another interpreter's extension suffix or linker flags.
NVCC ?= nvcc
PYTHON ?= python3
.DEFAULT_GOAL := all

GPU ?= B200

PYTHON_VERSION := $(shell $(PYTHON) -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_CONFIG := $(shell $(PYTHON) -c 'import pathlib, sys, sysconfig; print(pathlib.Path(sysconfig.get_config_var("BINDIR")) / f"python{sys.version_info.major}.{sys.version_info.minor}-config")')
EXT_SUFFIX := $(shell $(PYTHON) -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
PYTHON_INCLUDES := $(shell $(PYTHON) -m pybind11 --includes)
PYTHON_LDFLAGS := $(shell $(PYTHON_CONFIG) --ldflags)
TARGETS = mk_llama$(EXT_SUFFIX) mk_mlp_simt$(EXT_SUFFIX)
DEPFILES = $(addsuffix .d,$(TARGETS))
BUILD_CONFIG = .build-config

PERSISTENT_VM ?= 0

NVCCFLAGS=-DNDEBUG -Xcompiler=-fPIE --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing --use_fast_math -forward-unknown-to-host-compiler -O3 -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills -std=c++20 -x cu -lrt -lpthread -ldl -lcuda -lcudadevrt -lcudart_static -lcublas -lineinfo
NVCCFLAGS+= -I${THUNDERKITTENS_ROOT}/include -I${MEGAKERNELS_ROOT}/include $(PYTHON_INCLUDES) $(PYTHON_LDFLAGS) -shared -fPIC -lpython$(PYTHON_VERSION)

ifeq ($(PERSISTENT_VM),1)
NVCCFLAGS+= -DENABLE_PERSISTENT_VM
endif

# Rebuild when a command-line build choice changes. Header dependencies are
# emitted by nvcc below, so editing constexpr configuration cannot silently
# reuse an old extension either.
BUILD_CONFIG_TEXT = NVCC=$(NVCC);GPU=$(GPU);PERSISTENT_VM=$(PERSISTENT_VM);PYTHON=$(PYTHON);PYTHON_VERSION=$(PYTHON_VERSION);NVCCFLAGS=$(NVCCFLAGS)

.PHONY: FORCE
FORCE:

$(BUILD_CONFIG): FORCE
	@printf '%s\n' '$(BUILD_CONFIG_TEXT)' | cmp -s - $@ || \
		printf '%s\n' '$(BUILD_CONFIG_TEXT)' > $@

# Conditional setup based on the target GPU
ifeq ($(GPU),4090)
NVCCFLAGS+= -DKITTENS_4090 -arch=sm_89
else ifeq ($(GPU),A100)
NVCCFLAGS+= -DKITTENS_A100 -arch=sm_80
else ifeq ($(GPU),H100)
NVCCFLAGS+= -DKITTENS_HOPPER -arch=sm_90a
else
NVCCFLAGS+= -DKITTENS_HOPPER -DKITTENS_BLACKWELL -arch=sm_100a
endif

# Default target
all: $(TARGETS)

# run: $(TARGET)
# 	python test.py

mk_llama$(EXT_SUFFIX): llama.cu Makefile $(BUILD_CONFIG)
	$(NVCC) llama.cu $(NVCCFLAGS) -MMD -MP -MF $@.d -o $@

mk_mlp_simt$(EXT_SUFFIX): mlp_simt_module.cu Makefile $(BUILD_CONFIG)
	$(NVCC) mlp_simt_module.cu $(NVCCFLAGS) -MMD -MP -MF $@.d -o $@

-include $(DEPFILES)

# Clean target
clean:
	rm -f $(TARGETS) $(DEPFILES) $(BUILD_CONFIG)
