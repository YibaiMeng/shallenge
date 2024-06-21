ARCHS=-gencode=arch=compute_80,code=sm_80 \
-gencode=arch=compute_80,code=compute_80 \
-gencode=arch=compute_86,code=sm_86 \
-gencode=arch=compute_86,code=compute_86

NVCC_FLAGS=-O3 $(ARCHS)
NVCC_PROFILE_FLAGS=-O3 $(ARCHS) -lineinfo
BUILD_DIR=build
ROOT_DIR := $(shell pwd)
PROFILE=0
ifeq ($(PROFILE), 1)
NVCC_FLAGS=$(NVCC_PROFILE_FLAGS)
BUILD_DIR=profile
endif

SRCS=$(wildcard src/*.cu) $(wildcard src/*.cpp)
HEADERS=$(wildcard include/*.cuh) $(wildcard include/*.h)

INCLUDE_DIRS=-I$(ROOT_DIR)/include -I$(ROOT_DIR)/third_party/argparse/include

OBJS=$(SRCS:src/%.cu=$(BUILD_DIR)/%.o)

$(BUILD_DIR)/shallenge: $(BUILD_DIR) $(OBJS)
	nvcc $(NVCC_FLAGS) -o $(BUILD_DIR)/shallenge -O3 $(OBJS) $(INCLUDE_DIRS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRCS) $(HEADERS)
	nvcc $(NVCC_FLAGS) -c -o $@ $< $(INCLUDE_DIRS)

ifeq ($(PROFILE), 1)
NCU_COMMAND=ncu -o profile -f --import-source true --set full --target-processes=application-only -k regex:find_lowest_sha256 $(BUILD_DIR)/shallenge --seed 0 --iter 1
else
NCU_COMMAND=@echo "Error: ncu can only be run when DEBUG is set to 1
endif

.PHONY: ncu
ncu: $(BUILD_DIR)/shallenge
	$(NCU_COMMAND)

clean:
	rm -rf $(BUILD_DIR)
