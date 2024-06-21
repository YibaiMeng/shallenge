ARCHS=-arch=compute_86 -code=sm_86
NVCC_FLAGS=-O3 $(ARCHS)

SRCS=shallenge.cu
HEADERS=$(wildcard *.cuh) $(wildcard *.h)
BUILD_DIR=build

INCLUDE_DIRS=-Ithird_party/argparse/include


OBJS=$(SRCS:%.cu=$(BUILD_DIR)/%.o)

all: $(BUILD_DIR) $(OBJS)
	nvcc $(NVCC_FLAGS) -o $(BUILD_DIR)/shallenge -O3 $(OBJS) $(INCLUDE_DIRS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRCS) $(HEADERS)
	nvcc $(NVCC_FLAGS) -c -o $@ $< $(INCLUDE_DIRS)

clean:
	rm -rf $(BUILD_DIR)
