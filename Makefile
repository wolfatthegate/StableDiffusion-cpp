# StableDiffusion-cpp build.
#   make              build ./sd and ./sd-server
#   make test         build and run the kernel tests (no weights needed)
#   make test_tokenizer && ./test_tokenizer   tokenizer test (needs models/sd15)
#   make ACCELERATE=0 use the built-in GEMM instead of Apple's BLAS (macOS defaults to 1)
#   make clean
# Every source in src/ except main.cpp and server.cpp is shared by the
# binaries and the tests.

CXX      ?= clang++
CXXFLAGS ?= -std=c++17 -O3 -Wall -Wextra -pthread
LDFLAGS  ?=

# -march=native on x86 enables AVX/FMA; on arm64 NEON is always available.
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),x86_64)
  CXXFLAGS += -march=native
endif

# Apple's BLAS (Accelerate) is 2.6x faster than the built-in GEMM on Apple
# Silicon for the 6B Z-Image transformer, so it is the default on macOS.
# Pass ACCELERATE=0 to use the dependency-free kernel.
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  ACCELERATE ?= 1
endif
ifeq ($(ACCELERATE),1)
  CXXFLAGS += -DUSE_ACCELERATE
  LDFLAGS  += -framework Accelerate
endif

LIB_SRCS := $(filter-out src/main.cpp src/server.cpp,$(wildcard src/*.cpp))
LIB_OBJS := $(LIB_SRCS:src/%.cpp=build/%.o)
HEADERS  := $(wildcard src/*.h)

all: sd sd-server

sd: $(LIB_OBJS) build/main.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

sd-server: $(LIB_OBJS) build/server.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

test_ops: $(LIB_OBJS) build/test_ops.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

test: test_ops
	./test_ops

# Every object depends on every header: simple and always correct for a
# project this size.
build/%.o: src/%.cpp $(HEADERS) | build
	$(CXX) $(CXXFLAGS) -c -o $@ $<

build/test_ops.o: tests/test_ops.cpp $(HEADERS) | build
	$(CXX) $(CXXFLAGS) -Isrc -c -o $@ $<

build:
	mkdir -p build

clean:
	rm -rf build sd sd-server test_ops test_tokenizer test_qwen_tokenizer test_qwen3 test_vae test_zimage

.PHONY: all test clean

test_tokenizer: $(LIB_OBJS) build/test_tokenizer.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

build/test_tokenizer.o: tests/test_tokenizer.cpp $(HEADERS) | build
	$(CXX) $(CXXFLAGS) -Isrc -c -o $@ $<

test_qwen_tokenizer: $(LIB_OBJS) build/test_qwen_tokenizer.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

build/test_qwen_tokenizer.o: tests/test_qwen_tokenizer.cpp $(HEADERS) | build
	$(CXX) $(CXXFLAGS) -Isrc -c -o $@ $<

test_qwen3: $(LIB_OBJS) build/test_qwen3.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

build/test_qwen3.o: tests/test_qwen3.cpp $(HEADERS) | build
	$(CXX) $(CXXFLAGS) -Isrc -c -o $@ $<

test_vae: $(LIB_OBJS) build/test_vae.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

build/test_vae.o: tests/test_vae.cpp $(HEADERS) | build
	$(CXX) $(CXXFLAGS) -Isrc -c -o $@ $<

test_zimage: $(LIB_OBJS) build/test_zimage.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

build/test_zimage.o: tests/test_zimage.cpp $(HEADERS) | build
	$(CXX) $(CXXFLAGS) -Isrc -c -o $@ $<
