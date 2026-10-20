CXX      ?= clang++
CXXFLAGS ?= -std=c++17 -O3 -Wall -Wextra -pthread
LDFLAGS  ?=

UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),x86_64)
  CXXFLAGS += -march=native
endif

# make ACCELERATE=1 -> use Apple's BLAS for the GEMM instead of the built-in kernel
ifeq ($(ACCELERATE),1)
  CXXFLAGS += -DUSE_ACCELERATE
  LDFLAGS  += -framework Accelerate
endif

LIB_SRCS := $(filter-out src/main.cpp,$(wildcard src/*.cpp))
LIB_OBJS := $(LIB_SRCS:src/%.cpp=build/%.o)
HEADERS  := $(wildcard src/*.h)

all: sd

sd: $(LIB_OBJS) build/main.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

test_ops: $(LIB_OBJS) build/test_ops.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

test: test_ops
	./test_ops

build/%.o: src/%.cpp $(HEADERS) | build
	$(CXX) $(CXXFLAGS) -c -o $@ $<

build/test_ops.o: tests/test_ops.cpp $(HEADERS) | build
	$(CXX) $(CXXFLAGS) -Isrc -c -o $@ $<

build:
	mkdir -p build

clean:
	rm -rf build sd test_ops test_tokenizer

.PHONY: all test clean

test_tokenizer: $(LIB_OBJS) build/test_tokenizer.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

build/test_tokenizer.o: tests/test_tokenizer.cpp $(HEADERS) | build
	$(CXX) $(CXXFLAGS) -Isrc -c -o $@ $<
