CC = g++
NVCC = /usr/local/cuda/bin/nvcc

SRC_DIR = ./src
OBJ_DIR = ./obj
BIN_DIR = .
INC_DIR = ./inc

TARGET = $(BIN_DIR)/test

CUDA_SRCS = $(wildcard $(SRC_DIR)/*.cu)
CUDA_OBJS = $(addprefix $(OBJ_DIR)/, $(notdir $(CUDA_SRCS:.cu=.cu.o)))
CPP_SRCS  = $(wildcard $(SRC_DIR)/*.cpp)
CPP_OBJS  = $(addprefix $(OBJ_DIR)/, $(notdir $(CPP_SRCS:.cpp=.o)))

CPPFLAGS = -g -std=c++17 -O2
NVCCFLAGS = -g -std=c++11 -O2 --expt-extended-lambda --expt-relaxed-constexpr -DTHRUST_DEBUG -arch=sm_30 -Wno-deprecated-declarations -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored -rdc=true
INCLUDE = -I. -I$(INC_DIR) -I./third_party $(shell pkg-config --cflags glfw3) -I/usr/include/
LIBS = -lm -Xlinker="$(shell pkg-config --libs glfw3) $(shell pkg-config --static --libs gl)"
# LIBS = -lm $(shell pkg-config --libs glfw3) $(shell pkg-config --static --libs gl)
# CUDALIBS = -L/usr/local/cuda/lib64 -lcuda -lcudart
CUDALIBS =

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(CPP_OBJS) $(CUDA_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LIBS) $(CUDALIBS)

$(OBJ_DIR)/%.cu.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -o $@ --device-c $<

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CC) $(CPPFLAGS) $(INCLUDE) -o $@ -c $< $(CUDALIBS)

clean:
	$(RM) $(CPP_OBJS) $(CUDA_OBJS)

run:
	$(TARGET)
