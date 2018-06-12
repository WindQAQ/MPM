CC = /usr/local/cuda/bin/nvcc

SRC_DIR = ./src
OBJ_DIR = ./obj
BIN_DIR = .
INC_DIR = ./inc

TARGET = $(BIN_DIR)/test

SRCS = $(wildcard $(SRC_DIR)/*.cu)
OBJS = $(addprefix $(OBJ_DIR)/, $(notdir $(SRCS:.cu=.o)))

CPPFLAGS = -g -std=c++11 -O2 -DGLEW_STATIC --expt-extended-lambda --expt-relaxed-constexpr -DTHRUST_DEBUG -arch=sm_30 -Wno-deprecated-declarations
INCLUDE = -I. -I$(INC_DIR)
LIBS = -lm

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) -o $@ $^ $(LIBS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	$(CC) $(CPPFLAGS) $(INCLUDE) -o $@ --device-c $<

clean:
	$(RM) $(OBJS)

run:
	$(TARGET)
