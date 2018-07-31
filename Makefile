#!/bin/bash

CXX = g++
CXXFLAGS = -std=c++11

NVCXX = nvcc
NVCXXFLAGS = -std=c++11 -D GPU_CUDA -Xcompiler "-fopenmp"

all: matrix.o
	$(CXX) $(CXXFLAGS) main.cpp $^

cuda: gmatrix.o matrix.o cuda_test.o
	$(NVCXX) $(NVCXXFLAGS) main.cpp $^

gmatrix.o: gmatrix.cu 
	$(NVCXX) $(NVCXXFLAGS) -c $^ -o $@

cuda_test.o: cuda_test.cu
	$(NVCXX) $(NVCXXFLAGS) -c $^ -o $@

clean:
	rm *.o &
	rm *.exe &
	rm *.out &
