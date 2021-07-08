CUDA_VERSION ?= 10.1

default: main.cpp
	/usr/local/cuda-$(CUDA_VERSION)/bin/nvcc -g -std=c++14 -lcublas -O2 main.cpp -o main
