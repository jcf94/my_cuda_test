#!/bin/bash

all: matrix.o
	g++ main.cpp matrix.o

clean:
	rm *.o
	rm *.exe