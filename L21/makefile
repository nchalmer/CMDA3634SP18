
mandelbrot : mandelbrot.o png_util.o 
	nvcc -arch=sm_60 -o mandelbrot mandelbrot.o png_util.o -lpng -lm

mandelbrot.o : mandelbrot.cu png_util.h
	nvcc -arch=sm_60 -I. -c mandelbrot.cu

png_util.o : png_util.c png_util.h
	nvcc -arch=sm_60 -I. -c png_util.c

clean :
	rm mandelbrot.o png_util.o mandelbrot
