
mandelbrot : mandelbrot.o png_util.o 
	gcc -O3 -fopenmp -o mandelbrot mandelbrot.o png_util.o -lpng -lm

mandelbrot.o : mandelbrot.c png_util.h
	gcc -O3 -I. -fopenmp -c mandelbrot.c

png_util.o : png_util.c png_util.h
	gcc -O3 -I. -c png_util.c

clean :
	rm mandelbrot.o png_util.o mandelbrot
