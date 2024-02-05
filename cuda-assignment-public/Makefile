
all: baseline.x

baseline.x: cuda_prog.cu
	nvcc -lineinfo -Xcompiler -fopenmp $^ -o $@ 

clean:
	rm *.x
