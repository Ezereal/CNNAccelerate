export OMP_NUM_THREADS=4
g++ -I /opt/intel/mkl/include/ StemConvLayer.cc -lmkl_rt -L/opt/intel/mkl/lib/intel64 -L/opt/intel/lib/intel64 -fopenmp

