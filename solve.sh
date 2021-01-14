export OMP_NUM_THREADS=1
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
g++ -std=c++11 -shared -I /opt/intel/mkl/include/ -lmkl_rt -L/opt/intel/mkl/lib/intel64 -L/opt/intel/lib/intel64 -fopenmp tf_ops/StemConv.cc -o StemConv.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
g++ -std=c++11 -shared -I /opt/intel/mkl/include/ -lmkl_rt -L/opt/intel/mkl/lib/intel64 -L/opt/intel/lib/intel64 -fopenmp tf_ops/Cell.cc -o Cell.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
python inference.py
