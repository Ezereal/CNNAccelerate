# CNNAccelerate
This project aims at accelerating convolutional neural networks based on single CPU.
Through the rewrite of tensorflow op, and integrate some optimization methods in it, the inference of model has been improved.
Op that has been rewritten includes relu, im2col, Conv2D, batchnorm, depthwise_im2col, depthwiseconv, pointwiseconv.

The acceleration steps include:
1. rewrite high-performance op in tf_ops.
2. compile and generate dynamic link library with g++ compiler.
3. write modify.py to replace the op which will generate new computing graph.
4. load the .so file and import the graph to complete the acceleration of inference.


# Result
This model contains two module: StemConv and Cell.
StemConv module run by tensorflow consumes 0.865ms, while it consumes only 0.407ms accelerated by this project.
Cell module run by tensorflow consumes 2.94ms, while it consumes only 1.829ms accelerated by this project.
Total model run by tensorflow consumes 3.83ms, while it consumes only 2.35ms accelerated by this project which 1.63x faster.
It should be noted that this acceleration would lead 1e-2 bias, which could improved by using 'double' datatype.


# Environment
single cpu (omp_num_threads = 1, inter_op_parallelism_threads = intra_op_parallelism_threads = 1)
CPU: Intel(R) Core(TM) i5-7500 CPU @ 3.40GHz
