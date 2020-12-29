# CNNAccelerate
This project aims at accelerating convolutional neural networks based on single CPU.
Through the rewrite of tensorflow op, and integrate some optimization methods in it, the inference of model has been improved.
Op that has been rewritten includes im2col, Conv2D, batchnorm.

The acceleration steps include:
1. rewrite high-performance op in tf_ops.
2. compile and generate dynamic link library with g++ compiler.
3. write modify.py to replace the op which will generate new computing graph.
4. load the .so file and import the graph to complete the acceleration of inference.
