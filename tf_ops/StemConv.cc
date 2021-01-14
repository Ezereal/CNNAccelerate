#include <algorithm>
#include <vector>
#include <cmath>
 
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/kernels/bounds_check.h"
 
#include "time.h"
#include "matrix.h"
#include "StemConvLayer.h"
namespace tf = tensorflow;
using tf::shape_inference::DimensionHandle;
using tf::shape_inference::InferenceContext;
using tf::shape_inference::ShapeHandle;
 
using namespace tensorflow;


REGISTER_OP("StemConv")    
    .Input("input: float32")
    .Input("conv_weight: float32")
    .Input("bn_gamma: float32")
    .Input("bn_beta: float32")
    .Output("custom_output: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        //c->set_output(0, c->input(0));
        return Status::OK();
      });


class StemConvOp : public OpKernel {
    public:
    explicit StemConvOp(OpKernelConstruction* context) : OpKernel(context) 
    {
        stemconv_layer = nullptr;
        initialized = false;
    }
    
    ~StemConvOp()
    {
        delete stemconv_layer;
    }

    void Compute(OpKernelContext* context) override 
    {
        // 获取输入 tensor
        const Tensor& input_tensor = context->input(0);
        float* input = (float*)input_tensor.tensor_data().data();
        
        if(!initialized)
        {
            stemconv_layer = new StemConvLayer<tokenSize, hiddenSize, intermediateSize>();
            initWeights(context);
            initialized = true;
        }

        //TODO:transpose inputbuffer
        Matrix<float> inputBuffer;
        inputBuffer.Resize(32*32, 3);
        copyInputs(inputBuffer, input, 32, 32);

        // 执行计算操作。
        Matrix<float> &out = stemconv_layer->forward(inputBuffer); 

        // 创建输出 tensor, context->allocate_output 用来分配输出内存？
        Tensor* output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({1, 32, 32, 108}),
                                                         &output_tensor));
        float* output = (float*)output_tensor->tensor_data().data();
    
        memcpy(output, out.Data(), 32 * 32 * 108 * sizeof(float));
    
    }

private:
    void initWeights(OpKernelContext* context)
    {
        int index = 1;
        for(int i = 0; i < 1; i++)
        {
            float *pData[3];
            for(int j = 0; j < 3; j++)
            {
                pData[j] = (float*)context->input(index++).tensor_data().data();
            }
            stemconv_layer->setWeights(pData[0], pData[1], pData[2]);
        }
    }

    void copyInputs(Matrix<float> &w, const float *data, int height, int width)
    {
        for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++)
                for (int k = 0; k < w.Cols(); k++)
                {
                    w(i * width + j, k) = *data++;
                }
    }

    static const int tokenSize = 32 * 32;
    static const int hiddenSize = 3;
    static const int intermediateSize = 108;

    StemConvLayer<tokenSize, hiddenSize, intermediateSize> *stemconv_layer;
    bool initialized;
};

REGISTER_KERNEL_BUILDER(Name("StemConv").Device(DEVICE_CPU), StemConvOp);

