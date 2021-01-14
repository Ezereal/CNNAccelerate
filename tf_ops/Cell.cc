#include <algorithm>
#include <vector>
#include <cmath>
 
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/kernels/bounds_check.h"
 
#include "time.h"
#include "matrix.h"
#include "CellLayer.h"
namespace tf = tensorflow;
using tf::shape_inference::DimensionHandle;
using tf::shape_inference::InferenceContext;
using tf::shape_inference::ShapeHandle;
 
using namespace tensorflow;


REGISTER_OP("Cell")    
    .Input("input: float32")
    .Input("poolx_conv_weight: float32")
    .Input("poolx_bn_gamma: float32")
    .Input("poolx_bn_beta: float32")
    .Input("pooly_conv_weight: float32")
    .Input("pooly_bn_gamma: float32")
    .Input("pooly_bn_beta: float32")
    .Input("layer_base_conv_weight: float32")
    .Input("layer_base_bn_gamma: float32")
    .Input("layer_base_bn_beta: float32")
    .Input("xconv_sep0_depth_weight: float32")
    .Input("xconv_sep0_point_weight: float32")
    .Input("xconv_sep0_bn_gamma: float32")
    .Input("xconv_sep0_bn_beta: float32")
    .Input("xconv_sep1_depth_weight: float32")
    .Input("xconv_sep1_point_weight: float32")
    .Input("xconv_sep1_bn_gamma: float32")
    .Input("xconv_sep1_bn_beta: float32")
    .Input("yconv_sep0_depth_weight: float32")
    .Input("yconv_sep0_point_weight: float32")
    .Input("yconv_sep0_bn_gamma: float32")
    .Input("yconv_sep0_bn_beta: float32")
    .Input("yconv_sep1_depth_weight: float32")
    .Input("yconv_sep1_point_weight: float32")
    .Input("yconv_sep1_bn_gamma: float32")
    .Input("yconv_sep1_bn_beta: float32")
    .Input("final_combine_conv1_weight: float32")
    .Input("final_combine_conv2_weight: float32")
    .Input("final_combine_bn_gamma: float32")
    .Input("final_combine_bn_beta: float32")
    .Output("custom_output: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        //c->set_output(0, c->input(0));
        return Status::OK();
      });


class CellOp : public OpKernel {
    public:
    explicit CellOp(OpKernelConstruction* context) : OpKernel(context) 
    {
        cell_layer = nullptr;
        initialized = false;
    }
    
    ~CellOp()
    {
        delete cell_layer;
    }

    void Compute(OpKernelContext* context) override 
    {
        // 获取输入 tensor
        const Tensor& input_tensor = context->input(0);
        float* input = (float*)input_tensor.tensor_data().data();
        
        if(!initialized)
        {
            cell_layer = new CellLayer<tokenSize, cellTokenSize, hiddenSize, intermediateSize, cellIntermediateSize>();
            initWeights(context);
            initialized = true;
        }

        //TODO:transpose inputbuffer
        Matrix<float> inputBuffer;
        inputBuffer.Resize(32*32, 108);
        copyInputs(inputBuffer, input, 32, 32);

        // 执行计算操作。
        Matrix<float> &out = cell_layer->forward(inputBuffer); 

        // 创建输出 tensor, context->allocate_output 用来分配输出内存？
        Tensor* output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({1, 16, 16, 86}),
                                                         &output_tensor));
        float* output = (float*)output_tensor->tensor_data().data();
    
        memcpy(output, out.Data(), 16 * 16 * 86 * sizeof(float));
    }

private:
    void initWeights(OpKernelContext* context)
    {
        int index = 1;
        for(int i = 0; i < 1; i++)
        {
            float *pData[29];
            for(int j = 0; j < 29; j++)
            {
                pData[j] = (float*)context->input(index++).tensor_data().data();
            }
            cell_layer->setWeights(pData[0], pData[1], pData[2], pData[3], pData[4],
                pData[5], pData[6], pData[7], pData[8], pData[9], pData[10], 
                pData[11], pData[12], pData[13], pData[14], pData[15], pData[16], 
                pData[17], pData[18], pData[19], pData[20], pData[21], pData[22], 
                pData[23], pData[24], pData[25], pData[26], pData[27], pData[28]);
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
    static const int cellTokenSize = 16 * 16;
    static const int hiddenSize = 108;
    static const int intermediateSize = 72;
    static const int cellIntermediateSize = 14;

    CellLayer<tokenSize, cellTokenSize, hiddenSize, intermediateSize, cellIntermediateSize> *cell_layer;
    bool initialized;
};

REGISTER_KERNEL_BUILDER(Name("Cell").Device(DEVICE_CPU), CellOp);

