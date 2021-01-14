#ifndef STEMCONV_LAYER_H_
#define STEMCONV_LAYER_H_

#include <new>
#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <mkl.h>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <immintrin.h>
#include "matrix.h"
#include "timer.h"

// 32*32 16*16 108 72 14
template<int tokenSize, int cellTokenSize, int hiddenSize, int intermediateSize, int cellIntermediateSize>
class CellLayer {
public:
    CellLayer()
    {
    }
    
    ~CellLayer()
    {
    }

    void setWeights(const float *_poolxConv, const float *_poolxGamma, const float *_poolxBeta,
        const float *_poolyConv, const float *_poolyGamma, const float *_poolyBeta,
        const float *_layerbaseConv, const float *_layerbaseGamma, const float *_layerbaseBeta,
        const float *_xconv0Depth, const float *_xconv0Point, const float *_xconv0Gamma, const float *_xconv0Beta,
        const float *_xconv1Depth, const float *_xconv1Point, const float *_xconv1Gamma, const float *_xconv1Beta,
        const float *_yconv0Depth, const float *_yconv0Point, const float *_yconv0Gamma, const float *_yconv0Beta,
        const float *_yconv1Depth, const float *_yconv1Point, const float *_yconv1Gamma, const float *_yconv1Beta,
        const float *_final_combineConv1, const float *_final_combineConv2, const float *_final_combineGamma, const float *_final_combineBeta)
    {
        poolx_conv_weight.Resize(1 * 1 * hiddenSize, intermediateSize);
        copyConvWeights(poolx_conv_weight, _poolxConv, 1, 1, hiddenSize);
        poolx_bn_gamma.Resize(1, intermediateSize);
        copyWeights(poolx_bn_gamma, _poolxGamma);
        poolx_bn_beta.Resize(1, intermediateSize);
        copyWeights(poolx_bn_beta, _poolxBeta);

        pooly_conv_weight.Resize(1 * 1 * hiddenSize, intermediateSize);
        copyConvWeights(pooly_conv_weight, _poolyConv, 1, 1, hiddenSize);
        pooly_bn_gamma.Resize(1, intermediateSize);
        copyWeights(pooly_bn_gamma, _poolyGamma);
        pooly_bn_beta.Resize(1, intermediateSize);
        copyWeights(pooly_bn_beta, _poolyBeta);

        layer_base_conv_weight.Resize(1 * 1 * intermediateSize, intermediateSize);
        copyConvWeights(layer_base_conv_weight, _layerbaseConv, 1, 1, intermediateSize);
        layer_base_bn_gamma.Resize(1, intermediateSize);
        copyWeights(layer_base_bn_gamma, _layerbaseGamma);
        layer_base_bn_beta.Resize(1, intermediateSize);
        copyWeights(layer_base_bn_beta, _layerbaseBeta);

        xconv_sep0_depth_weight.Resize(intermediateSize, 5 * 5);
        copyDepthConvWeights(xconv_sep0_depth_weight, _xconv0Depth, 5, 5, intermediateSize);
        xconv_sep0_point_weight.Resize(1 * 1 * intermediateSize, cellIntermediateSize);
        copyConvWeights(xconv_sep0_point_weight, _xconv0Point, 1, 1, intermediateSize);
        xconv_sep0_bn_gamma.Resize(1, cellIntermediateSize);
        copyWeights(xconv_sep0_bn_gamma, _xconv0Gamma);
        xconv_sep0_bn_beta.Resize(1, cellIntermediateSize);
        copyWeights(xconv_sep0_bn_beta, _xconv0Beta);

        xconv_sep1_depth_weight.Resize(cellIntermediateSize, 5 * 5);
        copyDepthConvWeights(xconv_sep1_depth_weight, _xconv1Depth, 5, 5, cellIntermediateSize);
        xconv_sep1_point_weight.Resize(1 * 1 * cellIntermediateSize, cellIntermediateSize);
        copyConvWeights(xconv_sep1_point_weight, _xconv1Point, 1, 1, cellIntermediateSize);
        xconv_sep1_bn_gamma.Resize(1, cellIntermediateSize);
        copyWeights(xconv_sep1_bn_gamma, _xconv1Gamma);
        xconv_sep1_bn_beta.Resize(1, cellIntermediateSize);
        copyWeights(xconv_sep1_bn_beta, _xconv1Beta);

        yconv_sep0_depth_weight.Resize(intermediateSize, 5 * 5);
        copyDepthConvWeights(yconv_sep0_depth_weight, _yconv0Depth, 5, 5, intermediateSize);
        yconv_sep0_point_weight.Resize(1 * 1 * intermediateSize, cellIntermediateSize);
        copyConvWeights(yconv_sep0_point_weight, _yconv0Point, 1, 1, intermediateSize);
        yconv_sep0_bn_gamma.Resize(1, cellIntermediateSize);
        copyWeights(yconv_sep0_bn_gamma, _yconv0Gamma);
        yconv_sep0_bn_beta.Resize(1, cellIntermediateSize);
        copyWeights(yconv_sep0_bn_beta, _yconv0Beta);

        yconv_sep1_depth_weight.Resize(cellIntermediateSize, 5 * 5);
        copyDepthConvWeights(yconv_sep1_depth_weight, _yconv1Depth, 5, 5, cellIntermediateSize);
        yconv_sep1_point_weight.Resize(1 * 1 * cellIntermediateSize, cellIntermediateSize);
        copyConvWeights(yconv_sep1_point_weight, _yconv1Point, 1, 1, cellIntermediateSize);
        yconv_sep1_bn_gamma.Resize(1, cellIntermediateSize);
        copyWeights(yconv_sep1_bn_gamma, _yconv1Gamma);
        yconv_sep1_bn_beta.Resize(1, cellIntermediateSize);
        copyWeights(yconv_sep1_bn_beta, _yconv1Beta);

        final_combine_conv1_weight.Resize(1 * 1 * intermediateSize, intermediateSize / 2);
        copyConvWeights(final_combine_conv1_weight, _final_combineConv1, 1, 1, intermediateSize);
        final_combine_conv2_weight.Resize(1 * 1 * intermediateSize, intermediateSize / 2);
        copyConvWeights(final_combine_conv2_weight, _final_combineConv2, 1, 1, intermediateSize);
        final_combine_bn_gamma.Resize(1, intermediateSize);
        copyWeights(final_combine_bn_gamma, _final_combineGamma);
        final_combine_bn_beta.Resize(1, intermediateSize);
        copyWeights(final_combine_bn_beta, _final_combineBeta);
    }
    
    //inputsize (32*32)*108
    Matrix<float>& forward(Matrix<float> &inputBuffer)
    {
        relu(inputBuffer);
        
        // pool_x
        Matrix<float> poolxBuffer;
        poolxBuffer.Resize(tokenSize, intermediateSize);
        sgemm(inputBuffer, poolx_conv_weight, poolxBuffer);
        batchnorm(poolxBuffer, poolx_bn_gamma, poolx_bn_beta);

        // pool_y
        Matrix<float> poolyBuffer;
        poolyBuffer.Resize(tokenSize, intermediateSize);
        sgemm(inputBuffer, pooly_conv_weight, poolyBuffer);
        batchnorm(poolyBuffer, pooly_bn_gamma, pooly_bn_beta);

        // final_combine
        result.Resize(cellTokenSize, intermediateSize + cellIntermediateSize);
        path_conv(poolxBuffer, result, final_combine_conv1_weight, 32, 32, 2, 2);
        Matrix<float> resultBuffer(result, 0, 256, 36, 36);
        path_conv(poolxBuffer, resultBuffer, final_combine_conv2_weight, 32, 32, 2, 2, true);
        batchnorm(result, final_combine_bn_gamma, final_combine_bn_beta);
        
        // layer_base
        relu(poolyBuffer);
        Matrix<float> layerBaseBuffer;
        layerBaseBuffer.Resize(tokenSize, intermediateSize);
        sgemm(poolyBuffer, layer_base_conv_weight, layerBaseBuffer);
        batchnorm(layerBaseBuffer, layer_base_bn_gamma, layer_base_bn_beta);
        
        // cell xconv yconv
        Matrix<float> cellxconvResult(result, 0, 256, 72, 14);
        cell_conv(layerBaseBuffer, cellxconvResult, xconv_sep0_depth_weight, \
                xconv_sep0_point_weight, xconv_sep0_bn_gamma, xconv_sep0_bn_beta, \
                xconv_sep1_depth_weight, xconv_sep1_point_weight, xconv_sep1_bn_gamma, xconv_sep1_bn_beta);
        
        Matrix<float> cellyconvResult;
        cellyconvResult.Resize(cellTokenSize, cellIntermediateSize);
        cell_conv(layerBaseBuffer, cellyconvResult, yconv_sep0_depth_weight, \
                yconv_sep0_point_weight, yconv_sep0_bn_gamma, yconv_sep0_bn_beta, \
                yconv_sep1_depth_weight, yconv_sep1_point_weight, yconv_sep1_bn_gamma, yconv_sep1_bn_beta);
        matrix_add(cellxconvResult, cellyconvResult);

        relu(result);
        
        return result;
    }

private:
    Matrix<float> poolx_conv_weight;
    Matrix<float> poolx_bn_gamma;
    Matrix<float> poolx_bn_beta;
    Matrix<float> pooly_conv_weight;
    Matrix<float> pooly_bn_gamma;
    Matrix<float> pooly_bn_beta;
    Matrix<float> layer_base_conv_weight;
    Matrix<float> layer_base_bn_gamma;
    Matrix<float> layer_base_bn_beta;
    Matrix<float> xconv_sep0_depth_weight;
    Matrix<float> xconv_sep0_point_weight;
    Matrix<float> xconv_sep0_bn_gamma;
    Matrix<float> xconv_sep0_bn_beta;
    Matrix<float> xconv_sep1_depth_weight;
    Matrix<float> xconv_sep1_point_weight;
    Matrix<float> xconv_sep1_bn_gamma;
    Matrix<float> xconv_sep1_bn_beta;
    Matrix<float> yconv_sep0_depth_weight;
    Matrix<float> yconv_sep0_point_weight;
    Matrix<float> yconv_sep0_bn_gamma;
    Matrix<float> yconv_sep0_bn_beta;
    Matrix<float> yconv_sep1_depth_weight;
    Matrix<float> yconv_sep1_point_weight;
    Matrix<float> yconv_sep1_bn_gamma;
    Matrix<float> yconv_sep1_bn_beta;
    Matrix<float> final_combine_conv1_weight;
    Matrix<float> final_combine_conv2_weight;
    Matrix<float> final_combine_bn_gamma;
    Matrix<float> final_combine_bn_beta;

    Matrix<float> result;
    const float sep_conv_div_value = 1 / 0.9951317;
    
    void cell_conv(Matrix<float> &inputBuffer, Matrix<float> &outputBuffer, \
            Matrix<float> &depth0_weight, Matrix<float> &point0_weight, \
            Matrix<float> &bn_gamma0, Matrix<float> &bn_beta0, \
            Matrix<float> &depth1_weight, Matrix<float> &point1_weight, \
            Matrix<float> &bn_gamma1, Matrix<float> &bn_beta1)
    {
        //cell_0 x_conv sep_conv0
        relu(inputBuffer);
        Matrix<float> tmpBuffer;
        depthwise_im2col(inputBuffer, tmpBuffer, 32, 32, 5, 5, 2, 2);
        Matrix<float> convBuffer;
        convBuffer.Resize(cellTokenSize, intermediateSize);
        depthwiseconv(tmpBuffer, depth0_weight, convBuffer, cellTokenSize, 5 * 5);
        Matrix<float> convResult;
        convResult.Resize(cellTokenSize, cellIntermediateSize);
        sgemm(convBuffer, point0_weight, convResult);
        batchnorm(convResult, bn_gamma0, bn_beta0);

        //cell_0 x_conv sep_conv1
        relu(convResult);
        tmpBuffer.Release();
        depthwise_im2col(convResult, tmpBuffer, 16, 16, 5, 5, 1, 1);
        convBuffer.Release();
        convBuffer.Resize(cellTokenSize, cellIntermediateSize);
        depthwiseconv(tmpBuffer, depth1_weight, convBuffer, cellTokenSize, 5 * 5);
        sgemm(convBuffer, point1_weight, outputBuffer);
        batchnorm(outputBuffer, bn_gamma1, bn_beta1);
        matrix_mul(outputBuffer, sep_conv_div_value);
    }
    
    void path_conv(Matrix<float> &inputBuffer, Matrix<float> &outputBuffer, \
            Matrix<float> &conv_weight, int img_height, int img_width, \
            int stride_height, int stride_width, bool is_strided=false)
    {
        int col = inputBuffer.Cols();
        Matrix<float> poolResult;
        poolResult.Resize(cellTokenSize, col);

        if(is_strided)
        {
            for (int k = 0; k < col; k++)
            {
                for (int i = 0; i < img_height; i += stride_height)
                {
                    for (int j = 0; j < img_width; j += stride_width)
                    {
                        float val = 0;
                        if(i < img_height - 1 && j < img_width - 1)
                        {
                            val = inputBuffer((i+1)*img_width+j+1, k);
                        }
                        int x = i / stride_height;
                        int y = j / stride_width;
                        poolResult(x * img_width / stride_width + y, k) = val;
                    }
                }
            }
        }
        else
        {
            for (int k = 0; k < col; k++)
            {
                for (int i = 0; i < img_height; i += stride_height)
                {
                    for (int j = 0; j < img_width; j += stride_width)
                    {
                        int x = i / stride_height;
                        int y = j / stride_width;
                        poolResult(x * img_width / stride_width + y, k) \
                             = inputBuffer(i * img_width + j, k);
                    }
                }
            }
        }

        sgemm(poolResult, conv_weight, outputBuffer);
    }

    void matrix_add(Matrix<float> &A, Matrix<float> &B)
    {
        int row = A.Rows();
        int col = A.Cols();
        #pragma omp parallel for
        for (int i = 0; i < row; i++)
        {
            #pragma omp simd
            for (int j = 0; j < col; j++)
            {
                A(i, j) += B(i, j);
            }
        }
    }
    
    void matrix_mul(Matrix<float> &A, float val)
    {
        int row = A.Rows();
        int col = A.Cols();
        #pragma omp parallel for
        for (int i = 0; i < row; i++)
        {
            #pragma omp simd
            for (int j = 0; j < col; j++)
            {
                A(i, j) *= val;
            }
        }

    }

    void relu(Matrix<float> &A)
    {
        #pragma omp parallel for
        for (int i = 0; i < A.Rows(); i++)
            for (int j = 0; j < A.Cols(); j++)
            {
                A(i, j) = A(i, j) > 0? A(i, j) : 0;
            }
    }

    void sgemm(Matrix<float> &A, Matrix<float> &B, Matrix<float> &C) 
    {
        bool bTranspose = (A.Cols() != B.Rows());
        int m = A.Rows();
        int k = A.Cols();
        int n = (bTranspose ? B.Rows() : B.Cols());
        float alpha = 1;
        float beta = 0;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, (bTranspose ? CblasTrans : CblasNoTrans), 
                    m, n, k, alpha,
                    A.Data(), A.Stride(), 
                    B.Data(), B.Stride(), beta,
                    C.Data(), C.Stride());
    }
    
    float pos2matrix(Matrix<float> &A, int img_width, int i, int j, int k)
    {
        return *(A.Data() + (i * img_width + j) * A.Cols() + k);
    }

    void depthwise_im2col(Matrix<float> &A, Matrix<float> &B, int img_height, int img_width, \
            int kernel_height, int kernel_width, int stride_height, int stride_width)
    {
        const int pool_height = img_height / stride_height;
        const int pool_width = img_width / stride_width;

        const int kernel_size = kernel_height * kernel_width;
        const int img_size = pool_height * pool_width;
        const int column_size = kernel_size * img_size;

        Matrix<float> &output = B;
        output.Resize(A.Cols(), column_size);

        int pad_left = kernel_width / 2 - (stride_width > 1? 1 : 0);
        int pad_top = kernel_height / 2 - (stride_height > 1? 1 : 0);
        int pad_right = kernel_width / 2;
        int pad_bottom = kernel_height / 2;
        int pad_height = img_height + pad_top + pad_bottom;
        int pad_width = img_width + pad_left + pad_right;

        Matrix<float> padding;
        int padding_row = A.Cols();
        int padding_col = pad_height * pad_width;
        padding.Resize(padding_row, padding_col);
        
        assert (pad_height == pad_width);
        float pad_zeros[pad_height] = {0};

        //#pragma omp parallel for
        for (int k = 0; k < A.Cols(); k++)
        {
            int offset = k * padding_col;
            #pragma omp simd
            for (int i = 0; i < pad_top; i++)
            {
                memcpy(padding.Data() + offset + i * pad_width, pad_zeros, pad_height * sizeof(float)); 
            }
            #pragma omp simd
            for (int i = pad_top; i < pad_height - pad_bottom; i++)
            {
                padding(k, i * pad_width) = 0;
                padding(k, i * pad_width + pad_width - 2) = 0;
                padding(k, i * pad_width + pad_width - 1) = 0;
            }
            #pragma omp simd
            for (int i = pad_height - pad_bottom; i < pad_height; i++)
            {
                memcpy(padding.Data() + offset + i * pad_width, pad_zeros, pad_height * sizeof(float)); 
            }
        }
        for (int i = pad_top; i < pad_height - pad_bottom; i++)
        {
            for(int j = pad_left; j < pad_width - pad_right; j++)
            {    
                int offset = ((i - pad_top) * img_width + j - pad_left) * A.Cols();
                int col_index = i * pad_width + j;
                #pragma omp simd
                for (int k = 0; k < A.Cols(); k++)
                {
                    padding(k, col_index) = *(A.Data() + offset + k);
                }
            }
        }

        assert (kernel_height == kernel_width);
        //#pragma omp parallel for
        for(int k = 0; k < A.Cols(); k++)
        {
            for(int i = 0; i < pool_height; i++)
            {
                #pragma omp simd
                for(int j = 0; j < pool_width; j++)
                {
                    int h = i * stride_height;
                    int w = j * stride_width;
                    int out_offset = k * column_size + (i * pool_width + j) * kernel_size;
                    int pad_offset = k * padding_col + h * pad_width + w;
                    memcpy(output.Data() + out_offset,      padding.Data() + pad_offset, \
                            kernel_height * sizeof(float));
                    memcpy(output.Data() + out_offset + 5,  padding.Data() + pad_offset + pad_width,  \
                            kernel_height * sizeof(float));
                    memcpy(output.Data() + out_offset + 10, padding.Data() + pad_offset + pad_width * 2, \
                            kernel_height * sizeof(float));
                    memcpy(output.Data() + out_offset + 15, padding.Data() + pad_offset + pad_width * 3, \
                            kernel_height * sizeof(float));
                    memcpy(output.Data() + out_offset + 20, padding.Data() + pad_offset + pad_width * 4, \
                            kernel_height * sizeof(float));
                }
            }
        }
    }

    /*void depthwiseconv(Matrix<float> &A, Matrix<float> &B, Matrix<float> &C, int img_size, int kernel_size)
    {
        #define GRP_COUNT 1
        MKL_INT    m[GRP_COUNT] = {img_size};
        MKL_INT    k[GRP_COUNT] = {kernel_size};
        MKL_INT    n[GRP_COUNT] = {1};

        MKL_INT    lda[GRP_COUNT] = {kernel_size};
        MKL_INT    ldb[GRP_COUNT] = {1};
        MKL_INT    ldc[GRP_COUNT] = {1};

        CBLAS_TRANSPOSE    transA[GRP_COUNT] = { CblasNoTrans };
        CBLAS_TRANSPOSE    transB[GRP_COUNT] = { CblasNoTrans };

        const float    alpha[GRP_COUNT] = {1.0};
        const float    beta[GRP_COUNT] = {0.0};

        const MKL_INT    size_per_grp[GRP_COUNT] = {A.Rows()};

        // Total number of multiplications: A.Cols()
        const float *a_array[A.Rows()], *b_array[A.Rows()];
        float *c_array[A.Rows()];
        for (int i = 0; i < A.Rows(); ++i) {
            a_array[i] = A.Data() + i * img_size * kernel_size;
            b_array[i] = B.Data() + i * kernel_size;
        }

        // Call cblas_sgemm_batch
        cblas_sgemm_batch (
                CblasRowMajor,
                transA,
                transB,
                m,
                n,
                k,
                alpha,
                a_array,
                lda,
                b_array,
                ldb,
                beta,
                c_array,
                ldc,
                GRP_COUNT,
                size_per_grp);

        for (int i = 0; i < A.Rows(); i++)
        {
            memcpy(C.Data() + i * img_size, c_array[i], sizeof(float) * img_size);
        }
    }*/

    void depthwiseconv(Matrix<float> &A, Matrix<float> &B, Matrix<float> &C, int img_size, int kernel_size)
    {
        int m = img_size;
        int k = kernel_size;
        int n = 1;
        
        int lda = kernel_size;
        int ldb = 1;
        int ldc = 1;

        float alpha = 1;
        float beta = 0;
        int num = A.Rows();
        
        Matrix<float> transC;
        transC.Resize(num, cellTokenSize);
        for (int i = 0; i < num; i++)
        {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                    m, n, k, alpha,
                    A.Row(i), lda, 
                    B.Row(i), ldb, beta,
                    transC.Row(i), ldc);
        }
        copyTransposed(C, transC); 
    }

    void batchnorm(Matrix<float> &A, Matrix<float> &gamma, Matrix<float> &beta)
    {
        //assert(A.Cols() == intermediateSize);

        float *pgamma = gamma.Data();
        float *pbeta = beta.Data();

        float mean[A.Cols()+ 10] = {0};
        float rvariance[A.Cols()+ 10] = {0};

        //#pragma omp parallel for
        for (int i = 0; i < A.Rows(); i++)
        {
            float *px = A.Row(i);
            #pragma omp simd
            for (int j = 0; j < A.Cols(); j++)
            {
                mean[j] += px[j];
            }
        }

        //#pragma omp parallel for
        for (int i = 0; i < A.Rows(); i++)
        {
            float *px = A.Row(i);
            #pragma omp simd
            for (int j = 0; j < A.Cols(); j++)
            {
                if(i == 0)
                {
                    mean[j] /= A.Rows();
                }
                float delta = (px[j] - mean[j]);
                rvariance[j] += delta * delta;
            }
        }
        
        //#pragma omp parallel for
        for (int i = 0; i < A.Rows(); i++)
        {
            float *px = A.Row(i);
            #pragma omp simd
            for (int j = 0; j < A.Cols(); j++)
            {
                if(i == 0)
                {
                    float tmp = rvariance[j] / A.Rows() + 9.999999960041972e-13;
                    rvariance[j] = 1.0f / sqrt(tmp);
                }
                px[j] = (px[j] - mean[j]) * rvariance[j] * pgamma[j] + pbeta[j];
            }
        }
    }

    void copyConvWeights(Matrix<float> &w, const float *data, \
            int kernel_height, int kernel_width, int kernel_channel)
    {
        for (int i = 0; i < kernel_height; i++)
            for (int j = 0; j < kernel_width; j++)
                for (int k = 0; k < kernel_channel; k++)
                    for (int l = 0; l < w.Cols(); l++)
                    {
                        w(i * kernel_width + j + k * kernel_height * kernel_width, l) = *data++;
                    }
    }

    void copyDepthConvWeights(Matrix<float> &w, const float *data, \
            int kernel_height, int kernel_width, int kernel_channel)
    {
        for (int i = 0; i < kernel_height; i++)
            for (int j = 0; j < kernel_width; j++)
                for (int k = 0; k < kernel_channel; k++)
                {
                    w(k, i * kernel_width + j) = *data++;
                }
    }

    void copyWeights(Matrix<float> &w, const float *data) 
    {
        for (int i = 0; i < w.Rows(); ++i) {
            for (int j = 0; j < w.Cols(); ++j) {
                w(i, j) = *data++;
            }
        }
    }

    void copyTransposed(Matrix<float> &dst, Matrix<float> &src)
    {
        dst.Resize(src.Cols(), src.Rows());
        for(int i = 0; i < src.Rows(); i++)
            for(int j = 0; j < src.Cols(); j++)
            {
                dst(j, i) = src(i, j);
            }
    }

    void dumpImage(Matrix<float> &m, int img_height, int img_width) 
    {
        int cols = m.Cols();
        for (int l = 0; l < std::min(5, m.Cols()); l++) 
        {
            for(int i = 0; i < img_height; i++)
            {
                if (img_width < 10) 
                {
                    for (int j = 0; j < img_width; j++) 
                    {
                        std::cout << m(i * img_width + j, l) << " ";
                    }
                } 
                else 
                {
                    std::cout << m(i*img_width+0, l) << " " << m(i*img_width+1, l) \
                        << " " << m(i*img_width+2, l) << " ... " << m(i*img_width+img_width-3, l) \
                        << " " <<  m(i*img_width+img_width-2, l) << " " <<  m(i*img_width+img_width-1, l);
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    void dumpMatrix(Matrix<float> &m) {
        int cols = m.Cols();
        for (int i = 0; i < std::min(20, m.Rows()); ++i) 
        {
            if (m.Cols() < 2000000) 
            {
                for (int j = 0; j < m.Cols(); ++j) 
                {
                    std::cout << m(i, j) << " ";
                }
            } 
            else 
            {
                std::cout << m(i, 0) << " " << m(i, 1) << " " << m(i, 2) << " ... " << m(i, cols-3) << " " <<  m(i, cols-2) << " " <<  m(i, cols-1);
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

};

/*int main()
{
    CellLayer<5*5, 5, 128> stem;
    Matrix<float> A, B;
    A.Resize(5*5, 5);
    for(int i = 0; i < 5*5*5; i++)
        *(A.Data() + i) = i;
    stem.im2col(A, B, 5, 5, 3, 3, 5);
    stem.dumpImage(A, 5, 5);
    stem.dumpMatrix(B);

    return 0;
}*/

#endif
