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
#include <immintrin.h>
#include "matrix.h"
#include "timer.h"


template<int tokenSize, int hiddenSize, int intermediateSize>
class StemConvLayer {
public:
    StemConvLayer()
    {
    }
    
    ~StemConvLayer()
    {
    }

    void setWeights(const float *_convWeight, const float *_bnGamma, const float *_bnBeta)
    {
        weight.Resize(3 * 3 * hiddenSize, intermediateSize);
        copyConvWeights(weight, _convWeight, 3, 3, hiddenSize);

        gamma.Resize(1, intermediateSize);
        copyWeights(gamma, _bnGamma);
        beta.Resize(1, intermediateSize);
        copyWeights(beta, _bnBeta);
    }
    
    //inputsize (32*32)*3
    Matrix<float>& forward(Matrix<float> &inputBuffer)
    {
        Matrix<float> tmpBuffer;
        tmpBuffer.Resize(tokenSize, 3 * 3 * hiddenSize);
        im2col(inputBuffer, tmpBuffer, 32, 32, 3, 3, 3);

        result.Resize(tokenSize, intermediateSize);
        sgemm(tmpBuffer, weight, result);
        
        batchnorm(result, gamma, beta);
        return result;
    }

private:
    Matrix<float> weight;
    Matrix<float> gamma;
    Matrix<float> beta;
    Matrix<float> result;
    
    
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

    void im2col(Matrix<float> &A, Matrix<float> &B, int img_height, int img_width, \
            int kernel_height, int kernel_width, int kernel_channel)
    {
        assert(A.Cols() == kernel_channel);

        Matrix<float> &output = B;
        output.Resize(tokenSize, kernel_width * kernel_height * kernel_channel);

        int pad_height = kernel_height / 2;
        int pad_width = kernel_width / 2;
        
        Matrix<float> padding;
        int padding_row = (img_height + 2*pad_height) * (img_width + 2*pad_width);
        int padding_col = A.Cols();
        padding.Resize(padding_row, padding_col);
        #pragma omp parallel for
        for(int k = 0; k < A.Cols(); k++)
        {
            for(int i = 0; i < img_height + 2*pad_height; i++)
            {    
                #pragma omp simd
                for(int j = 0; j < img_width + 2*pad_width; j++)
                {
                    if(i < pad_height || i >= img_height + pad_height \
                            || j < pad_width || j >= img_width + pad_width)
                    {
                        padding(i * (img_width + 2*pad_width) + j, k) = 0;
                    }
                    else
                    {
                        padding(i * (img_width + 2*pad_width) + j, k) \
                            = pos2matrix(A, img_width, i - pad_height, j - pad_width, k);
                    }
                }
            }
        }

        int kernel_size = kernel_height * kernel_width;
        #pragma omp parallel for
        for(int i = 0; i < img_height; i++)
        {
            for(int j = 0; j < img_width; j++)
            {
                #pragma omp simd
                for(int k = 0; k < A.Cols(); k++)
                {
                    for(int l = 0; l < kernel_size; l++)
                    {
                        int h = i + l / kernel_width;
                        int w = j + l % kernel_width;
                        *(output.Data() + (i * img_width + j) * A.Cols() * kernel_size + \
                                k * kernel_size + l) 
                            = pos2matrix(padding, img_width + 2*pad_width, h, w, k);
                    }
                }
            }
        }
    }

    void batchnorm(Matrix<float> &A, Matrix<float> &gamma, Matrix<float> &beta)
    {
        assert(A.Cols() == intermediateSize);

        float *pgamma = gamma.Data();
        float *pbeta = beta.Data();
        Matrix<float> &x = A;

        float mean[intermediateSize + 10] = {0};
        float rvariance[intermediateSize + 10] = {0};

        //#pragma omp parallel for
        for (int i = 0; i < tokenSize; i++)
        {
            float *px = x.Row(i);
            #pragma omp simd
            for (int j = 0; j < x.Cols(); j++)
            {
                mean[j] += px[j];
            }
        }

        //#pragma omp parallel for
        for (int i = 0; i < tokenSize; i++)
        {
            float *px = x.Row(i);
            #pragma omp simd
            for (int j = 0; j < x.Cols(); j++)
            {
                if(i == 0)
                {
                    mean[j] /= tokenSize;
                }
                float delta = (px[j] - mean[j]);
                rvariance[j] += delta * delta;
            }
        }
        
        //#pragma omp parallel for
        for (int i = 0; i < tokenSize; i++)
        {
            float *px = x.Row(i);
            #pragma omp simd
            for (int j = 0; j < x.Cols(); j++)
            {
                if(i == 0)
                {
                    float tmp = rvariance[j] / tokenSize + 9.999999960041972e-13;
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
        for (int l = 0; l < m.Cols(); l++) 
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
        for (int i = 0; i < m.Rows(); ++i) 
        {
            if (m.Cols() < 10) 
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
    StemConvLayer<5*5, 5, 128> stem;
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
