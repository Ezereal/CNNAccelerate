#ifndef __MATRIX_H_
#define __MATRIX_H_
#include <assert.h>



template <typename T>
class Matrix
{
private:
    bool shadow_;
    int rows_;
    int cols_;
    int stride_;
    T* data_;
    int alloc_size_;

    //Matrix& operator=(const Matrix &matrix);

public:
    Matrix()
    {
        this->shadow_ = false;
        this->rows_ = 0;
        this->cols_ = 0;
        this->stride_ = 0;
        this->data_ = nullptr;
        this->alloc_size_ = 0;
    }

    Matrix(Matrix &matrix, int start_row, int rows, int start_col, int cols)
    {
        this->shadow_ = true;
        this->rows_ = rows;
        this->cols_ = cols;
        this->stride_ = matrix.stride_;
        this->data_ = matrix.data_ + start_row * matrix.stride_ + start_col;
        this->alloc_size_ = 0;
    }

    Matrix(Matrix &matrix)
    {
        this->shadow_ = true;
        this->rows_ = matrix.rows_;
        this->cols_ = matrix.cols_;
        this->stride_ = matrix.stride_;
        this->data_ = matrix.data_;
        this->alloc_size_ = 0;
    }

    Matrix(Matrix &matrix, int start_row, int rows)
    {
        this->shadow_ = true;
        this->rows_ = rows;
        this->cols_ = matrix.cols_;
        this->stride_ = matrix.stride_;
        this->data_ = matrix.data_ + start_row * matrix.stride_;
        this->alloc_size_ = 0;
    }

    Matrix(T *data, int rows, int cols, int stride)
    {
        this->shadow_ = true;
        this->rows_ = rows;
        this->cols_ = cols;
        this->stride_ = stride;
        this->data_ = data;
        this->alloc_size_ = 0;
    }

    ~Matrix()
    {
        this->Release();
    }

    void Resize(int rows, int cols)
    {
        assert(!shadow_);
        if(rows == rows_ && cols == cols_)
        {
            return;
        }
        if(rows < 0 && cols < 0)
        {
            return;
        }
        if(rows == 0 || cols == 0)
        {
            this->Release();
            return;
        }
        /*if(cols > 16)
        {
            int skip = (16 - cols % 16) % 16;
            stride_ = cols + skip;
            if(stride_ % 256 == 0)
            {
                stride_ += 4;
            }
        }
        else
        {
            stride_ = cols;
        }*/
        stride_ = cols;
        rows_ = rows;
        cols_ = cols;
        if(alloc_size_ >= stride_ * rows)
        {
            return;
        }
        else
        {
            if(data_)
            {
                free(data_);
            }
            alloc_size_ = stride_ * rows;
            data_ = (T *)aligned_alloc(64, sizeof(T) * alloc_size_);
            if(data_ == nullptr)
            {
                throw std::bad_alloc();
            }
        }
    }

    T* Data()
    {
        return data_;
    }

    const T* Data() const
    {
        return data_;
    }

    void Release()
    {
        if(!shadow_ && data_)
        {
            free(data_);
            data_ = nullptr;
        }
        rows_ = 0;
        cols_ = 0;
        stride_ = 0;
        alloc_size_ = 0;
    }

    int Rows() const
    {
        return rows_;
    }

    int Cols() const
    {
        return cols_;
    }

    int Stride() const
    {
        return stride_;
    }

    T* Row(const int idx)
    {
        return data_ + stride_ * idx;
    }

    const T* Row(const int idx) const
    {
        return data_ + stride_ * idx;
    }

    T& operator()(int r, int c)
    {
        return *(data_ + r * stride_ + c);
    }

    Matrix& operator=(const Matrix& matrix)
    {

        this->shadow_ = true;
        this->rows_ = matrix.rows_;
        this->cols_ = matrix.cols_;
        this->stride_ = matrix.stride_;
        this->data_ = matrix.data_;
        this->alloc_size_ = 0;
    }
};

#endif
