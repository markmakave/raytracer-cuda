#pragma once
#include <cuda_runtime.h>

namespace lm {

template <typename T>
class map {

    int _width, _height;
    T* _data;

public:

    __host__ 
    map() : _width(0), _height(0), _data(nullptr) {}

    __host__ 
    map(int width, int height) : _width(width), _height(height), _data(new T[width * height]) {}

    __host__ 
    map(int width, int height, const T& value) : _width(width), _height(height) {
        this->fill(value);
    }

    __host__ __device__  
    int size() const {
        return _width * _height;
    }

    __host__ __device__ 
    const T& operator[] (int index) const {
        return _data[index];
    }

    __host__ __device__ 
    const T& operator() (int x, int y) const {
        return _data[y * _width + x];
    }

    __host__ __device__ 
    T& operator[] (int index) {
        return _data[index];
    }

    __host__ __device__ 
    T& operator() (int x, int y) {
        return _data[y * _width + x];
    }

    __host__ __device__ 
    void fill(const T& value) {
        int size = this->size();
        for (int i = 0; i < size; ++i) {
            _data[i] = value;
        }
    }

    __host__
    ~map() {
        delete[] _data;
    }

};

}