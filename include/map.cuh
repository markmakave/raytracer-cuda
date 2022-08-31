#pragma once
#include <cuda_runtime.h>
#include <cstring>

namespace lm {

template <typename T>
class map {

    int _width, _height;
    T* _data;
    bool custom = false;

public:

    __host__ __device__
    map() : _width(0), _height(0), _data(nullptr) {}

    __host__ __device__ 
    map(int width, int height) : _width(width), _height(height), _data(new T[width * height]) {}

    __host__ __device__ 
    map(int width, int height, const T& value) : _width(width), _height(height) {
        this->fill(value);
    }

    __host__ 
    map(int width, int height, T* data) : _width(width), _height(height), _data(data), custom(true) {}

    __host__ __device__
    map(const map& m) : _width(m._width), _height(m._height), _data(new T[m.size()]) {
        std::memcpy(_data, m._data, m.size() * sizeof(T));
    }

    __host__ __device__
    map(map&& m) : _width(m._width), _height(m._height), _data(m._data) {
        m._data = nullptr;
    }

    __host__ __device__
    map& operator= (const map& m) {
        _width = m._width;
        _height = m._height;

        _data = new T[this->size()];
        // std::memcpy(_data, m._data, m.size() * sizeof(T));

        return *this;
    }

    __host__ __device__
    map& operator= (map&& m) {
        this->~map();

        _width = m._width;
        _height = m._height;

        _data = m._data;
        m._data = nullptr;

        return *this;
    }

    __host__ __device__
    int width() const {
        return _width;
    }

    __host__ __device__
    int height() const {
        return _height;
    }

    __host__ __device__
    T* data() const {
        return _data;
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

    __host__ __device__
    void resize(int width, int height) {
        this->~map();
        _width = width;
        _height = height;
        _data = new T[this->size()];
    }

    __host__ __device__
    ~map() {
        if (!custom) delete[] _data;
    }

};

}