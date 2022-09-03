#pragma once
#include <iostream>

namespace lm {

template <typename T>
class allocator {
        
    __host__
    allocator() {};
    
public:

    __host__ __device__
    static T* allocate(size_t size) {
        if (size == 0) return nullptr;
        return reinterpret_cast<T*>(operator new(size * sizeof(T)));
    }

    __host__ __device__
    static void deallocate(T* ptr, size_t size) {
        operator delete(ptr);
    }
};

template <typename T, typename Allocator = allocator<T>>
class map {
protected:

    int _width, _height;
    T* _data;

public:

    __host__ __device__
    map(int width = 0, int height = 0)
        : _width(width), _height(height) {
        _data = Allocator::allocate(size());
    }

    __host__ __device__
    map(const map& m)
        : _width(m._width), _height(m._height) {
        _data = Allocator::allocate(size());
        for (size_t i = 0; i < size(); ++i) {
            _data[i] = m._data[i];
        }
    }

    __host__ __device__
    map(map&& m) 
        : _width(m._width), _height(m._height) {
        _data = m._data;
        m._data = nullptr;
    }

    __host__ __device__
    ~map() {
        Allocator::deallocate(_data, size());
    }

    __host__ __device__
    map& operator = (const map& m) {
        if (&m != this) {
            Allocator::deallocate(_data, size());
            _width = m._width;
            _height = m._height;
            _data = Allocator::allocate(size());
            for (size_t i = 0; i < size(); ++i) {
                _data[i] = m._data[i];
            }
        }
        return *this;
    }

    __host__ __device__
    map& operator = (map&& m) {
        if (&m != this) {
            Allocator::deallocate(_data, size());
            _width = m._width;
            _height = m._height;
            _data = m._data;
            m._data = nullptr;
        }
        return *this;
    }

    __host__ __device__
    T& operator [] (size_t index) {
        return _data[index];
    }
    __host__ __device__
    T operator [] (size_t index) const {
        return _data[index];
    }

    __host__ __device__
    T& operator () (int x, int y) {
        return _data[y * _width + x];
    }
    __host__ __device__
    T operator () (int x, int y) const {
        return _data[y * _width + x];
    }

    __host__ __device__
    size_t size() const {
        return _width * _height;
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
    T& at(int x, int y) {
        if (x >= 0 && y >= 0 && x < _width && y < _height) {
            return this->operator()(x, y);
        }
        static T trash_bin = {0};
        return trash_bin;
    }

    __host__ __device__
    T at(int x, int y) const {
        if (x >= 0 && y >= 0 && x < _width && y < _height) {
            return this->operator()(x, y);
        }
        return {0};
    }

    __host__ __device__
    void resize(int width, int height) {
        Allocator::deallocate(_data, size());
        _width = width;
        _height = height;
        _data = Allocator::allocate(size());
    }

};

}
