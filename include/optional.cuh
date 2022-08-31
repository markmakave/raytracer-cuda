#pragma once
#include <cuda_runtime.h>

namespace lm {

template <typename T>
class optional {

    T _value;
    bool _valid;

public:

    __host__ __device__
    optional() : _valid(false) {
        
    }

    __host__ __device__
    optional(const T& value) : _value(value), _valid(true) {}

    __host__ __device__
    operator bool() const {
        return _valid;
    }

    __host__ __device__
    operator T() const {
        return _value;
    }

    __host__ __device__
    T& value() {
        return _value;
    }

    __host__ __device__
    const T& value() const {
        return _value;
    }

    __host__ __device__
    bool has_value() const {
        return _valid;
    }

};

}
