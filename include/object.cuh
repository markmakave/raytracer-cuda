#pragma once
#include <cuda_runtime.h>

#include "dim.cuh"

namespace lm {

//
//  Abstract class
//  represents object in the 3D scene
//
class Object {
public:

    dim position;

    __host__ __device__
    Object(const dim& position = { 0.0, 0.0, 0.0 }) : position(position) {}

    // __host__ __device__
    // virtual void move(const dim& offset) = 0;

    // __host__ __device__
    // virtual void rotate(const dim& axis, const float angle) = 0;

};
    
}
