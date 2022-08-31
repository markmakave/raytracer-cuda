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
    Object(const dim& position) : position(position) {}

};
    
}
