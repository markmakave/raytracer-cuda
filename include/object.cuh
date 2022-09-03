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

    //
    //  CUDA transfer method
    //  allocates and copies object to device
    //  returns device memory pointer
    //
    __host__    // Callable only from Host
    virtual     // Mush be implemented for each type of object
    Object* replicate() const = 0;

};
    
}
