#pragma once
#include <cuda_runtime.h>

#include "object.cuh"
#include "dim.cuh"

namespace lm {

class Ray : public Object {
public:

    dim direction;

    __host__ __device__
    Ray(const dim& origin, const dim& target) : Object(origin), direction((target - origin).normal()) {}

    //
    //  CUDA transfering method for Object inheritance
    //  Currently not needed
    //  TODO
    //
    __host__
    Ray* replicate() const {
        return nullptr;
    }

};

}
