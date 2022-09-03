#pragma once
#include <cuda_runtime.h>

#include "dim.cuh"
#include "object.cuh"
#include "optional.cuh"
#include "ray.cuh"

namespace lm {

//
//  Abstract class
//  represents hitable object in 3D scene
//
class Hitable : public Object {
public:

    struct Impact {
        float distance = INFINITY;
        dim normal = { 0.0, 0.0, 0.0 };
    };

    //
    //  Hitable object construcor
    //  determines object position on 3D space
    //
    __host__ __device__
    Hitable(const dim& position = { 0.0, 0.0, 0.0 }) : Object(position) {}

    __host__ __device__
    virtual 
    optional<Impact> intersect(const Ray& ray) const = 0;

    __host__
    virtual
    Hitable* replicate() const = 0;

};

}
