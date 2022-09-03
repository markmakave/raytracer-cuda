#pragma once
#include <cuda_runtime.h>

#include "dim.cuh"
#include "hitable/hitable.cuh"
#include "optional.cuh"

namespace lm {

///////////////////////////////////////////////////////////////////////////

class Sphere : public Hitable {

    float radius;

public:

    //
    //  Sphere default constructor
    //  
    __host__ __device__
    Sphere();

    //
    //  Sphere conversion constructor
    //
    __host__ __device__
    Sphere(const dim& position, float radius);

    //
    //  CUDA transfering method for Object inheritance
    //
    __host__
    Sphere* 
    replicate() const override;

    //
    //  Intersection method for Hitable inheritance
    //
    __host__ __device__
    optional<Hitable::Impact> 
    intersect(const Ray& ray) const override;

};

///////////////////////////////////////////////////////////////////////////

}
