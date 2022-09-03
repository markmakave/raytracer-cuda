#pragma once
#include <cuda_runtime.h>

#include "dim.cuh"
#include "object.cuh"
#include "hitable.cuh"
#include "optional.cuh"

namespace lm {

class Triangle : public Hitable {
public:
    dim vertices[3];

public:

    //
    //  Triangle default constructor
    //
    __host__ __device__
    Triangle();

    //
    //  Triangle based on given vertices
    //
    __host__ __device__
    Triangle(const dim& v1, const dim& v2, const dim& v3);

    //
    //  CUDA transfering method for Object inheritance
    //  currently not used
    //
    __host__
    Triangle* replicate() const;

    //
    //  Intersection method for Hitable inheritance
    //
    __host__ __device__
    optional<Hitable::Impact> intersect(const Ray& ray) const;

    //
    //  Triangle normal calculator
    //
    __host__ __device__
    dim normal() const ;

    //
    //  Constant triangle vertex getter
    //
    __host__ __device__
    dim operator[] (int index) const;

    //
    //  Non-constant triangle vertex getter/setter
    //
    __host__ __device__
    dim& operator[] (int index); 

};

}
