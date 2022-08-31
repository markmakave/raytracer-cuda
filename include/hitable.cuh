#pragma once
#include <cuda_runtime.h>

#include "dim.cuh"
#include "object.cuh"
#include "optional.cuh"

namespace lm {

class Ray : public Object {
public:

    dim direction;

    __host__ __device__
    Ray(const dim& origin, const dim& target) : Object(origin), direction((target - origin).normal()) {}

    // __host__ __device__
    // std::optional<Hitable::Impact> intersect(const Hitable& object) const;

};

//
//  Abstract class
//  represents hitable object in 3D scene
//
class Hitable : public Object {
public:

    struct Impact {
        float distance;
        dim normal;
    };

    __host__ __device__
    Hitable(const dim& position = { 0.0, 0.0, 0.0 }) : Object(position) {}

    __host__ __device__
    virtual optional<Impact> intersect(const Ray& ray) const = 0;

};

}
