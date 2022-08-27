#pragma once
#include <cuda_runtime.h>
#include <optional>

#include "dim.cuh"
#include "object.cuh"

namespace lm {

class Ray : public Object {
public:

    dim direction;

    __host__ __device__
    Ray(const dim& origin, const dim& direction) : Object(origin), direction(direction.normal()) {}

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
        dim position;

        __host__ __device__
        Impact(const dim& position) : position(position) {}
    };

    __host__ __device__
    Hitable(const dim& position = { 0.0, 0.0, 0.0 }) : Object(position) {}

    __host__ __device__
    virtual std::optional<Impact> intersect(const Ray& ray) const = 0;

};

}
