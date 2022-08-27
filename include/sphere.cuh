#pragma once
#include <cuda_runtime.h>

#include "dim.cuh"
#include "object.cuh"
#include "hitable.cuh"

namespace lm {

class Sphere : public Hitable {

    float radius;

public:

    __host__ __device__
    Sphere(const dim& position, float radius) : Hitable(position), radius(radius) {}

    __host__ __device__
    void move(const dim& offset) {}

    __host__ __device__
    void rotate(const dim& axis, float angle) {}

    __host__ __device__
    std::optional<Hitable::Impact> intersect(const Ray& ray) const {
        dim oc = ray.position - this->position;
        float a = dim::dot(ray.direction, ray.direction);
        float b = 2.0 * dim::dot(oc, ray.direction);
        float c = dim::dot(oc, oc) - radius * radius;
        float descriminant = b * b - 4.0 * a * c;
        if (descriminant > 0) return Hitable::Impact({ 0.0, 0.0, 0.0});
        return {};
    }

};

}
