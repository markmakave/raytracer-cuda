#pragma once
#include <cuda_runtime.h>

#include "dim.cuh"
#include "object.cuh"
#include "hitable.cuh"
#include "optional.cuh"

namespace lm {

class Sphere : public Hitable {

    float radius;

public:

    __host__ __device__
    Sphere(const dim& position, float radius) : Hitable(position), radius(radius) {}

    __host__ __device__
    optional<Hitable::Impact> intersect(const Ray& ray) const {
        dim oc = ray.position - this->position;
        float a = dim::dot(ray.direction, ray.direction);
        float b = 2.0 * dim::dot(oc, ray.direction);
        float c = dim::dot(oc, oc) - radius * radius;
        float descriminant = b * b - 4.0 * a * c;

        if (descriminant < 0) {
            return {};
        }

        Hitable::Impact impact;

        if (descriminant > 0) {
            float t1 = (-b - sqrt(descriminant)) / (2 * a), 
                  t2 = (-b + sqrt(descriminant)) / (2 * a);

            if (t2 < t1) {
                float temp = t1;
                t1 = t2;
                t2 = temp;
            }

            if (abs(t2 - t1) > radius) return {};

            dim position = ray.position + ray.direction * t1;
            impact.normal = (position - this->position).normal();
            impact.distance = t1;

        } else {

        }

        return impact;
    }

};

}
