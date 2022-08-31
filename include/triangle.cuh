#pragma once
#include <cuda_runtime.h>

#include "dim.cuh"
#include "object.cuh"
#include "hitable.cuh"
#include "optional.cuh"

namespace lm {

class Triangle : public Hitable {

    dim vertices[3];

public:

    __host__ __device__
    Triangle(const dim& v1, const dim& v2, const dim& v3) : Hitable((v1 + v2 + v3) / 3) {
        vertices[0] = v1; vertices[1] = v2; vertices[2] = v3;
    }

    __host__ __device__
    dim normal() const {
        return dim::cross(vertices[1] - vertices[0], vertices[2] - vertices[0]).normal();
    }

    __host__ __device__
    optional<Hitable::Impact> intersect(const Ray& ray) const {
        dim e1 = vertices[1] - vertices[0];
        dim e2 = vertices[2] - vertices[0];

        // Вычисление вектора нормали к плоскости
        dim pvec = dim::cross(ray.direction, e2);
        float det = dim::dot(e1, pvec);

        // Луч параллелен плоскости
        if (det < 1e-8 && det > -1e-8) {
            return {};
        }

        float inv_det = 1 / det;
        dim tvec = ray.position - vertices[0];
        float u = dim::dot(tvec, pvec) * inv_det;
        if (u < 0 || u > 1) {
            return {};
        }

        dim qvec = dim::cross(tvec, e1);
        float v = dim::dot(ray.direction, qvec) * inv_det;
        if (v < 0 || u + v > 1) {
            return {};
        }

        float distance = dim::dot(e2, qvec) * inv_det;

        Hitable::Impact impact;
        impact.distance = distance;
        impact.normal = this->normal();

        return impact;
    }

};

}
