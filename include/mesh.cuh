#pragma once
#include <cuda_runtime.h>

#include "dim.cuh"
#include "object.cuh"
#include "hitable.cuh"
#include "optional.cuh"
#include "triangle.cuh"

namespace lm {

class Mesh : public Hitable {

    int ntriangles;
    Triangle* triangles;

public:

    __host__ __device__
    Mesh() {

    }

    __host__ __device__
    optional<Hitable::Impact> intersect(const Ray& ray) const {
        optional<Hitable::Impact> impact;
        for (int i = 0; i < ntriangles; ++i) {
            auto possible_impact = triangles[i].intersect(ray);
            if (possible_impact) {
                Hitable::Impact prior = possible_impact.value();
                if (!impact or impact.value().distance > prior.distance) {
                    impact = prior;   
                }
            }
        }

        return impact;
    }

};

}
