#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace lm {

struct rgba {

    uint8_t r, g, b, a;

    __host__ __device__ 
    rgba(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255) {
        this->r = r;
        this->g = g;
        this->b = b;
        this->a = a;
    }

};

}