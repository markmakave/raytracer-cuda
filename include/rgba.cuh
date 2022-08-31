#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace lm {

struct rgba {

    uint8_t r, g, b, a;

    __host__ __device__
    rgba() : r(0), g(0), b(0), a(0) {}

    __host__ __device__ 
    rgba(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255) : r(r), g(g), b(b), a(a) {}

};

inline
__host__ __device__
bool operator == (const rgba& c1, const rgba& c2) {
    return c1.r == c2.r && c1.g == c2.g && c1.b == c2.b && c1.a == c2.a;
}

}