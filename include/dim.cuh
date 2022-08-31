#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

namespace lm {

struct dim {

    float x, y, z;

    __host__ __device__
    dim(float x = 0.0, float y = 0.0, float z = 0.0) : x(x), y(y), z(z) {}

    __host__ __device__
    float len() const {
        return sqrt(dot(*this, *this));
    }

    __host__ __device__
    dim normal() const {
        float l = this->len();
        return { x / l, y / l, z / l };
    }

    __host__ __device__
    dim reflect(const dim& normal) const {
        return *this - normal * 2.0 * dot(*this, normal);
    }


    __host__ __device__
    static float dot(const dim& v1, const dim& v2) {
        return v1.x * v2.x + v1.y *v2.y + v1.z * v2.z;
    }

    __host__ __device__
    static dim cross(const dim& v1, const dim& v2) {
        return { v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x };
    }

    __host__ __device__
    static float distance(const dim& v1, const dim& v2) {
    return abs((v1 - v2).len());
}


    __host__ __device__
    dim operator + () const {
        return *this;
    }

    __host__ __device__
    dim operator - () const {
        return { -x, -y, -z };
    }

    __host__ __device__
    dim operator + (const dim& v) const {
        return { x + v.x, y + v.y, z + v.z };
    }

    __host__ __device__
    dim operator - (const dim& v) const {
        return { x - v.x, y - v.y, z - v.z };
    }

    __host__ __device__
    dim operator * (const float f) const {
        return { x * f, y * f, z * f };
    }

    __host__ __device__
    dim operator / (const float f) const {
        return { x / f, y / f, z / f };
    }

    __host__ __device__
    dim& operator += (const dim& v) {
        return *this = *this + v;
    }

    __host__ __device__
    dim& operator -= (const dim& v) {
        return *this = *this - v;
    }

    __host__ __device__
    dim& operator *= (const float f) {
        return *this = *this * f;
    }

    __host__ __device__
    dim& operator /= (const float f) {
        return *this = *this / f;
    }

    friend std::ostream& operator << (std::ostream& out, const dim& v) {
        return out << "{ " << v.x << ", " << v.y << ", " << v.z << " }";
    }

};

}
