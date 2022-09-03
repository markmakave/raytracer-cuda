#pragma once
#include <cuda_runtime.h>
#include <cmath>

#include "dim.cuh"
#include "object.cuh"
#include "ray.cuh"

namespace lm {

class Camera : public Object {

    int _width, _height;
    dim direction;
    dim corner;
    dim x_step, y_step;

public:

    //
    //  Camera constructor baseon on its position and target
    //  width and height in pixels and fov can also be provided
    //
    __host__ __device__
    Camera(const dim& position, const dim& target, int width = 640, int height = 480, float fov = 45.0) 
        : _width(width), _height(height), Object(position), direction((target - position).normal())
    {
        dim grid_center = this->position + direction;

        dim x_offset = dim::cross(direction, { 0.0, 0.0, 1.0 }).normal();
        dim y_offset = dim::cross(x_offset, direction).normal();

        float ratio = float(width) / height;
        x_offset *= tan(fov * M_PI / 360);
        y_offset *= x_offset.len() / ratio;
        corner = grid_center + x_offset + y_offset;

        x_step = -x_offset / (width / 2);
        y_step = -y_offset / (height / 2);
    }

    //
    //  CUDA transfering method for Object inheritance
    //
    __host__
    Camera* replicate() const {
        Camera* devptr;
        cudaMalloc((void**)&devptr, sizeof(*this));
        cudaMemcpy(devptr, this, sizeof(*this), cudaMemcpyHostToDevice);
        return devptr;
    }

    //
    //  Camera screen width getter
    //
    __host__ __device__
    int width() const {
        return _width;
    }

    //
    //  Camera screen height getter
    //
    __host__ __device__
    int height() const {
        return _height;
    }

    //
    //  Camera ray casting method
    //  returns Ray object based on (x, y) coordinates on camera screen
    //
    __host__ __device__ 
    Ray cast(int x, int y) const {
        return Ray(this->position, corner + x_step * x + y_step * y);
    }

};

}