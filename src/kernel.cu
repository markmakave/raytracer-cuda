#include <cuda_runtime.h>

#include "kernel.cuh"

#include "rgba.cuh"
#include "object.cuh"
#include "map.cuh"
#include "dim.cuh"
#include "camera.cuh"

#include "sphere.cuh"
#include "triangle.cuh"

namespace lm {

#define WORLD_SIZE 1

__global__
static void build(Hitable** world) {
    *world = new Triangle({-1, -1, 1}, {-1, 1, 1}, {1, 1, -1});
}

__global__ 
static void _render(map<rgba>* frame, const Camera* camera, Hitable** world) {

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= frame->width() || y >= frame->height()) return;

    Ray ray = camera->cast(x, y);

    auto object = *world;
    auto possible_impact = object->intersect(ray);

    if (possible_impact) {
        auto impact = possible_impact.value();

        dim normal = impact.normal;
        dim factor = (normal + dim(1, 1, 1)) / 2.0;

        (*frame)(x, y) = { uint8_t(255 * factor.x), uint8_t(255 * factor.y), uint8_t(255 * factor.z), 255 };

    } else { 
        (*frame)(x, y) = { 0, 0, 0, 0 };
    }
}

__host__
map<rgba> render(Camera camera) {
    map<rgba> frame(camera.width(), camera.height());

    map<rgba>* _frame;
    cudaMalloc((void**)&_frame, sizeof(*_frame));

    rgba* _data;
    cudaMalloc((void**)&_data, sizeof(*_data) * frame.size());

    map<rgba> temp(frame.width(), frame.height(), _data);
    cudaMemcpy(_frame, &temp, sizeof(temp), cudaMemcpyHostToDevice);

    Hitable** world;
    cudaMalloc((void**)&world, sizeof(*world) * WORLD_SIZE);
    build <<<1, 1>>>(world);
    cudaDeviceSynchronize();

    Camera* _camera;
    cudaMalloc((void**)&_camera, sizeof(*_camera));
    cudaMemcpy(_camera, &camera, sizeof(*_camera), cudaMemcpyHostToDevice);

    int tx = 8, ty = 8;
    dim3 blocks(frame.width() / tx + 1, frame.height() / ty + 1), threads(tx, ty);

    _render <<<blocks, threads>>>(_frame, _camera, world);
    cudaDeviceSynchronize();

    cudaMemcpy(frame.data(), _data, frame.size() * sizeof(rgba), cudaMemcpyDeviceToHost);

    return frame;
}

}