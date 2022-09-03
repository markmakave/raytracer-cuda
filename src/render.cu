#include <cuda_runtime.h>

#include "render.cuh"

#include "rgba.cuh"
#include "object.cuh"
#include "map.cuh"
#include "dim.cuh"
#include "camera.cuh"

#include "hitable/sphere.cuh"
#include "hitable/triangle.cuh"
#include "hitable/mesh.cuh"
#include "hitable/world.cuh"

namespace lm {

//
//  Render kernel
//
__global__ 
static void _render(map<rgba>* frame, const Camera* camera, Hitable* object) {

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= frame->width() || y >= frame->height()) return;

    Ray ray = camera->cast(x, y);

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

//
//  CUDA cudaMalloc allocator wrapper for map
//
template <typename T>
class cuda_allocator {
    
    __host__ __device__
    cuda_allocator() {};
    
public:

    __host__
    static T* allocate(size_t size) {
        if (size == 0) return nullptr;
        T* devptr;
        cudaMalloc((void**)&devptr, size * sizeof(T));
        return devptr;
    }

    __host__
    static void deallocate(T* ptr, size_t size) {
        cudaFree(ptr);
    }
};

//
//  Render kernel wrapper for host
//
__host__
map<rgba> render(const Camera& camera, const Hitable& object) {
    map<rgba> frame(camera.width(), camera.height());

    Camera* dev_camera = camera.replicate();
    Hitable* dev_object  = object.replicate();

    map<rgba>* dev_frame;
    cudaMalloc((void**)&dev_frame, sizeof(frame));

    map<rgba, cuda_allocator<rgba>> _frame(frame.width(), frame.height());
    cudaMemcpy(dev_frame, &_frame, sizeof(frame), cudaMemcpyHostToDevice);

    int tx = 8, ty = 8;
    dim3 blocks(frame.width() / tx + 1, frame.height() / ty + 1), threads(tx, ty);
    _render <<<blocks, threads>>> (dev_frame, dev_camera, dev_object);
    cudaDeviceSynchronize();

    if (cudaError error = cudaGetLastError()) {
        std::cout << cudaGetErrorString(error) << std::endl;
        cudaDeviceReset();
        throw;
    }

    cudaMemcpy(frame.data(), _frame.data(), frame.size() * sizeof(rgba), cudaMemcpyDeviceToHost);
    
    cudaFree(dev_camera);
    cudaFree(dev_object);
    cudaFree(dev_frame);

    cudaDeviceReset();

    return frame;
}

}
