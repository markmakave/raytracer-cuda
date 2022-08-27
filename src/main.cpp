#include <iostream>
#include <cuda_runtime.h>

#include "map.cuh"
#include "rgba.cuh"
#include "camera.cuh"
#include "object.cuh"
#include "hitable.cuh"
#include "dim.cuh"
#include "sphere.cuh"

using namespace lm;

int main() {
    
    Sphere sphere({ 0, 0, 0 }, 1);

    Ray ray({-5, 0, 0}, {0, 1, 0});

    std::cout << sphere.intersect(ray).has_value() << std::endl;
    
    return 0;
}
