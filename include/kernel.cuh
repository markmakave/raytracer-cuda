#pragma once
#include <cuda_runtime.h>

#include "map.cuh"
#include "camera.cuh"
#include "rgba.cuh"

namespace lm {

__host__
map<rgba> render(Camera camera);

}
