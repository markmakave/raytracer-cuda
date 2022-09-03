#pragma once
#include <cuda_runtime.h>

#include "dim.cuh"
#include "map.cuh"
#include "camera.cuh"
#include "rgba.cuh"

#include "hitable/world.cuh"
#include "hitable/sphere.cuh"
#include "hitable/triangle.cuh"
#include "hitable/mesh.cuh"

namespace lm {

__host__
map<rgba> render(const Camera& camera, const Hitable& world);

}
