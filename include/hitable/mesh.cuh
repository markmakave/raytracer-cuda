#pragma once
#include <cuda_runtime.h>

#include <fstream>

#include "dim.cuh"
#include "object.cuh"
#include "hitable.cuh"
#include "optional.cuh"
#include "triangle.cuh"

namespace lm {

class Mesh : public Hitable {
public:
    int ntriangles;
    Triangle* triangles;

public:

    //
    //  Mesh default constructor
    //
    __host__ __device__
    Mesh();

    //
    //  Device object instance constructor
    //
    __host__ __device__
    Mesh(int ntriangles, Triangle* triangles);

    //
    //  Mesh filename based constructor
    //  reads 3D object data from STL file
    //
    __host__ 
    Mesh(const std::string& filename);

    //
    //  CUDA transfering method for Object inheritance
    //
    __host__
    Mesh* replicate() const;

    //
    //  Intersection method for Hitable inheritance
    //
    __host__ __device__
    optional<Hitable::Impact> intersect(const Ray& ray) const;

};

}
