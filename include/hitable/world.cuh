#pragma once
#include <cuda_runtime.h>
#include <cstring>

#include "object.cuh"
#include "hitable.cuh"

#include "sphere.cuh"
#include "triangle.cuh"
#include "mesh.cuh"

#include "optional.cuh"

#include <ostream>

namespace lm {

class World;

__global__ 
void build_world(World* ptr, int size, Hitable** objs);

class World : public Hitable {

    int _size;
    Hitable** _objs;

public:

    //
    //  World default constructor
    //
    __host__ __device__
    World() : _size(0), _objs(nullptr) {}

    //
    //  Device object instance constructor
    //
    __device__
    World(int size, Hitable** objs) : _size(size), _objs(objs) {}

    //
    //  CUDA transfering method for Object inheritance
    //
    __host__
    World* replicate () const {
        World* devptr;
        cudaMalloc((void**)&devptr, sizeof(*devptr));

        Hitable** devobjptrs = new Hitable*[_size];
        for (int i = 0; i < _size; ++i) {
            devobjptrs[i] = _objs[i]->replicate();
        }

        Hitable** devobjs;
        cudaMalloc((void**)&devobjs, _size * sizeof(*devobjs));
        cudaMemcpy(devobjs, devobjptrs, _size * sizeof(*devobjs), cudaMemcpyHostToDevice);
        delete[] devobjptrs;

        World temp;
        temp._size = _size;
        temp._objs = devobjs;
        cudaMemcpy(devptr, &temp, sizeof(temp), cudaMemcpyHostToDevice);

        return devptr;
    }

    //
    //  Intersection method for Hitable inheritance
    //
    __host__ __device__ 
    optional<Hitable::Impact> intersect(const Ray& ray) const {
        optional<Hitable::Impact> impact;

        for (int i = 0; i < _size; ++i) {
            auto obj = _objs[i];
            auto possible_impact = obj->intersect(ray);

            if (possible_impact) {
                auto prior = possible_impact.value();
                if (!impact || prior.distance < impact.value().distance) {
                    impact = prior;
                }
            }
        }

        return impact;
    }

    //
    //  World adding method
    //  adds Hitable objest to world list
    //
    template <typename T>
    __host__ 
    void add(const T& obj) {
        Hitable** new_objs = new Hitable*[_size + 1];
        std::memcpy(new_objs, _objs, sizeof(Hitable*) * _size);
        new_objs[_size] = new T;
        *new_objs[_size] = obj;
        delete[] _objs;
        _objs = new_objs;
        _size++;
    }

    __host__ __device__
    int& size() {
        return _size;
    }

    __host__ __device__
    Hitable**& objs() {
        return _objs;
    }

    __host__ __device__
    Hitable* operator[] (int index) const {
        return _objs[index];
    }

};

}
