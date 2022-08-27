#pragma once
#include <cuda_runtime.h>

#include "dim.cuh"
#include "object.cuh"

namespace lm {

class Camera : public Object {

    dim direction;

public:

    void move(const dim& offset) {
        this->position += offset;
        this->direction += offset;
    }

    void rotate() {}

};

}