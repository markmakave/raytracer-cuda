#include "hitable/sphere.cuh"

#include <array>

namespace lm {

///////////////////////////////////////////////////////////////////////////

//
//  Sphere building kernel
//  locally used
//
__global__
void build_sphere(Sphere* ptr, dim pos, float radius) {
    new(ptr) Sphere(pos, radius);
}

///////////////////////////////////////////////////////////////////////////

Sphere::Sphere() {}

Sphere::Sphere(const dim& position, float radius) : Hitable(position), radius(radius) {}

Sphere* 
Sphere::replicate() const {
    Sphere* devptr;
    cudaMalloc((void**)&devptr, sizeof(*this));
    
    build_sphere <<<1,1>>> (devptr, position, radius);
    cudaDeviceSynchronize();

    return devptr;
}

optional<Hitable::Impact> 
Sphere::intersect(const Ray& ray) const {
    dim oc = ray.position - this->position;
    float a = dim::dot(ray.direction, ray.direction);
    float b = 2.0 * dim::dot(oc, ray.direction);
    float c = dim::dot(oc, oc) - radius * radius;
    float descriminant = b * b - 4.0 * a * c;

    if (descriminant < 0) {
        return {};
    }

    Hitable::Impact impact;

    if (descriminant > 0) {
        float t1 = (-b - sqrt(descriminant)) / (2 * a), 
                t2 = (-b + sqrt(descriminant)) / (2 * a);

        if (t2 < t1) {
            float temp = t1;
            t1 = t2;
            t2 = temp;
        }

        dim position = ray.position + ray.direction * t1;
        impact.normal = (position - this->position).normal();
        impact.distance = t1;

    } else {
        // TODO (tangent case)
    }

    return impact;
}

///////////////////////////////////////////////////////////////////////////

}
