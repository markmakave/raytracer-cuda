#include "hitable/mesh.cuh"

namespace lm {

__global__
void build_mesh(Mesh* ptr, int ntriangles, Triangle* triangles) {
    new(ptr) Mesh(ntriangles, triangles);
}

Mesh::Mesh() : ntriangles(0), triangles(nullptr) {}

Mesh::Mesh(int ntriangles, Triangle* triangles) : ntriangles(ntriangles), triangles(triangles) {}

Mesh::Mesh(const std::string& filename) {
    std::ifstream file(filename, std::ios_base::binary);
    if (!file) throw;

    file.ignore(80);

    uint32_t ntriangles;
    file.read((char*)&ntriangles, sizeof(ntriangles));
    this->ntriangles = ntriangles;
    this->triangles = new Triangle[ntriangles];

    for (uint32_t i = 0; i < ntriangles; ++i) {
        Triangle& trg = triangles[i];
        file.ignore(12);
        for (int j = 0; j < 3; ++j) {
            file.read((char*)&trg[j], sizeof(trg[j]));
        }
        file.ignore(2);
    }

    file.close();
}

Mesh* 
Mesh::replicate() const {
    Mesh* devptr;
    cudaMalloc((void**)&devptr, sizeof(*this));

    Triangle* dev_triangles;
    cudaMalloc((void**)&dev_triangles, ntriangles * sizeof(Triangle));
    cudaMemcpy(dev_triangles, triangles, ntriangles * sizeof(Triangle), cudaMemcpyHostToDevice);

    build_mesh <<<1,1>>> (devptr, ntriangles, dev_triangles);
    cudaDeviceSynchronize();

    return devptr;
}

optional<Hitable::Impact>
Mesh::intersect(const Ray& ray) const {
    optional<Hitable::Impact> impact;
    for (int i = 0; i < ntriangles; ++i) {
        auto trg = triangles[i];
        auto possible_impact = trg.intersect(ray);
        if (possible_impact) {
            Hitable::Impact prior = possible_impact.value();
            if (!impact or impact.value().distance > prior.distance) {
                impact = prior;
            }
        }
    }

    return impact;
}

}
