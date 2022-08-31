#include <iostream>
#include <cuda_runtime.h>

#include <png++/png.hpp>

#include "map.cuh"
#include "rgba.cuh"
#include "camera.cuh"
#include "dim.cuh"
#include "timer.h"

#include "sphere.cuh"
#include "triangle.cuh"
#include "mesh.cuh"

#include "kernel.cuh"

using namespace lm;

int main() {
    
    Camera camera(
        { 5.0, 0.0, 0.0 }, 
        { 0.0, 0.0, 0.0 },
        3840, 2160, 45);

    map<rgba> frame;

    {
        Timer timer("render");
        frame = render(camera);
    }

    png::image<png::rgba_pixel> png(frame.width(), frame.height());

    {
        Timer timer("copy");
        #pragma omp parallel for
        for (int y = 0; y < frame.height(); ++y) {
            #pragma omp parallel for
            for(int x = 0; x < frame.width(); ++x) {
                rgba color = frame(x, y);

                png[y][x].red   = color.r;
                png[y][x].green = color.g;
                png[y][x].blue  = color.b;
                png[y][x].alpha = color.a;
                
            }
        }
    }
    

    {
        Timer timer("save");
        png.write("frame.png");
    }

    if (cudaError error = cudaGetLastError()) {
        std::cout << cudaGetErrorString(error) << std::endl;
    }
    
    return 0;
}
