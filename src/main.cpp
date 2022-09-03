#include <iostream>
#include <cuda_runtime.h>

#include <png++/png.hpp>

#include "map.cuh"
#include "rgba.cuh"
#include "camera.cuh"
#include "dim.cuh"
#include "timer.h"

#include "hitable/sphere.cuh"
#include "hitable/triangle.cuh"
#include "hitable/mesh.cuh"
#include "hitable/world.cuh"

#include "render.cuh"

using namespace lm;

int main() {

    Mesh mesh("../resource/cube.stl");
    Sphere sphere({0,0,0}, 1);

    Camera camera(
        { 5.0, 5.0, 2.0 }, 
        { 0.0, 0.0, 0.0 },
        4000, 4000, 30);

    map<rgba> frame;

    {
        Timer timer("render");
        frame = render(camera, sphere);
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
        
    return 0;
}
