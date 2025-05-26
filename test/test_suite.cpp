#include "test.h"
#include "pathtracer.cuh"

#include "third_party/helper_math.h"
#include "scan.cuh"

#include <iostream>

T_FUNC(test_computGI){
    std::vector<Color> fdata(512*512);
    Framebuffer framebuffer = {512, 512, fdata.data()};
    CpuMesh cpuMesh = CpuMesh::fromObj("sample/cornell.obj", framebuffer.size);
    cpuMesh.materials[12].light = true;

    computeLightmap(cpuMesh, framebuffer);

    framebuffer.save("test/output/path.png");
}