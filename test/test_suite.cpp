#include "test.h"
#include "pathtracer.cuh"

#include "third_party/helper_math.h"
#include "scan.cuh"

#include <iostream>

T_FUNC(test_computGI){
    std::vector<int> tv = {1,2,4,5,6};
    std::vector<Color> fdata(512*512);
    Framebuffer framebuffer = {512, 512, fdata.data()};
    CpuMesh cpuMesh = CpuMesh::fromObj("sample/plane.obj", framebuffer.size);
    cpuMesh.materials[12].light = true;

    computeGI(cpuMesh, framebuffer);

    framebuffer.save("test/output/path.png");
}