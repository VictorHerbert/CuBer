#include "test.h"

#include "pathtracer.h"
#include "parser.h"

T_FUNC(test_pathtracer){
    auto mesh  = parse_obj("sample/cornell.obj");
    
    std::vector<Pixel> fdata(512*512);
    std::vector<float> fzbuffer(512*512, 1e9);
    Framebuffer framebuffer = {512, 512, fdata.data(), fzbuffer.data()};

    for(auto triangle : mesh){
        pathtraceTriangle(triangle, framebuffer);
    }

    framebuffer.save("test/output/path.png");
}